import torch
from torch import fx
import torch.nn.functional as F

from typing import Callable, Union, Tuple, Iterable, Dict, Any, Type
import types

def get_parent_name(target : str) -> Tuple[str, str]:
    """
    Splits a qualname into parent path and last atom.
    For example, `foo.bar.baz` -> (`foo.bar`, `baz`)
    """
    *parent, name = target.rsplit('.', 1)
    return parent[0] if parent else '', name
    
def replace_node_module(node: fx.Node, modules: Dict[str, Any], new_module: torch.nn.Module):
    assert(isinstance(node.target, str))
    parent_name, name = get_parent_name(node.target)
    new_name = node.name
    modules[new_name] = new_module
    setattr(modules[parent_name], name, new_module)
    return new_name

def get_inputs_and_outputs(graph: fx.Graph):
    inputs, outputs = [], []
    for node in graph.nodes:
        if node.op == 'placeholder':
            inputs.append(node)
        elif node.op == 'output':
            outputs.append(node)
    return inputs, outputs

    
class ShapeProp:
    """
    Shape propagation. This class takes a `GraphModule`.
    Then, its `propagate` method executes the `GraphModule`
    node-by-node with the given arguments. As each operation
    executes, the ShapeProp class stores away the shape and
    element type for the output values of each operation on
    the `shape` and `dtype` attributes of the operation's
    `Node`.
    """
    def __init__(self, mod):
        self.mod = mod
        self.graph = mod.graph
        self.modules = dict(self.mod.named_modules())

    def propagate(self, *args):
        args_iter = iter(args)
        env : Dict[str, Node] = {}

        def load_arg(a):
            return torch.fx.graph.map_arg(a, lambda n: env[n.name])

        def fetch_attr(target : str):
            target_atoms = target.split('.')
            attr_itr = self.mod
            for i, atom in enumerate(target_atoms):
                if not hasattr(attr_itr, atom):
                    raise RuntimeError(f"Node referenced nonexistant target {'.'.join(target_atoms[:i])}")
                attr_itr = getattr(attr_itr, atom)
            return attr_itr

        for node in self.graph.nodes:
            if node.op == 'placeholder':
                result = next(args_iter)
            elif node.op == 'get_attr':
                result = fetch_attr(node.target)
            elif node.op == 'call_function':
                result = node.target(*load_arg(node.args), **load_arg(node.kwargs))
            elif node.op == 'call_method':
                self_obj, *args = load_arg(node.args)
                kwargs = load_arg(node.kwargs)
                result = getattr(self_obj, node.target)(*args, **kwargs)
            elif node.op == 'call_module':
                result = self.modules[node.target](*load_arg(node.args), **load_arg(node.kwargs))

            # This is the only code specific to shape propagation.
            # you can delete this `if` branch and this becomes
            # a generic GraphModule interpreter.
            if isinstance(result, torch.Tensor):
                node.shape = result.shape
                node.dtype = result.dtype

            env[node.name] = result

        return load_arg(self.graph.result)

if __name__ == "__main__":
    from tqdm import tqdm
    from torch import nn
    # Simple module for demonstration
    class MyModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.param = nn.Parameter(torch.rand(3, 4))
            self.linear = nn.Linear(4, 5)
    
        def forward(self, x):
            return self.linear(torch.add(x, self.param)).clamp(min=0.0, max=1.0)
    
    module = MyModule()
    
    draw_computational_graph("./example_model", module, (1,3,4))
    
    def transform(m: torch.nn.Module,
                  tracer_class : type = fx.Tracer) -> torch.nn.Module:
        graph : fx.Graph = tracer_class().trace(m)
        # FX represents its Graph as an ordered list of
        # nodes, so we can iterate through them.
        for node in tqdm(graph.nodes):
            print("Node info:", node.__dict__)
            # Checks if we're calling a function (i.e:
            # torch.add)
            
            if node.op == 'call_function':
                # The target attribute is the function
                # that call_function calls.
                if node.target == torch.add:
                    node.target = torch.mul
            if node.target != torch.relu and node.op == 'call_method':
    
                # Specifies the insertion point. Any nodes added to the
                # Graph within this scope will be inserted after `node`
                with graph.inserting_after(node):
                    # Insert a new `call_function` node calling `torch.relu`
                    new_node = graph.call_function(
                        torch.relu, args=(node,))
                
                    # We want all places that used the value of `node` to
                    # now use that value after the `relu` call we've added.
                    # We use the `replace_all_uses_with` API to do this.
                    node.replace_all_uses_with(new_node)
                    new_node.update_arg(0,node) 
    
        graph.lint() # Does some checks to make sure the
                     # Graph is well-formed.
    
        return fx.GraphModule(m, graph)
    
    module_relu = transform(module)
    
    draw_computational_graph("./example_model_relu", module_relu, (1,3,4))
    
    def relu_decomposition(x):
        return (x > 0) * x
    
    decomposition_rules = {}
    decomposition_rules[torch.relu] = relu_decomposition
    
    def decompose(model: torch.nn.Module,
                  tracer_class : type = fx.Tracer) -> torch.nn.Module:
        """
        Decompose `model` into smaller constituent operations.
        Currently,this only supports decomposing ReLU into its
        mathematical definition: (x > 0) * x
        """
        graph : fx.Graph = tracer_class().trace(model)
        new_graph = fx.Graph()
        env = {}
        tracer = fx.proxy.GraphAppendingTracer(new_graph)
        for node in graph.nodes:
            if node.op == 'call_function' and node.target in decomposition_rules:
                # By wrapping the arguments with proxies,
                # we can dispatch to the appropriatef
                # decomposition rule and implicitly add it
                # to the Graph by symbolically tracing it.
                proxy_args = [
                    fx.Proxy(env[x.name], tracer) if isinstance(x, fx.Node) else x for x in node.args]
                output_proxy = decomposition_rules[node.target](*proxy_args)
    
                # Operations on `Proxy` always yield new `Proxy`s, and the
                # return value of our decomposition rule is no exception.
                # We need to extract the underlying `Node` from the `Proxy`
                # to use it in subsequent iterations of this transform.
                new_node = output_proxy.node
                env[node.name] = new_node
            else:
                # Default case: we don't have a decomposition rule for this
                # node, so just copy the node over into the new graph.
                new_node = new_graph.node_copy(node, lambda x: env[x.name])
                env[node.name] = new_node
        return fx.GraphModule(model, new_graph)
    
    module_relu_decomposed = decompose(module_relu)
    
    draw_computational_graph("./example_model_relu_decomposed", module_relu_decomposed, (1,3,4))

    model = module
    
    model_image = transform(model)

    model_imported = save_and_reimport_nn_module(model = model_image)()

    cases = [model, model_image, model_imported]

    check_neural_equivalence(modules = cases, input_shape = (1,3,4))

