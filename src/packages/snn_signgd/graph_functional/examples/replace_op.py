import torch
from torch import nn, fx
import operator, copy
from functional_fx_utils import draw_computational_graph, check_neural_equivalence

"""
How to Replace One Op With Another

1. Iterate through all Nodes in your GraphModule's Graph.
2. Determine if the current Node should be replaced. (Suggested: match
on the Node's ``target`` attribute).
3. Create a replacement Node and add it to the Graph.
4. Use the FX built-in ``replace_all_uses_with`` to replace all uses of
the current Node with the replacement.
5. Delete the old Node from the graph.
6. Call ``recompile`` on the GraphModule. This updates the generated
Python code to reflect the new Graph state.

Currently, FX does not provide any way to guarantee that replaced
operators are syntactically valid. It's up to the user to confirm that
any new operators will work with the existing operands.

The following code demonstrates an example of replacing any instance of
addition with a bitwise AND.

To examine how the Graph evolves during op replacement, add the
statement `print(traced.graph)` after the line you want to inspect.
Alternatively, call `traced.graph.print_tabular()` to see the IR in a
tabular format.
"""

def replace_op(model: torch.nn.Module, inplace:bool=False) -> torch.nn.Module:
    patterns = set([operator.add, torch.add, "add"])
    
    if not inplace: model = copy.deepcopy(model)
    trace = fx.symbolic_trace(model)
    if not inplace: 
        graph = copy.deepcopy(trace.graph)
    else:
        graph = trace.graph
        
    for node in graph.nodes:
        if any(node.target == pattern for pattern in patterns):
            with graph.inserting_after(node):
                new_node = graph.call_function(torch.mul, node.args, node.kwargs)
                node.replace_all_uses_with(new_node)        
            graph.erase_node(node)

    graph.lint()
    
    if not inplace: 
        model = fx.GraphModule(trace, graph)
    else:
        trace.recompile()
    
    return model

    
if __name__ == "__main__":
    class MyModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.param = nn.Parameter(torch.rand(3, 4))
            self.linear = nn.Linear(4, 5)
    
        def forward(self, x):
            return self.linear(torch.add(x, self.param)).clamp(min=0.0, max=1.0)
            
    old_model = MyModule()
    new_model = replace_op(old_model, inplace = False)
    
    input_shape = (1,3,4)
    
    check_neural_equivalence(modules = [old_model, new_model], input_shape = input_shape)

    draw_computational_graph("./example_replace_op_old", old_model, input_shape)
    draw_computational_graph("./example_replace_op_new", new_model, input_shape)