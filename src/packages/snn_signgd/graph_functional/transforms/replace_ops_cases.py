from torch import fx
from typing import Callable, Any, Iterable, Type
from ..utils import get_parent_name


def replace_ops_cases(
    dest_modules: Iterable[Callable], 
    cases: Iterable[Iterable[Type]],
    inherit_kwargs: Iterable[dict] = None
):
    def replace_ops_cases_node_transform(node:fx.Node, trail, pattern, graph, modules:dict[Any]):
        if node.op == 'call_function':
            parent_name, name = '', node.name
        elif node.op == 'call_module':
            parent_name, name = get_parent_name(node.target)
            
        new_name = node.name
        
        if parent_name == '' :
            module_path = new_name
        else:
            module_path = '.'.join((parent_name,new_name))

        assert any([pattern in case for case in cases])
        
        for index, case in enumerate(cases):
            if pattern in case:
                break
                
        dest_module = dest_modules[index]
        inherit_args = inherit_kwargs[index]
                
        assert dest_module not in pattern # To avoid infinite recursion
        #print("Pattern:", pattern, "dest_module:", dest_module)
        if inherit_args is None:
            new_module = dest_module()
        else:
            module = modules[node.target]
            kwargs = {arg_name:getattr(module, attr_name) for arg_name, attr_name in inherit_args.items()}
            new_module = dest_module(**kwargs)
        
        #named_modules[new_name] = new_module
        setattr(modules[parent_name], new_name, new_module)
        
        with graph.inserting_after(node):
            new_node = graph.call_module(module_path, node.args)
        node.replace_all_uses_with(new_node)        
        graph.erase_node(node)
        return new_node, module_path
    
    return replace_ops_cases_node_transform