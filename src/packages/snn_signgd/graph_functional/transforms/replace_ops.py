from torch import fx
from typing import Callable, Any
from ..utils import get_parent_name

def replace_ops(dest_module: Callable):
    def replace_ops_node_transform(node:fx.Node, trail, pattern, graph, modules:dict[Any]):
        for node_elem in trail[:-1]:
            assert len(node_elem.users) == 1, "Assert subgraph is sequential forward"
        for node_elem in trail[1:]:
            assert len(node_elem.args) == 1, "Assert subgraph is sequential backward"
        
        if node.op == 'call_function':
            parent_name = ''
        elif node.op == 'call_module':
            parent_name, name = get_parent_name(node.target)

        new_name = node.name
        
        if parent_name == '' :
            module_path = new_name
        else:
            module_path = '.'.join((parent_name,new_name))
                
        assert dest_module not in pattern # To avoid infinite recursion
        new_module = dest_module()
        
        #named_modules[new_name] = new_module
        setattr(modules[parent_name], new_name, new_module)
        
        front_node = trail[0]
        with graph.inserting_before(front_node):
            if node.op == 'call_module':
                new_node = graph.call_module(module_path, front_node.args)
            elif node.op == 'call_function':
                new_node = graph.call_module(module_path, front_node.args, front_node.kwargs)
            else:
                raise NotImplementedError('replace_op graph functional does not support ' + str(node.op))
            
        node.replace_all_uses_with(new_node)
        front_node.replace_all_uses_with(new_node)
        
        for node_elem in trail:
            graph.erase_node(node_elem)
        print("Names:", parent_name, new_name, module_path)
    return replace_ops_node_transform