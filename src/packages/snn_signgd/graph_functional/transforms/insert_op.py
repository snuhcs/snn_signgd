from torch import fx
from typing import Callable, Any
from ..utils import get_parent_name

def insert_op(dest_module: Callable):
    def insert_op_node_transform(node:fx.Node, trail, pattern, graph, modules:dict[Any]):
        if node.op == 'call_function':
            parent_name = ''
        elif node.op == 'call_module':
            parent_name, name = get_parent_name(node.target)
    
        node_prev = trail[-2]
        new_name = "mid_"+ node_prev.name + node.name
        
        if parent_name == '' :
            module_path = new_name
        else:
            module_path = '.'.join((parent_name,new_name))
                
        assert dest_module not in pattern # To avoid infinite recursion
        new_module = dest_module()
        
        #named_modules[new_name] = new_module
        setattr(modules[parent_name], new_name, new_module)
        
        with graph.inserting_before(node):
            new_node = graph.call_module(module_path, (node_prev,))
        #for idx, nodearg in enumerate(node.args):
        #    if nodearg == node_prev:
        #        node.argsnode.args[idx] = new_node
        node.replace_input_with(node_prev,new_node)     
    return insert_op_node_transform