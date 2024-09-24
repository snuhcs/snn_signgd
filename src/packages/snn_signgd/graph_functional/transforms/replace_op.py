from torch import fx
from typing import Callable, Any
import copy
import inspect

from ..utils import get_parent_name

from snn_signgd.pretty_printer import print

def args_to_kwargs(function, args):
    signature = inspect.signature(function)
    parameters = list(signature.parameters)
    #args = copy.deepcopy(args)
    return dict(zip(parameters, args)), parameters

def replace_op(dest_module: Callable, inherit_args: dict = None):
    def replace_op_node_transform(node:fx.Node, trail, pattern, graph, modules:dict[Any]):
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
        #print("Pattern:", pattern, "dest_module:", dest_module)
        if inherit_args is None:
            new_module = dest_module()
        else:
            if node.op == 'call_module':
                module = modules[node.target]
                kwargs = {arg_name:getattr(module, attr_name) for arg_name, attr_name in inherit_args.items()}
                new_module = dest_module(**kwargs)
            elif node.op == 'call_function':
                #print("Node:", node, node.__dict__,"Args:", node.args, "Kwargs:", node.kwargs)
                '''
                positional_args, parameters = args_to_kwargs(function = node.target, args = node.args)
                
                kwargs1 = node.kwargs
                kwargs2 = {
                    arg_name: positional_args[attr_name] 
                     for arg_name, attr_name in inherit_args.items() 
                     if arg_name not in kwargs1
                    }

                remaining_args = []
                for index, name in enumerate(parameters):
                    if name not in inherit_args.values():
                        remaining_args.append(node.args[index])
                args = tuple(remaining_args)
                        
                print("Positional Arguments:", positional_args, "Remaining args:", node.args)
                new_module = dest_module(**kwargs1, **kwargs2)
                '''
                remaining_args = {arg_name:None for arg_name, attr_name in inherit_args.items()}
                new_module = dest_module(**(remaining_args))
                
                #print("function -> module:", node.target, "->", new_module)
            else:
                raise NotImplementedError('replace_op graph functional does not support ' + str(node.op))
        
        #named_modules[new_name] = new_module
        setattr(modules[parent_name], new_name, new_module)
        
        with graph.inserting_after(node):
            #print("ARGS:", node.args, "KWARGS:", node.kwargs)
            if node.op == 'call_module':
                new_node = graph.call_module(module_path, args = node.args, kwargs = node.kwargs)
            elif node.op == 'call_function':
                new_node = graph.call_module(module_path, args = node.args, kwargs = node.kwargs)
            else:
                raise NotImplementedError('replace_op graph functional does not support ' + str(node.op))
            
        node.replace_all_uses_with(new_node)        
        graph.erase_node(node)
        return new_node, module_path
    return replace_op_node_transform