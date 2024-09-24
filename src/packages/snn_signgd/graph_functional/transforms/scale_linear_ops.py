import torch
from torch import fx
from typing import Callable, Any
from ..utils import get_parent_name, get_inputs_and_outputs
import copy

def scale_linear_ops():
    def scale_linear_ops_node_transform(node:fx.Node, trail, pattern, graph, modules:dict[Any]):
        
        is_not_leaf_module = len(node.meta)
        if is_not_leaf_module: return
            
        if node.op == 'call_function':
            parent_name = ''
        elif node.op == 'call_module':
            parent_name, name = get_parent_name(node.target)
            
        new_name = node.name
        
        if parent_name == '' :
            module_path = new_name
        else:
            module_path = '.'.join((parent_name,new_name))

        module = getattr(modules[parent_name], name)
        assert hasattr(module, "activation_stats" )
        print("Is leaf module?:", fx.Tracer().is_leaf_module(module, module_path), node.meta, )
        max_act_in, max_act_out, min_act_in, min_act_out = None,None,None,None
        for key, stat in module.activation_stats.items():
            if 'amax' in key:
                if '0.' in key:
                    max_act_in = stat
                elif '-1.' in key:
                    max_act_out = stat
            elif 'amin' in key:
                if '0.' in key:
                    min_act_in = stat
                elif '-1.' in key:
                    min_act_out = stat
        assert max_act_out is not None and max_act_in is not None

        print("Node info on scaling op:", node.__dict__)
        graph_inputs, graph_outputs = get_inputs_and_outputs(graph)
        print("Graph input", graph_inputs, "Graph output", graph_outputs, "Node input", node.args, "Node output", node.users, list(node.users.keys()))
        new_module = weight_scaling(
            module, max_act_out, min_act_out, max_act_in, min_act_in, 
            is_input = set( node.args ).issubset( set(graph_inputs) ), 
            is_output = set( node.users.keys() ).issubset( set(graph_outputs) )
        )
        
        setattr(modules[parent_name], new_name, new_module)
        
        with graph.inserting_after(node):
            new_node = graph.call_module(module_path, node.args)
        node.replace_all_uses_with(new_node)        
        graph.erase_node(node)
        print("Names:", parent_name, new_name, module_path)
    return scale_linear_ops_node_transform


def weight_scaling(operator, max_act_out, min_act_out, max_act_in, min_act_in, is_input:bool, is_output:bool):
    #norm_coefficient = ((1.0 - 0.0) / (max_act_out - min_act_out)) * (max_act_in - min_act_in)
    nominator = (1.0 - 0.0) if is_input else (1.0 - 0.0) * max_act_in
    denominator = 1.0 if is_output else float(max_act_out)
    weight_coefficient = nominator / denominator
    bias_coefficient = 1.0 / denominator
    
    success("Scale linear ops factor:", weight_coefficient, bias_coefficient, "max out:", max_act_out, "max in", max_act_in)
    
    scaled_operator = copy.deepcopy(operator)

    if bias is None:
        bias = torch.zeros(weight.shape[:-1]).to(dtype = weight_dtype)
    scaled_operator.weight, scaled_operator.bias = \
        _scale_weights(
            operator.weight, operator.bias, scale_w = weight_coefficient , scale_b = bias_coefficient
        )

    return scaled_operator

def _scale_weights(weight, bias, scale_w:float, scale_b: float):
    weight_dtype = weight.dtype
    bias_dtype = bias.dtype if bias is not None else weight_dtype
    if bias is None:
        bias = torch.zeros(weight.shape[:-1]).to(dtype = bias_dtype)

    scaled_weight = (weight * scale_w).to(dtype = weight_dtype)
    scaled_bias = (bias * scale_b).to(dtype = bias_dtype)

    return torch.nn.Parameter(scaled_weight, weight.requires_grad), torch.nn.Parameter(scaled_bias, bias.requires_grad)