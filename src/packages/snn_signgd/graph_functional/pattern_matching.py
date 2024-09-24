import torch
from torch import nn, fx
import torch.nn.functional as F

from typing import Callable, Union, Tuple, Iterable, Dict, Any, Type, Set, List
import types
import copy

from snn_signgd.pretty_printer import print
from tqdm import tqdm


def walk_backward(node: fx.Node, length:int) -> List[Tuple[fx.Node]]:
    trails = traversal_pre(node, depth = 1, max_depth = length)
    output = []
    for trail in trails:
        if len(trail) == length:
            output.append(trail)
    return output
        
def traversal_pre(node: fx.Node, depth:int, max_depth:int) -> List[Tuple[fx.Node]]:
    if len(node.args)==0 or depth >= max_depth:
        return [(node,)]
    else:
        output = []
        for nodearg in node.args:
            if isinstance(nodearg, fx.Node):
                subtrails = traversal_pre(nodearg, depth = depth + 1, max_depth = max_depth)
                subtrails = [ subtrail + (node, ) for subtrail in subtrails]
                output.extend(subtrails)
        return output
    
def pattern_matching(pattern: Iterable[Type], nodes: Iterable[fx.Node], modules: Dict[str, Any], verbose: bool):
    assert len(nodes) == len(pattern)
    for expected_type, current_node in zip(pattern, nodes):
        if not isinstance(current_node, fx.Node):
            return False
        if current_node.op == 'call_module':
            if not isinstance(current_node.target, str) :
                return False
            if current_node.target not in modules:
                return False
            #print("Module:",modules[current_node.target], type(modules[current_node.target]), expected_type)
            if type(modules[current_node.target]) is not expected_type:
                return False
        elif current_node.op == 'call_function':
            #if verbose: print("Function:", current_node.target, type(current_node.target), repr(current_node.target), "Expected Type:", expected_type)
            if isinstance(current_node.target, types.BuiltinFunctionType):
                #if not isinstance(expected_type, str):
                #    return False
                #print("[Builtin] Pattern:", expected_type, "target:", current_node.target)
                if not all([keyword in str(current_node.target) 
                            for keyword in ['built-in'] + str(expected_type).split()
                           ]):
                    #if verbose: print("Keywords matching failed:", str(expected_type).split(), "not in", str(current_node.target) )
                    return False
                else:
                    if verbose: 
                        print(
                            "\nKeywords matching success:", str(expected_type).split(), "in", str(current_node.target) , '\n',
                            "[Builtin] Pattern:", expected_type, "target:", current_node.target,
                            "Function:", current_node.target, type(current_node.target), repr(current_node.target), 
                            "Expected Type:", expected_type
                        )

            elif callable(expected_type):
                if current_node.target != expected_type:
                    return False
            else:
                return False
        else:
            #if verbose: print("fxNode operator not supported for pattern matching:", current_node.op)
            return False            
    return True

def pattern_matching_transform(
        model: torch.nn.Module, patterns: Set[Any], 
        graph_transform: Callable, 
        inplace:bool=False,
        verbose = False
    ) -> torch.nn.Module:

    print(
        #"-"*30 + "\n"
        "<red>[Pattern Matching Graph Transform]</red> ", "Inplace:", inplace,'\n'
        "Graph Transform: <magenta>", graph_transform.__name__,'</magenta>\n'
        "Patterns:", str(patterns).replace('<',"") # for loguru color markup
    )
    
    if not inplace: model = copy.deepcopy(model)
    trace = fx.symbolic_trace(model)
    if not inplace: 
        graph = copy.deepcopy(trace.graph)
    else:
        graph = trace.graph

    named_modules = dict(trace.named_modules())

    log_transform = []
    for pat_index, pattern in enumerate(patterns):
        for node in tqdm(graph.nodes, desc = f"[{pat_index + 1}/{len(patterns)}] pattern: {str(pattern)}"):
            
            backward_trails = walk_backward(node, length = len(pattern))
            #if verbose: print("Backward Trails:", backward_trails)
            for trail in backward_trails:
                if pattern_matching(pattern = pattern, nodes = trail, modules = named_modules, verbose=verbose):
                    
                    if verbose: 
                        print("<green>[Matched]</green>", "Nodes:", trail)

                    output = graph_transform(node = node, trail = trail, pattern = pattern, graph = graph, modules = named_modules)
                    log_transform.append((pattern, node, output))
    
    trace.graph.lint()
    
    if not inplace: 
        trace = fx.GraphModule(trace, graph)
    else:
        trace.recompile()
    
    trace.graph.eliminate_dead_code()
    trace.delete_all_unused_submodules()
    
    return trace, log_transform
