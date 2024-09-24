from torch import fx
from torch.nn.utils.fusion import fuse_conv_bn_eval
from typing import Callable, Any
from ..utils import replace_node_module

def fuse_conv_bn():
    def fuse_conv_bn_node_transform(node:fx.Node, trail, pattern, graph, modules:dict[Any]):
        assert len(pattern) >= 2, "fuse_conv_bn must have pattern longer or equal to 2"
        assert len(trail) >= 2, "fuse_conv_bn must have detected subgraph longer or equal to 2"
        
        node_prev = trail[-2]
        
        if len(node_prev.users) > 1:  # Output of conv is used by other nodes
            return
            
        conv = modules[node_prev.target]
        bn = modules[node.target]
        fused_conv = fuse_conv_bn_eval(conv, bn)
        
        replace_node_module(node_prev, modules, fused_conv)
        node.replace_all_uses_with(node_prev)
        graph.erase_node(node)
        return
        
    return fuse_conv_bn_node_transform