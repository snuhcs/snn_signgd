import torch
from torch import nn, fx
import torch.nn.functional as F
from torch.nn.utils.fusion import fuse_conv_bn_eval

from typing import Callable, Union, Tuple, Iterable, Dict, Any, Type
import copy

from ..utils import replace_node_module, pattern_matching #matches_module_pattern, 

def fuse_conv_bn(model: torch.nn.Module, inplace:bool=False) -> torch.nn.Module:
    """
    Fuses convolution/BN layers for inference purposes. Will deepcopy your
    model by default, but can modify the model inplace as well.
    """
    patterns = [(nn.Conv1d, nn.BatchNorm1d),
                (nn.Conv2d, nn.BatchNorm2d),
                (nn.Conv3d, nn.BatchNorm3d)]
    if not inplace:
        model = copy.deepcopy(model)
    fx_model = fx.symbolic_trace(model)
    modules = dict(fx_model.named_modules())
    new_graph = copy.deepcopy(fx_model.graph)

    for pattern in patterns:
        for node in new_graph.nodes:
            if len(node.args) == 0:
                continue
            elif pattern_matching(pattern = pattern, nodes = (node.args[0], node), modules = modules):
                if len(node.args[0].users) > 1:  # Output of conv is used by other nodes
                    continue
                conv = modules[node.args[0].target]
                bn = modules[node.target]
                fused_conv = fuse_conv_bn_eval(conv, bn)
                replace_node_module(node.args[0], modules, fused_conv)
                node.replace_all_uses_with(node.args[0])
                new_graph.erase_node(node)
                
    new_graph.lint()
    
    return fx.GraphModule(fx_model, new_graph)

if __name__ == "__main__":
    from torchvision.models import resnet50, ResNet50_Weights, vgg16, VGG16_Weights, vgg16_bn, VGG16_BN_Weights, vit_b_32, ViT_B_32_Weights
    from src.utils import draw_computational_graph, check_neural_equivalence, save_and_reimport_nn_module, matches_module_pattern, replace_node_module

   # weights = ResNet50_Weights.DEFAULT
    #model = resnet50(weights = weights) # Step 1: Initialize model with the best available weights
    
    weights = VGG16_BN_Weights.DEFAULT
    model = vgg16_bn(weights = weights) # Step 1: Initialize model with the best available weights
    model.eval()
    input_shape = (32,3,224,224)
    
    draw_computational_graph("./example_fuse_conv_bn_before", model, input_shape)
    
    model_fuse = fuse_conv_bn(model)

    model_imported = save_and_reimport_nn_module(model = model_fuse)()

    cases = [model, model_fuse, model_imported]

    check_neural_equivalence(modules = cases, input_shape = input_shape)

