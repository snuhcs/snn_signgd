import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tqdm import tqdm

from .graph_functional import pattern_matching_transform
from .graph_functional.transforms import replace_op, fuse_conv_bn, replace_ops_cases

from .core.layer import ScaledOp, BinaryTreeMaxPool2d, DecomposedLayerNorm, DecomposedMultiHeadAttention, multiply_inverse_of_square_root
torch.fx.wrap("multiply_inverse_of_square_root") # fx.wrap should be at the top of every module 

from .core.hook import hook_context, activation_stats_hook

from functools import partial

unary_module_placeholder = nn.Hardtanh

@torch.no_grad()
def porting(
        model,
        dataloader: DataLoader, 
        max_activation_iterations:int,
        scale_relu_with_max_activation:bool,
    ):
    assert isinstance(scale_relu_with_max_activation, bool), 'Only bool type is supported for scale_relu_with_max_activation.'

    model = _fuse_conv_bn(model)
    model = _decompose_maxpool(model)
    model = _decompose_layer_norm(model)
    model = _modularize_relu(model)
    model = _decompose_multihead_attention(model)

    #model = _modularize_ops_for_analysis(model)
    #model = _modularize_relu(model)

    if scale_relu_with_max_activation is not None:
        _collect_activations(model, dataloader=dataloader, max_activation_iterations=max_activation_iterations)
        model = _scale_relu(model=model)
    return model

def _fuse_conv_bn(model):
    model, _ = pattern_matching_transform(
        model, 
        patterns = [
            (nn.Conv1d, nn.BatchNorm1d),
            (nn.Conv2d, nn.BatchNorm2d),
            (nn.Conv3d, nn.BatchNorm3d)
        ],
        graph_transform = fuse_conv_bn(), 
        inplace = False
    )
    return model

def _decompose_maxpool(model):
    model, _ = pattern_matching_transform(
        model, 
        patterns = [(F.max_pool2d,), (nn.MaxPool2d,)], 
        graph_transform = replace_op(
            BinaryTreeMaxPool2d,
            inherit_args = { 
                'kernel_size':"kernel_size", 
                "stride": "stride", 
                "padding":"padding", 
                "dilation" : "dilation"
            }
        ), 
        inplace = False
    )
    return model

def _decompose_layer_norm(model):
    model, _ = pattern_matching_transform(
        model, 
        patterns = [(nn.LayerNorm,), (F.layer_norm,)], 
        graph_transform = replace_op(
            DecomposedLayerNorm,
            inherit_args = { 
                'normalized_shape':'normalized_shape', 
                'eps':'eps', 
                #'elementwise_affine':'elementwise_affine',
                'weight':'weight', 
                'bias':'bias',
            }
        ), 
        inplace = False
    )
    return model

def _decompose_multihead_attention(model):
    model, _ = pattern_matching_transform(
        model, 
        patterns = [(nn.MultiheadAttention,)], 
        graph_transform = replace_op(
            DecomposedMultiHeadAttention,
            inherit_args = { 
                'embed_dim': "embed_dim", 
                'num_heads':'num_heads',
                'dropout':'dropout',
                'add_zero_attn':'add_zero_attn',
                'q_proj_weight':'q_proj_weight',
                'k_proj_weight':'k_proj_weight',
                'v_proj_weight':'v_proj_weight',
                'in_proj_weight':'in_proj_weight',
                'out_proj':'out_proj',
                'in_proj_bias':'in_proj_bias',
                'bias_k':'bias_k',
                'bias_v':'bias_v',
                '_qkv_same_embed_dim':'_qkv_same_embed_dim',
                'batch_first':'batch_first',
            }
        ), 
        inplace = False
    )
    return model

def _collect_activations(model, dataloader, max_activation_iterations):
    with hook_context(hook_fn = activation_stats_hook) as context:
        device = next(model.parameters()).device
        for index, (input, _) in tqdm(enumerate(dataloader), desc = 'Collect Activations', total = max_activation_iterations):
            input = input.to(device=device)
            
            _ = model(input)
            
            if index > max_activation_iterations: 
                break 
        '''
        for idx, (key, module) in enumerate(dict(model.named_modules()).items()):
            print(idx, '->', module)
            if hasattr(module,"activation_stats"):
                for key, value in module.activation_stats.items():
                    if isinstance(value, int):
                        print(key, value)
                    else:
                        print(key,value.shape )
                        if isinstance(module, nn.ReLU):
                            print(value)
        '''
    return

class Square(nn.Module):
    def forward(self, input):
        return torch.square(input)
class Exp(nn.Module):
    def forward(self, input):
        return torch.exp(input)
class Abs(nn.Module):
    def forward(self, input):
        return torch.abs(input)
class Div(nn.Module):
    def forward(self, input, other):
        return torch.div(input, other)
class Matmul(nn.Module):
    def forward(self, input, other):
        return torch.matmul(input, other)
class MultiplyInverseSquareRoot(nn.Module):    
    def forward(self, x,y):
        return multiply_inverse_of_square_root(x,y) 
    
class Modular(nn.Module):
    def __init__(self, forward_fn):
        super().__init__()
        self.forward_fn = forward_fn
    def forward(self, *args, **kwargs):
        return self.forward_fn(*args, **kwargs)

class ReLUWrapper(nn.Module):
    def __init__(self):
        super(ReLUWrapper, self).__init__()
        self.fn = nn.ReLU()
    def forward(self, input, *args, **kwargs):
        return self.fn(input)

def _modularize_ops_for_analysis(model):
    model, _ = pattern_matching_transform(
        model, 
        patterns = [
                (multiply_inverse_of_square_root,),
                (torch.square,),
                (torch.div,),
                (torch.exp,),
                (torch.matmul,),
                (torch.abs,),
            ], 
        graph_transform = replace_ops_cases(
            dest_modules = (
                MultiplyInverseSquareRoot,
                Square,
                Div,
                Exp,
                Matmul,
                Abs,
            ),
            cases = (
                [(multiply_inverse_of_square_root,)],
                [(torch.square,)], 
                [(torch.div,)],
                [(torch.exp,)],
                [(torch.matmul,)],
                [(torch.abs,),],
            ),
            inherit_kwargs = (
                None,
                None,
                None,
                None,
                None,
                None,
            ),
        ), 
        inplace = False,
        verbose = True,
    ) 

    return model

def _modularize_relu(model):
    model, _ = pattern_matching_transform(
        model, 
        patterns = [(torch.relu,), (F.relu,), (nn.ReLU,)], 
        graph_transform = replace_op(
            unary_module_placeholder
        ), 
        inplace = False
    ) 
    model, _ = pattern_matching_transform(
        model, 
        patterns = [(unary_module_placeholder,)], 
        graph_transform = replace_op(
            #nn.ReLU
            ReLUWrapper
        ), 
        inplace = False
    ) 
    return model

def _scale_relu(model):
    model, _ = pattern_matching_transform(
        model, 
        patterns = [(torch.relu,), (F.relu,), (nn.ReLU,)], 
        graph_transform = replace_op(
            partial(
                ScaledOp, 
                op = unary_module_placeholder, 
                scale_transform = lambda x : x
            ),
            inherit_args = {'statistics':"activation_stats"}
        ), 
        inplace = False
    )    
    
    model, _ = pattern_matching_transform(
        model, 
        patterns = [(unary_module_placeholder,)], 
        graph_transform = replace_op(
            nn.ReLU
        ), 
        inplace = False
    ) 
    return model