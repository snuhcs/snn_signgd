import torch
from torch import nn
import torch.nn.functional as F
from typing import Callable, List, Any, Optional, Tuple
import os
import math
import copy
from torch import Tensor

from snn_signgd.pretty_printer import print

class SignPreservingScaler(nn.Module):
    def __init__(self, scale: torch.Tensor, inverse:bool):
        super().__init__()
        self.inverse = inverse
        self.scale_factor = scale
    def forward(self,x):
        if not self.inverse:
            x *= self.scale_factor
        else:
            x /= self.scale_factor
        return x

class ScaledOp(nn.Module):
    def __init__(self, op:Callable, scale_transform:Callable, statistics:dict):
        super().__init__()
        
        max = statistics[os.path.join('output','max')]
        scale = max
            
        scale[scale <= 1e-5] = 1.0 # Prevent nan
        
        scale = 1.0/ (torch.unsqueeze(torch.abs(scale), dim = 0))
        
        self.forward_scaler = SignPreservingScaler(scale = scale, inverse = False)
        self.relu = op()
        self.backward_scaler = SignPreservingScaler(
            scale = scale_transform(scale), 
            inverse = True
        )
    def forward(self,x):
        y = self.forward_scaler(x)
        y = self.relu(y)
        y = self.backward_scaler(y)
        return y

torch.fx.wrap("ScaledOp") 

class BinaryTreeMaxPool2d(nn.Module):
    def __init__(self, 
                 kernel_size, stride, padding, dilation
                ):
        super().__init__()
        print("K:", kernel_size, "S:", stride, "P:", padding, "D:", dilation)
        assert dilation == 1

        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = (padding, padding, padding, padding)

    def forward(self, x):
        x = F.pad(x, self.pad, "constant", -200)

        chunk0 = x[:,:,::self.stride]
        for index in range(1,self.kernel_size):
            chunk1 = x[:,:,index ::self.stride]
            B, C, H, W = chunk1.shape
            chunk0 = torch.maximum(chunk0[:,:,:H], chunk1)
            #height_overlap_C = torch.complex(chunk0[:,:,:H], chunk1)
        
            #chunk0 = neuron(height_overlap_C).to(x) # ComplexFloat to x.dtype

        spikes = chunk0
        
        chunk0 = spikes[:,:,:,::self.stride]
        for index in range(1, self.kernel_size):
            chunk1 = spikes[:,:,:,index ::self.stride]
            B, C, H, W = chunk1.shape
            chunk0 = torch.maximum(chunk0[:,:,:,:W], chunk1)
            #width_overlap_C = torch.complex(chunk0[:,:,:,:W], chunk1)
        
            #chunk0 = neuron(width_overlap_C).to(x) # ComplexFloat to x.dtype
            
        spikes = chunk0
        return spikes

torch.fx.wrap("BinaryTreeMaxPool2d") 

def multiply_inverse_of_square_root(x:torch.Tensor,y:torch.Tensor) -> torch.Tensor:
    return torch.divide(x, torch.sqrt(y))
    
torch.fx.wrap("multiply_inverse_of_square_root") 
# Inspired by https://pytorch.org/vision/stable/_modules/torchvision/ops/stochastic_depth.html#stochastic_depth

class DecomposedLayerNorm(nn.Module):
    def __init__(self, 
                 normalized_shape, 
                 eps, weight, bias,
                ):
        super().__init__()

        self.normalized_shape = normalized_shape
        
        self.eps = eps

        self.weight = weight
        self.bias = bias
        self.size_list_for_max = []


    def average_last_dims(self, x, dims):
        out = x
        for _ in range(dims):
            out = out.mean(-1, keepdim = True)
        return out
        
    def forward(self, x, normalized_shape = None, weight = None, bias = None, eps = None):
        if normalized_shape is not None:
            dims = len(normalized_shape)
        else:
            dims = len(self.normalized_shape)

        torch._assert(dims == 1, "Only 1D input supported")

        if weight is None:
            weight = self.weight
        if bias is None:
            bias = self.bias
        if eps is None:
            eps = self.eps

        mean = self.average_last_dims(x, dims)
        unbiased_x = x - mean

        square = torch.square(unbiased_x)
        
        variance = self.average_last_dims(square, dims)

        denominator = variance 
        
        normalized_x = multiply_inverse_of_square_root( unbiased_x , denominator + eps ) 

        affine_x =  weight * normalized_x + bias
        return affine_x
        
torch.fx.wrap("DecomposedLayerNorm") 

from torch.nn.parameter import Parameter

def copy_if_exists(param):
    if param is not None:
        return Parameter(param.clone())
    else:
        return None

class DecomposedMultiHeadAttention(nn.Module):
    def __init__(self, 
                 embed_dim, num_heads, 
                 dropout, add_zero_attn,
                 q_proj_weight, k_proj_weight, v_proj_weight, 
                 in_proj_weight, out_proj,
                 in_proj_bias, bias_k, bias_v,
                 _qkv_same_embed_dim, 
                 batch_first, 
                 #device, dtype
                ):
        super().__init__()

        '''
        if embed_dim <= 0 or num_heads <= 0:
            raise ValueError(
                f"embed_dim and num_heads must be greater than 0,"
                f" got embed_dim={embed_dim} and num_heads={num_heads} instead"
            )
        '''
        #factory_kwargs = {'device': device, 'dtype': dtype}
        '''
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim
        '''

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.add_zero_attn = add_zero_attn
        self._qkv_same_embed_dim = _qkv_same_embed_dim

        self.q_proj_weight = copy_if_exists(q_proj_weight)
        self.v_proj_weight = copy_if_exists(v_proj_weight)
        self.k_proj_weight = copy_if_exists(k_proj_weight)
        self.in_proj_weight = copy_if_exists(in_proj_weight) #Parameter(in_proj_weight.clone())
        self.in_proj_bias = copy_if_exists(in_proj_bias) #Parameter(in_proj_bias.clone())
        self.out_proj = copy.deepcopy(out_proj)
        self.bias_k = copy_if_exists(bias_k)
        self.bias_v = copy_if_exists(bias_v)

        '''
        if not self._qkv_same_embed_dim:
            self.q_proj_weight = Parameter(torch.empty((embed_dim, embed_dim), **factory_kwargs))
            self.k_proj_weight = Parameter(torch.empty((embed_dim, self.kdim), **factory_kwargs))
            self.v_proj_weight = Parameter(torch.empty((embed_dim, self.vdim), **factory_kwargs))
            self.register_parameter('in_proj_weight', None)
        else:
            self.in_proj_weight = Parameter(torch.empty((3 * embed_dim, embed_dim), **factory_kwargs))
            self.register_parameter('q_proj_weight', None)
            self.register_parameter('k_proj_weight', None)
            self.register_parameter('v_proj_weight', None)

        if bias:
            self.in_proj_bias = Parameter(torch.empty(3 * embed_dim, **factory_kwargs))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = NonDynamicallyQuantizableLinear(embed_dim, embed_dim, bias=bias, **factory_kwargs)

        if add_bias_kv:
            self.bias_k = Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
            self.bias_v = Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
        else:
            self.bias_k = self.bias_v = None
        '''
        
    def forward(self,             
            query: Tensor,
            key: Tensor,
            value: Tensor,
            need_weights: bool,
            key_padding_mask: Optional[Tensor] = None,
            attn_mask: Optional[Tensor] = None,
            average_attn_weights: bool = True,
            is_causal : bool = False
        ):
        torch._assert(query.dim() == 3, "Only batched input is supported")

        key_padding_mask = F._canonical_mask(
            mask=key_padding_mask,
            mask_name="key_padding_mask",
            other_type=F._none_or_dtype(attn_mask),
            other_name="attn_mask",
            target_type=query.dtype
        )

        attn_mask = F._canonical_mask(
            mask=attn_mask,
            mask_name="attn_mask",
            other_type=None,
            other_name="",
            target_type=query.dtype,
            check_other=False,
        )

        if self.batch_first: #and is_batched:
            print("Ours batch first")
            # make sure that the transpose op does not affect the "is" property
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = (x.transpose(1, 0) for x in (query, key))
                    value = key
            else:
                query, key, value = (x.transpose(1, 0) for x in (query, key, value))
        if not self._qkv_same_embed_dim:
            print("Ours not same embed dim")
            attn_output, attn_output_weights = self.multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask,
                use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight,
                average_attn_weights=average_attn_weights,
                is_causal=is_causal)
        else:
            attn_output, attn_output_weights = self.multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                average_attn_weights=average_attn_weights,
                is_causal=is_causal)
        if self.batch_first: #and is_batched:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights
        
    def multi_head_attention_forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        embed_dim_to_check: int,
        num_heads: int,
        in_proj_weight: Optional[Tensor],
        in_proj_bias: Optional[Tensor],
        bias_k: Optional[Tensor],
        bias_v: Optional[Tensor],
        add_zero_attn: bool,
        dropout_p: float,
        out_proj_weight: Tensor,
        out_proj_bias: Optional[Tensor],
        need_weights: bool,
        training: bool,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        use_separate_proj_weight: bool = False,
        q_proj_weight: Optional[Tensor] = None,
        k_proj_weight: Optional[Tensor] = None,
        v_proj_weight: Optional[Tensor] = None,
        static_k: Optional[Tensor] = None,
        static_v: Optional[Tensor] = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        # Symbolically traceable assert https://pytorch.org/docs/stable/generated/torch._assert.html

        #is_batched = #F._mha_shape_check(query, key, value, key_padding_mask, attn_mask, num_heads) #TODO

        torch._assert(query.dim() == key.dim(), "query and key must have same number of dimensions")
        torch._assert(query.dim() == 3, "Only batched input is supported")
        torch._assert(value.dim() == key.dim(), "value and key must have same number of dimensions")
        is_batched = True
        if key_padding_mask is not None:   
            assert key_padding_mask.dim() == 2, \
                ("For batched (3-D) `query`, expected `key_padding_mask` to be `None` or 2-D"
                 f" but found {key_padding_mask.dim()}-D tensor instead")
        if attn_mask is not None:
            assert attn_mask.dim() in (2, 3), \
                ("For batched (3-D) `query`, expected `attn_mask` to be `None`, 2-D or 3-D"
                 f" but found {attn_mask.dim()}-D tensor instead")

        # set up shape vars
        tgt_len, bsz, embed_dim = query.shape
        src_len, _, _ = key.shape

        key_padding_mask = F._canonical_mask(
            mask=key_padding_mask,
            mask_name="key_padding_mask",
            other_type=F._none_or_dtype(attn_mask),
            other_name="attn_mask",
            target_type=query.dtype
        )

        torch._assert(
            not (is_causal and attn_mask is None), 
            "Need attn_mask if specifying the is_causal hint. You may use the Transformer module method `generate_square_subsequent_mask` to create this mask."
        )

        if is_causal and key_padding_mask is None and not need_weights:
            # when we have a kpm or need weights, we need attn_mask
            # Otherwise, we use the is_causal hint go as is_causal
            # indicator to SDPA.
            attn_mask = None
        else:
            attn_mask = F._canonical_mask(
                mask=attn_mask,
                mask_name="attn_mask",
                other_type=None,
                other_name="",
                target_type=query.dtype,
                check_other=False,
            )

            if key_padding_mask is not None:
                # We have the attn_mask, and use that to merge kpm into it.
                # Turn off use of is_causal hint, as the merged mask is no
                # longer causal.
                is_causal = False

        torch._assert(embed_dim == embed_dim_to_check, f"was expecting embedding dimension of {embed_dim_to_check}, but got {embed_dim}") 
        head_dim = embed_dim // num_heads
        torch._assert(head_dim * num_heads == embed_dim, f"embed_dim {embed_dim} not divisible by num_heads {num_heads}") 
        if use_separate_proj_weight:
            # allow MHA to have different embedding dimensions when separate projection weights are used
            torch._assert(key.shape[:2] == value.shape[:2], f"key's sequence and batch dims {key.shape[:2]} do not match value's {value.shape[:2]}")
        else:
            torch._assert(key.shape == value.shape, f"key shape {key.shape} does not match value shape {value.shape}")

        #
        # compute in-projection
        #
        if not use_separate_proj_weight:
            torch._assert(in_proj_weight is not None, "use_separate_proj_weight is False but in_proj_weight is None")
            q, k, v = F._in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)
        else:
            torch._assert(q_proj_weight is not None, "use_separate_proj_weight is True but q_proj_weight is None")
            torch._assert(k_proj_weight is not None, "use_separate_proj_weight is True but k_proj_weight is None")
            torch._assert(v_proj_weight is not None, "use_separate_proj_weight is True but v_proj_weight is None")
            if in_proj_bias is None:
                b_q = b_k = b_v = None
            else:
                b_q, b_k, b_v = in_proj_bias.chunk(3)
            q, k, v = F._in_projection(query, key, value, q_proj_weight, k_proj_weight, v_proj_weight, b_q, b_k, b_v)

        # prep attention mask

        if attn_mask is not None:
            # ensure attn_mask's dim is 3
            if attn_mask.dim() == 2:
                correct_2d_size = (tgt_len, src_len)
                if attn_mask.shape != correct_2d_size:
                    raise RuntimeError(f"The shape of the 2D attn_mask is {attn_mask.shape}, but should be {correct_2d_size}.")
                attn_mask = attn_mask.unsqueeze(0)
            elif attn_mask.dim() == 3:
                correct_3d_size = (bsz * num_heads, tgt_len, src_len)
                if attn_mask.shape != correct_3d_size:
                    raise RuntimeError(f"The shape of the 3D attn_mask is {attn_mask.shape}, but should be {correct_3d_size}.")
            else:
                raise RuntimeError(f"attn_mask's dimension {attn_mask.dim()} is not supported")

        # add bias along batch dimension (currently second)
        if bias_k is not None and bias_v is not None:
            assert static_k is None, "bias cannot be added to static key."
            assert static_v is None, "bias cannot be added to static value."
            k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = F.pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = F.pad(key_padding_mask, (0, 1))
        else:
            assert bias_k is None
            assert bias_v is None

        #
        # reshape q, k, v for multihead attention and make em batch first
        #
        q = q.view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
        if static_k is None:
            k = k.view(k.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
        else:
            # TODO finish disentangling control flow so we don't do in-projections when statics are passed
            assert static_k.size(0) == bsz * num_heads, \
                f"expecting static_k.size(0) of {bsz * num_heads}, but got {static_k.size(0)}"
            assert static_k.size(2) == head_dim, \
                f"expecting static_k.size(2) of {head_dim}, but got {static_k.size(2)}"
            k = static_k
        if static_v is None:
            v = v.view(v.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
        else:
            # TODO finish disentangling control flow so we don't do in-projections when statics are passed
            assert static_v.size(0) == bsz * num_heads, \
                f"expecting static_v.size(0) of {bsz * num_heads}, but got {static_v.size(0)}"
            assert static_v.size(2) == head_dim, \
                f"expecting static_v.size(2) of {head_dim}, but got {static_v.size(2)}"
            v = static_v

        # add zero attention along batch dimension (now first)
        if add_zero_attn:
            zero_attn_shape = (bsz * num_heads, 1, head_dim)
            k = torch.cat([k, torch.zeros(zero_attn_shape, dtype=k.dtype, device=k.device)], dim=1)
            v = torch.cat([v, torch.zeros(zero_attn_shape, dtype=v.dtype, device=v.device)], dim=1)
            if attn_mask is not None:
                attn_mask = F.pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = F.pad(key_padding_mask, (0, 1))

        # update source sequence length after adjustments
        src_len = k.size(1)

        # merge key padding and attention masks
        if key_padding_mask is not None:
            assert key_padding_mask.shape == (bsz, src_len), \
                f"expecting key_padding_mask shape of {(bsz, src_len)}, but got {key_padding_mask.shape}"
            key_padding_mask = key_padding_mask.view(bsz, 1, 1, src_len).   \
                expand(-1, num_heads, -1, -1).reshape(bsz * num_heads, 1, src_len)
            if attn_mask is None:
                attn_mask = key_padding_mask
            else:
                attn_mask = attn_mask + key_padding_mask

        # adjust dropout probability
        if not training:
            dropout_p = 0.0

        #
        # (deep breath) calculate attention and out projection
        #

        if need_weights:
            print("Need Weights, Decomposable MHA")
            B, Nt, E = q.shape
            q_scaled = q * math.sqrt(1.0 / E)

            assert not (is_causal and attn_mask is None), "FIXME: is_causal not implemented for need_weights"
            if attn_mask is not None:
                attn_output_weights = torch.baddbmm(attn_mask, q_scaled, k.transpose(-2, -1))
            else:
                attn_output_weights = torch.bmm(q_scaled, k.transpose(-2, -1))
            attn_output_weights = F.softmax(attn_output_weights, dim=-1)
            if dropout_p > 0.0:
                attn_output_weights = F.dropout(attn_output_weights, p=dropout_p)

            attn_output = torch.bmm(attn_output_weights, v)

            attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len * bsz, embed_dim)
            attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)
            attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))

            # optionally average attention weights over heads
            attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
            if average_attn_weights:
                attn_output_weights = attn_output_weights.mean(dim=1)

            if not is_batched:
                # squeeze the output if input was unbatched
                attn_output = attn_output.squeeze(1)
                attn_output_weights = attn_output_weights.squeeze(0)
            return attn_output, attn_output_weights
        else:
            print("No Weights, Decomposable MHA")
            # attn_mask can be either (L,S) or (N*num_heads, L, S)
            # if attn_mask's shape is (1, L, S) we need to unsqueeze to (1, 1, L, S)
            # in order to match the input for SDPA of (N, num_heads, L, S)
            if attn_mask is not None:
                if attn_mask.size(0) == 1 and attn_mask.dim() == 3:
                    attn_mask = attn_mask.unsqueeze(0)
                else:
                    attn_mask = attn_mask.view(bsz, num_heads, -1, src_len)

            q = q.view(bsz, num_heads, tgt_len, head_dim)
            k = k.view(bsz, num_heads, src_len, head_dim)
            v = v.view(bsz, num_heads, src_len, head_dim)

            attn_output = self._scaled_dot_product_attention_base(q, k, v, attn_mask, dropout_p, is_causal)
            attn_output = attn_output.permute(2, 0, 1, 3).contiguous().view(bsz * tgt_len, embed_dim)

            attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)
            attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))
            if not is_batched:
                # squeeze the output if input was unbatched
                attn_output = attn_output.squeeze(1)
            return attn_output, None
        
    def _scaled_dot_product_attention(self, query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None) -> torch.Tensor:
        # Efficient implementation equivalent to the following:
        L, S = query.size(-2), key.size(-2)
        scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
        attn_bias = 0.0
        print("scaled dot product attention my impelementation")
        if is_causal:
            assert attn_mask is None
            attn_bias = torch.zeros_like(query)[0,0,:,0:1].repeat(1,S)
            temp_mask = torch.ones_like(attn_bias, dtype=torch.bool).tril(diagonal=0)
            attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
            attn_bias.to(query.dtype)

        if attn_mask is not None:
            attn_bias = torch.zeros_like(attn_mask, dtype=query.dtype)
            if attn_mask.dtype == torch.bool:
                attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
            else:
                attn_bias += attn_mask

        attn_weight = torch.matmul(scale_factor * query, key.transpose(-2, -1))
        attn_weight =  attn_weight + attn_bias

        # e^{x_1 + x_2 + \cdots + x_n} = e^{\mathbb{E}[X] \cdot N}, e^{x_1} + e^{x_2} + \cdots + e^{x_n} \approx e^{\mathbb{E}[X]} \cdot N

        attn_weight = torch.exp(attn_weight - math.log(attn_weight.size(-1))) 
        attn_weight = torch.div(attn_weight * key.size(-1),torch.sum(attn_weight, dim=-1, keepdim=True))

        attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
        output = torch.matmul(attn_weight, value) / key.size(-1)
        return output
    
    def _scaled_dot_product_attention_base(self, query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None) -> torch.Tensor:
        # Efficient implementation equivalent to the following:
        L, S = query.size(-2), key.size(-2)
        scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
        attn_bias = 0.0
        if is_causal:
            assert attn_mask is None
            attn_bias = torch.zeros_like(query)[0,0,:,0:1].repeat(1,S)
            temp_mask = torch.ones_like(attn_bias, dtype=torch.bool).tril(diagonal=0)
            attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
            attn_bias.to(query.dtype)

        if attn_mask is not None:
            attn_bias = torch.zeros_like(attn_mask, dtype=query.dtype)
            if attn_mask.dtype == torch.bool:
                attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
            else:
                attn_bias += attn_mask

        attn_weight = torch.matmul(query, key.transpose(-2, -1))
        attn_weight =  scale_factor * attn_weight + attn_bias

        attn_weight = torch.exp(attn_weight)
        attn_weight = torch.div(attn_weight,torch.sum(attn_weight, dim=-1, keepdim=True))
        attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
        output = torch.matmul(attn_weight, value)
        return output 
        
        
torch.fx.wrap("DecomposedMultiHeadAttention") 

class SquareModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.mean = nn.Identity()
        #self.square_m = nn.Threshold(-9999, 0)
    def forward(self, x):
        out = torch.square(self.mean(x))
        #out = torch.square(x)
        #out = torch.square(x)
        return out
        #return self.square_m(out)
        
class Square(nn.Module):
    def forward(self, x):
        return torch.square(x)
        
class Identity(nn.Module):
    def forward(self, x):
        return x
        
torch.fx.wrap("SquareModule") 

        