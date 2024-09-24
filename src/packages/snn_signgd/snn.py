import torch
from torch import nn
import torch.nn.functional as F

from .graph_functional import pattern_matching_transform
from .graph_functional.transforms import replace_op, replace_ops_cases
from spikingjelly.activation_based import functional

from typing import Callable, List

from tqdm import tqdm
from functools import partial
import copy

from .core.layer import multiply_inverse_of_square_root
torch.fx.wrap("multiply_inverse_of_square_root") # fx.wrap should be at the top of every module 

class SpikingNeuralNetwork(nn.Module):
    def __init__(self, ann_model:nn.Module, config:dict, default_simulation_length:int, dynamics_type:str, sample_data:torch.Tensor):
        super().__init__()        
        
        self.simulation_length = default_simulation_length 

        self.model, self.log_transform = nonlinearity_to_spiking_neuron[dynamics_type](ann_model = ann_model, config = config)
        
        corrections = config.correction(net = self.model, sample_data = sample_data)

        self.codec = config.codec(statistics = corrections)
        
    def forward(self, x, timestamps: List[int] = []):
        if timestamps: 
            simulation_length = max(timestamps)
            history = []
        else:
            simulation_length = self.simulation_length
        
        functional.reset_net(self.model)
        if hasattr(self.codec, 'reset'):
            self.codec.reset()
        
        x_enc = self.codec.encode(x)
        
        y = 0.0 
        for timestep in tqdm(range(1, simulation_length + 1)):
            x_enc_t = next(x_enc)
            y_enc_t = self.model(x_enc_t)
            y = self.codec.decode( y, y_enc_t , timestep )

            if timestep in timestamps:
                history.append( y.clone().detach() )

        if timestamps: 
            return y, history
        else:
            return y

def _to_spiking_neuron_signgd(ann_model, config):
    model, log = pattern_matching_transform(
        ann_model, 
        patterns = [
            (torch.relu,), (F.relu,), (nn.ReLU,), 
            (nn.LeakyReLU,), 
            (nn.GELU,), 
            (torch.maximum,),
            (multiply_inverse_of_square_root,),
            (torch.square,), 
            (torch.exp,),
            (torch.matmul,),
            (torch.div,),
            (torch.abs,),
        ], 
        graph_transform = replace_ops_cases(
            dest_modules = (
                lambda : config.relu(step_mode='s', v_reset= None),
                lambda : config.leakyrelu(step_mode='s', v_reset= None),
                lambda : config.gelu(step_mode='s', v_reset= None),
                lambda : config.maxpool(step_mode='s', v_reset= None),
                lambda : config.mul_inverse_sqrt(step_mode='s', v_reset= None),
                lambda : config.square(step_mode='s', v_reset= None),
                lambda : config.exp(step_mode='s', v_reset= None),
                lambda : config.matmul(step_mode='s', v_reset= None),  
                lambda : config.div(step_mode='s', v_reset= None),
                lambda : config.abs(step_mode='s', v_reset= None),
            ),   
            cases = (
                [(torch.relu,), (F.relu,), (nn.ReLU,)],
                [(nn.LeakyReLU,)],
                [(nn.GELU,)],
                [(torch.maximum,)],
                [(multiply_inverse_of_square_root,)],
                [(torch.square,)],
                [(torch.exp,)],
                [(torch.matmul,)],
                [(torch.div,)],
                [(torch.abs,)],
            ),
            inherit_kwargs = (
                None,
                None,
                None,
                None,
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
    return model, log

def _to_spiking_neuron_subgradient(ann_model, config):
    model, log = pattern_matching_transform(
        ann_model, 
        patterns = [(torch.relu,), (F.relu,), (nn.ReLU,)], 
        graph_transform =  replace_op(
            lambda : config.neuron(step_mode='s', v_reset= None)
        ), 
        inplace = False
    )    
    return model, log

nonlinearity_to_spiking_neuron = {
    'signgd' : _to_spiking_neuron_signgd,
    'subgradient' : _to_spiking_neuron_subgradient
}