import torch
from torch import nn
from spikingjelly.activation_based import neuron, encoding
from .template import BaseNeuron, BaseCodec
from munch import Munch
'''
def Neuron(tau:float, v_threshold:float, decay_input:bool):
    return neuron.LIFNode(step_mode='s', tau= tau, v_threshold = v_threshold, v_reset = None, decay_input = decay_input)
'''

def Neuron(**kwargs):
    return neuron.LIFNode(**kwargs)

def decode(state: torch.Tensor, spikes: torch.Tensor, timestep:int, neuronal_dynamics: neuron.BaseNode):#f, spike, neuron):
    return (1-(1.0/neuronal_dynamics.tau)) * state + (1.0 / neuronal_dynamics.tau) * spikes

class Codec(BaseCodec):
    def __init__(self, choice:str, tau:float, **kwargs):
        self.tau = tau
        self.choice = choice

    def encodings(self) -> dict:
        return {
            'poisson': self.poisson_encode,
            'float': self.float_encode
        }
        
    def decode(self, state:torch.Tensor, spikes:torch.Tensor, timestep:int) -> torch.Tensor:
        return (timestep - 1) / timestep * state + 1 / timestep * spikes # Running Average
        #return (1-(1.0/self.tau)) * state + (1.0 / self.tau) * spikes # Running Average
        
    def float_encode(self, x: torch.Tensor): # float encoding
        while True:
            yield x
            
    def poisson_encode(self, x: torch.Tensor): # float encoding
        encoder = encoding.PoissonEncoder()
        while True:
            yield encoder(x)

def correction(net:nn.Module, sample_data: torch.Tensor) -> (torch.Tensor, dict):
    return Munch()