import torch
from torch import nn
from spikingjelly.activation_based import neuron, encoding
from .template import BaseNeuron, BaseCodec
from munch import Munch

def Neuron(**kwargs):
    return neuron.IFNode(**kwargs)

class Codec(BaseCodec):
    def __init__(self, choice:str, **kwargs ):
        self.choice = choice

    def encodings(self) -> dict:
        return {
            'poisson': self.poisson_encode,
            'float': self.float_encode
        }
        
    def decode(self, state:torch.Tensor, spikes:torch.Tensor, timestep:int) -> torch.Tensor:
        return (timestep - 1) / timestep * state + 1 / timestep * spikes # Running Average
        
    def float_encode(self, x: torch.Tensor): # float encoding
        while True:
            yield x
            
    def poisson_encode(self, x: torch.Tensor): # float encoding
        encoder = encoding.PoissonEncoder()
        while True:
            yield encoder(x)

def correction(net:nn.Module, sample_data: torch.Tensor) -> dict:
    return Munch()