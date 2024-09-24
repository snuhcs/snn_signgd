import torch
from torch import nn
from spikingjelly.activation_based import neuron
from .psychoactive_substance import Psychoactive

class BaseNeuron(neuron.BaseNode, Psychoactive):
    def __init__(self, **kwargs):
        raise NotImplementedError()
        
    def reset(self):
        raise NotImplementedError()        
        
    def neuronal_charge(self, x: torch.Tensor): 
        raise NotImplementedError()
        
    def neuronal_fire(self):
        raise NotImplementedError()
        
    def neuronal_reset(self, spike):
        raise NotImplementedError()
    
class BaseNeuronWrapper(nn.Module):
    pass
    
class BaseCodec(nn.Module):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError()

    def encodings(self) -> dict:
        raise NotImplementedError()
        
    def encode(self, x:torch.Tensor) -> torch.Tensor:
        encodings = self.encodings()
        assert self.choice in encodings.keys(), list(encodings.keys())
        return encodings[self.choice](x)
        
    def decode(self, state:torch.Tensor, spikes:torch.Tensor, timestep:int) -> torch.Tensor:
        raise NotImplementedError()
    