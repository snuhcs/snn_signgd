import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from spikingjelly.activation_based import functional, neuron

from typing import Callable, List, Any

class Psychoactive:
    def __init__(self):
        self.activations = {}
        
    def excite(self):
        return torch.ones_like(self.v)
        
    def depress(self):
        return torch.zeros_like(self.v)
        
    def fetch_activation(self, name: str):
        def fetch(x:torch.Tensor):
            self.activations[name] = x[0:1]
            return
        return fetch
            

def check_neuron(module:nn.Module):
    if not isinstance(module, neuron.BaseNode):
        print(f'Trying to call `force_excite()` of {module}, which is not '
                        f'spikingjelly.activation_based.base.MemoryModule')
    return
    
class stimulant(object):
    def __init__(self, net:nn.Module) -> None:
        self.net = net
        self.excite = torch.ones_like

    def __enter__(self) -> None:
        functional.reset_net(self.net)
        self._stimulate(self.net)
        return self

    def __exit__(self, exc_type:Any, exc_value:Any, traceback:Any) -> None:
        self._recover(self.net)
        functional.reset_net(self.net)
        
    def _stimulate(self, net: torch.nn.Module):
        for m in net.modules():
            if hasattr(m, 'neuronal_fire'):
                check_neuron(m)
                m.neuronal_fire_natural = m.neuronal_fire
                m.neuronal_fire = m.excite
            
            if hasattr(m, 'neuronal_charge'):
                check_neuron(m)
                m.neuronal_charge_natural = m.neuronal_charge
                m.neuronal_charge = m.fetch_activation("excited")
       
    def _recover(self, net: torch.nn.Module):
        for m in net.modules():
            if hasattr(m, 'neuronal_fire'):
                check_neuron(m)
                m.neuronal_fire = m.neuronal_fire_natural
            
            if hasattr(m, 'neuronal_charge'):
                check_neuron(m)
                m.neuronal_charge = m.neuronal_charge_natural

class depressant(object):
    def __init__(self, net:nn.Module) -> None:
        self.net = net
        self.depress = torch.zeros_like

    def __enter__(self) -> None:
        functional.reset_net(self.net)
        self._depress(self.net)
        return self

    def __exit__(self, exc_type:Any, exc_value:Any, traceback:Any) -> None:
        self._recover(self.net)
        functional.reset_net(self.net)
        
    def _depress(self, net: torch.nn.Module):
        for m in net.modules():
            if hasattr(m, 'neuronal_fire'):
                check_neuron(m)
                m.neuronal_fire_natural = m.neuronal_fire
                m.neuronal_fire = m.depress
            
            if hasattr(m, 'neuronal_charge'):
                check_neuron(m)
                m.neuronal_charge_natural = m.neuronal_charge
                m.neuronal_charge = m.fetch_activation("depressed")
       
    def _recover(self, net: torch.nn.Module):
        for m in net.modules():
            if hasattr(m, 'neuronal_fire'):
                check_neuron(m)
                m.neuronal_fire = m.neuronal_fire_natural
            
            if hasattr(m, 'neuronal_charge'):
                check_neuron(m)
                m.neuronal_charge = m.neuronal_charge_natural