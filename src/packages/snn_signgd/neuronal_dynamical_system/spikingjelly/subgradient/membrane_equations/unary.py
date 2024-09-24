import math
import torch
from torch import nn
from typing import Callable, List
from spikingjelly.activation_based import neuron
from ...psychoactive_substance import stimulant, depressant, Psychoactive
from ...template import BaseNeuron, BaseCodec
from itertools import count
from munch import Munch
from tqdm import tqdm
from snn_signgd.system_optimizations import reduce_duplicates

def relocate(src, dst):
    if torch.is_tensor(src):
        src = src.to(dst.device)
    return src


class Neuron(BaseNeuron):
    def __init__(self, 
                 initial_learning_rate:float, learning_rate_scheduler:Callable,
                 momentum: float, nesterov_momentum : bool, 
                 subgradient_function:Callable, 
                 weight_decay:float,
                 function_value_initialization:float = 0.0,
                 **kwargs
                ):
        neuron.BaseNode.__init__(self, **kwargs)
        Psychoactive.__init__(self)
        self.lr_init = initial_learning_rate
        self.lr_schedule = learning_rate_scheduler
        
        self.subgradient = subgradient_function()
        
        self.beta = momentum
        self.nesterov_momentum = nesterov_momentum
        
        self.bias = 1.0
        self.reset()
        
    def reset(self):
        self.timestep = 1
        
        self.v0 = 0.0
        self.v = self.v0
        self.x = self.bias
            
        self.momentum_input = 0.0
        self.momentum = 0.0
        
    def neuronal_charge(self, x: torch.Tensor): # Update internal input x: Similar to Codec.decode
        if isinstance(self.x, float): # x_float_to_tensor
            x_init = self.x
            self.x = torch.full_like(x.data, x_init)

        learning_rate = self.lr_init * self.lr_schedule(self.timestep)

        Wx = self.x - self.bias
            
        input_gradient = self.subgradient.compute(Wx, x - self.bias) 
        
        self.momentum_input = self.beta * self.momentum_input + learning_rate * input_gradient

        I = self.momentum_input
        
        self.x = self.x - I
        
    def neuronal_fire(self): # Compute gradient
        
        f = self.v
        
        if self.nesterov_momentum:
            y = f - self.beta * self.momentum
        else:
            y = f
            
        spikes = self.subgradient.spike(y,self.x)
        return spikes
        
    def neuronal_reset(self, spike): # Update internal function value: Similar to Codec.encode 
        if self.detach_reset:
            spike_d = spike.detach()
        else:
            spike_d = spike
        
        learning_rate = self.lr_init * self.lr_schedule(self.timestep)
        
        f = self.v
        
        gradient = self.subgradient.compute(f, spike)
        self.momentum = self.beta * self.momentum + learning_rate * gradient
        
        self.v = self.v - self.momentum #learning_rate * sign(spikes)
        self.timestep += 1

class Codec(BaseCodec):
    def __init__(self, 
                 choice:str, 
                 initial_learning_rate:float, learning_rate_scheduler:Callable,
                 subgradient_function:Callable,
                 momentum: float, nesterov_momentum : bool, 
                 weight_decay:float, 
                 function_value_initialization:float = 0.0,
                 statistics:dict = None
                ):
        nn.Module.__init__(self)
        self.choice = choice
        self.lr_init = initial_learning_rate
        self.lr_schedule = learning_rate_scheduler
        self.beta = momentum
        self.subgradient = subgradient_function()
        self.nesterov_momentum = nesterov_momentum
                
        if statistics is None:
            self.offset, self.bias = 1.0, 0.0
        else:
            for key, value in statistics.items():
                #self.register_parameter(key, nn.Parameter(value))
                self.register_buffer(key, value)
            
        self.reset()
        
    def encodings(self) -> dict:
        return {
            'float': self.encode_sg,#self.float_encode,
        }
        
    def reset(self):
        self.output_momentum = 0.0

    def encode_sg(self, x):
        f = 0.0
        momentum = 0.0
        for timestep in count(1):
            
            if self.nesterov_momentum:
                y = f - self.beta * momentum
            else:
                y = f
                
            spike = self.subgradient.encode(y,x)
            
            gradient = self.subgradient.compute(y,spike)
            
            learning_rate = self.lr_init * self.lr_schedule(timestep)
            
            momentum = self.beta * momentum + learning_rate * gradient
            
            f = f - momentum 
            yield spike  
    
    def decode(self, state:torch.Tensor, spikes:torch.Tensor, timestep:int) -> torch.Tensor:
        if timestep <= 1 :
            state = 0.0
            
        Wx = state - self.bias

        gradient = self.subgradient.compute(Wx, spikes - self.bias)
        
        learning_rate = self.lr_init * self.lr_schedule(timestep)        

        self.output_momentum = self.beta * self.output_momentum + learning_rate * gradient
        
        Wx = Wx - self.output_momentum  # Running Average
        
        return Wx + self.bias
    
def correction(net:nn.Module, sample_data: torch.Tensor):
    
    with stimulant(net) as S:
        excited_output = net(S.excite(sample_data[0:1]))
    with depressant(net) as D:
        depressed_output = net(D.depress(sample_data[0:1]))
    
    for m in tqdm(net.modules(), desc = "Dynamics Correction", total = len(list(net.modules()))):
        if hasattr(m, 'activations'):
            bias = m.activations["depressed"]
            bias = reduce_duplicates(bias[0])

            del(m.activations)
            del(m.v)

            m.bias = bias

    statistics = Munch(excited = excited_output.clone(), depressed = depressed_output.clone())
    offset = statistics["excited"] - statistics["depressed"] + 2 * statistics["depressed"]
    bias = statistics["depressed"]

    offset = reduce_duplicates(offset[0])
    bias = reduce_duplicates(bias[0])

    del(statistics)

    return Munch(offset = offset, bias = bias, input_init = None)
