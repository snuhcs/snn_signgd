import torch
from typing import Callable, List
from spikingjelly.activation_based import neuron
from ...psychoactive_substance import Psychoactive
from ...template import BaseNeuron, BaseCodec
from itertools import count
from munch import Munch
from torch import nn

#from pytorch_memlab import profile, set_target_gpu

def relocate(src, dst):
    if torch.is_tensor(src):
        src = src.to(dst.device)
    return src

def sign(spike):
    return 2 *spike - 1

class Neuron(BaseNeuron):
    def __init__(self, 
                 optimizer_input:Callable,
                 optimizer_output:Callable,
                 lr_scheduler_input:Callable,
                 lr_scheduler_output:Callable,
                 spike_mechanism:Callable, 
                 **kwargs
                ):
        neuron.BaseNode.__init__(self, **kwargs)
        Psychoactive.__init__(self)
        self.optimizer_input = optimizer_input
        self.optimizer_output = optimizer_output
        self.lr_scheduler_input = lr_scheduler_input
        self.lr_scheduler_output = lr_scheduler_output

        self.spike_mechanism = spike_mechanism
        
        self.bias = 0.0
        self.reset()
        
    def reset(self):

        self.optimizer_input.reset(initializer = self.bias)
        self.optimizer_output.reset(initializer = 0.0)
        self.lr_scheduler_input.reset(moduleoptimizer = self.optimizer_input)
        self.lr_scheduler_output.reset(moduleoptimizer = self.optimizer_output)

        self.v = 0.0
        #print("Neuron reset called")
    
    #@profile
    def neuronal_charge(self, x: torch.Tensor): # Update internal input x: Similar to Codec.decode
        '''
        if isinstance(self.x, float): # x_float_to_tensor
            x_init = self.x
            self.x = torch.full_like(x.data, x_init)
        '''
        
        gradient = 2 * x - relocate(self.offset, x) #relocate(self.offset, x) # approx. += W * x(t)

        self.x = self.optimizer_input.step(grad = gradient)
        self.lr_scheduler_input.schedule()
    #@profile    
    def neuronal_fire(self): # Compute gradient
        spikes = self.spike_mechanism(self)
        return spikes
    #@profile    
    def neuronal_reset(self, spike): # Update internal function value: Similar to Codec.encode 
        '''
        if self.detach_reset:
            spike_d = spike.detach()
        else:
            spike_d = spike
        '''
        
        gradient = sign(spike)

        self.v = self.optimizer_output.step(grad = gradient)
        self.lr_scheduler_output.schedule()

class Codec(BaseCodec, nn.Module):
    def __init__(self, 
                 choice:str, 
                 optimizer_enc:Callable,
                 optimizer_dec:Callable,
                 lr_scheduler_enc:Callable,
                 lr_scheduler_dec:Callable,
                 statistics:dict = None
                ):
        nn.Module.__init__(self)
        self.choice = choice
        
        self.optimizer_enc = optimizer_enc
        self.optimizer_dec = optimizer_dec
        self.lr_scheduler_enc = lr_scheduler_enc
        self.lr_scheduler_dec = lr_scheduler_dec
                
        if statistics is None:
            self.offset, self.bias = 1.0, 0.0
        else:
            for key, value in statistics.items():
                #self.register_parameter(key, nn.Parameter(value))
                self.register_buffer(key, value)
            
        self.reset()
        
    def encodings(self) -> dict:
        return {
            'float': self.encode_template(self.grad_float),#self.float_encode,
            'spike': self.encode_template(self.grad_spike),#self.spike_encode,
            'stochastic_spike': self.encode_template(self.grad_stochastic_spike),
        }
        
    def reset(self):
        self.optimizer_enc.reset(initializer = 0.0)
        self.optimizer_dec.reset(initializer = self.bias)
        self.lr_scheduler_enc.reset(moduleoptimizer = self.optimizer_enc)
        self.lr_scheduler_dec.reset(moduleoptimizer = self.optimizer_dec)
        #print("Codec reset called")
        
    def grad_float(self, y, x):
        gradient = y - x # Loss function L(y;x) = \Vert y - x \Vert_2^2
        spike = 0.5 * (1 + gradient)
        return gradient, spike
        
    def grad_spike(self, y, x):                
        spike = torch.heaviside( y - x , torch.zeros_like(x)) # Loss function L(y;x) = \Vert y - x \Vert_2^2
        gradient = sign(spike)
        return gradient, spike
        
    def grad_stochastic_spike(self, y, x):                
        spike = torch.bernoulli(torch.sigmoid( (y - x) )) # Loss function L(y;x) = \Vert y - x \Vert_2^2
        gradient = sign(spike)
        return gradient, spike

    def encode_template(self, gradient_fn):
        def encode_fn(x):
            f = 0.0
            for timestep in count(1):
                gradient, spike = gradient_fn(f,x) 
        
                f = self.optimizer_enc.step(grad = gradient)
                self.lr_scheduler_enc.schedule()
    
                yield spike             
        return encode_fn
    
    def decode(self, state:torch.Tensor, spikes:torch.Tensor, timestep:int) -> torch.Tensor:
        gradient = 2 * spikes - relocate(self.offset, spikes) # W * x(t)
        
        y = self.optimizer_dec.step(grad = gradient)
        self.lr_scheduler_dec.schedule()

        return y
    
