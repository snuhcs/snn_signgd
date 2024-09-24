#import numpy as np
#from typing import Callable, List
import math
import torch
from torch import nn

class LRScheduler(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        # Register kwargs as attributes
        for key, value in kwargs.items():
            setattr(self, key, value)

        T_max = 2048
        self.register_buffer('lr_buffer', torch.tensor([self.compute_lr(i) for i in range(T_max)]))

    def compute_lr(self, timestep:int):
        raise NotImplementedError("Subclass must implement this method")
    
    def __call__(self, timestep:int):
        return self.lr_buffer[timestep]

class SqrtLR(LRScheduler):
    def __init__(self):
        super().__init__()
    
    def compute_lr(self, timestep: int):
        return 1.0 / math.sqrt(timestep)

class ConstantLR(LRScheduler):
    def __init__(self):
        super().__init__()
        
    def compute_lr(self, timestep: int):
        return 1.0 

class InverseLR(LRScheduler):
    def __init__(self):
        super().__init__()
        
    def compute_lr(self, timestep: int):
        return 1.0 / (timestep + 1)
        
class PolynomialLR(LRScheduler):
    def __init__(self, power:float):
        super().__init__(power = power)

    def compute_lr(self, timestep: int):
        return 1.0 / math.pow(timestep, self.power)

class ExponentialLR(LRScheduler):
    def __init__(self, decay_factor:float):
        super().__init__(gamma = decay_factor)

    def compute_lr(self, timestep: int):
        return self.gamma ** (timestep - 1)
'''

class ExponentialLR:
    def __init__(self, decay_factor:float):
        self.gamma = decay_factor

    def __call__(self, timestep: int):
        return self.gamma ** (timestep - 1)
'''