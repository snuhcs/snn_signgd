import torch
from ..template import BaseNeuronWrapper
from .membrane_equations.binary import Neuron
from .membrane_equations.binary import TensorPair
from ..dtype import TensorPair 
        
class MatMulNeuron(BaseNeuronWrapper):
    def __init__(self, **neuronal_dynamics_kwargs):
        super().__init__()
        self.neuron = Neuron(**neuronal_dynamics_kwargs)
    def forward(self, x,y):
        # Naive implementation
        num_right_tokens = y.shape[-1]

        repeat_count = num_right_tokens

        x = torch.unsqueeze(x, dim = -1)
        size = list(x.shape)
        size[-1] = repeat_count

        x, y = x.expand(*size), torch.unsqueeze(y, dim = -3)

        pair = TensorPair(x, y)
        
        output = self.neuron(pair).to(x)
        output = torch.sum(output, dim = -2)
        '''
        # reorder dim to reduce memory usage of neuron correction mechanism
        # x : (N, K) y: (K, M)
        num_right_tokens = y.shape[-1]

        repeat_count = num_right_tokens

        x = torch.unsqueeze(x, dim = -3)
        size = list(x.shape)
        size[-3] = repeat_count
        # x: (M, N, K) y: (M, 1, K)
        x, y = x.expand(*size), torch.unsqueeze(torch.transpose(y,dim0=-1,dim1=-2), dim = -2)

        pair = TensorPair(x, y)
        
        output = self.neuron(pair).to(x)
        output = torch.sum(output, dim = -1)
        '''
        '''
        num_right_tokens = y.shape[-1]

        repeat_count = num_right_tokens

        size = list(x.shape)
        size[-1] = repeat_count

        out = torch.ones(size).to(x)
        pair = TensorPair(out,out)
        output = self.neuron(pair).to(x)
        '''
        return output
    
    
def spike_mechanism_multiply(neuron):     
    y = neuron.v

    x1, x2  = neuron.x.x, neuron.x.y
    y = y.to(x1)
    spike = (y >= torch.mul(x1,x2)).to(y)
    #spike = torch.heaviside(y - torch.mul(x1, x2), torch.zeros_like(neuron.x.data)) # TODO
    
    return spike

class ParallelogramMatMulNeuron(BaseNeuronWrapper):
    def __init__(self, **neuronal_dynamics_kwargs):
        super().__init__()
        self.square_x_plus_y = Neuron(**neuronal_dynamics_kwargs)
        self.square_x = Neuron(**neuronal_dynamics_kwargs)
        self.square_y = Neuron(**neuronal_dynamics_kwargs)
    def forward(self, x,y):
        '''
        num_right_tokens = y.shape[-1]

        repeat_count = num_right_tokens

        size = list(x.shape)
        size[-1] = repeat_count
        output = torch.zeros(*size, device = x.device, dtype = x.dtype)
        output = self.neuron(output).to(x)
        '''
        x = torch.unsqueeze(x, dim = -1)
        y = torch.unsqueeze(y, dim = -3)
        xplusy = x + y

        xplusy_sq = self.square_x_plus_y(xplusy)
        xplusy_sq = torch.sum(xplusy_sq, dim = -2)
        x_sq = self.square_x(x)
        x_sq = torch.sum(x_sq, dim = -2)
        y_sq = self.square_y(y)
        y_sq = torch.sum(y_sq, dim = -2)

        output = 0.5 * (xplusy_sq - x_sq - y_sq)

        return output
    
    
def spike_mechanism_square(neuron):     
    y = neuron.v

    spike = torch.heaviside(y - neuron.x ** 2, neuron.x) # TODO
    
    return spike

