import torch

def spike_mechanism_relu(neuron):               
    y = neuron.v 
    
    '''
    condition = neuron.x >= 0
    trueval = torch.heaviside( y - neuron.x , torch.zeros_like(y))
    falseval = torch.heaviside( y , torch.zeros_like(y))
    spike = condition * trueval + torch.logical_not(condition) * falseval
    '''

    '''
    spike = torch.heaviside(y - torch.clamp(neuron.x, min = 0), torch.zeros_like(y)) 
    '''
    
    condition = neuron.x >= 0
    trueval = y >= neuron.x
    falseval = y >= 0
    spike = torch.logical_or(torch.logical_and(condition, trueval), torch.logical_and(torch.logical_not(condition), falseval)).to(neuron.x)

    return spike