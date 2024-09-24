import torch

def spike_mechanism_leakyrelu(neuron, negative_slope:float):               
    y = neuron.v
    
    condition = neuron.x >= 0
    trueval = torch.heaviside( y - neuron.x , torch.tensor([0.0]))
    falseval = torch.heaviside( y - negative_slope * neuron.x, torch.tensor([0.0]))
    spike = condition * trueval + torch.logical_not(condition) * falseval
    
    return spike