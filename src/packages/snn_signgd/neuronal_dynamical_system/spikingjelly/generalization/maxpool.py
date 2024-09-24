import torch

def spike_mechanism_maximum(neuron):     
    y = neuron.v
    x1, x2  = neuron.x.x, neuron.x.y

    y = y.to(x1)
    #torch._assert(x1.shape == x2.shape, "Chunk shapes do not match | " + str(x1.shape[-1] == 4) + " " + str(x2.shape[-1] == 4))
    spike = torch.heaviside(y - torch.max(x1, x2), torch.zeros_like(x1))
    
    return spike