import torch

def spike_mechanism_exp(neuron):     
    y = neuron.v 
    '''
    spike = torch.heaviside( 
        y - torch.exp(neuron.x), 
        torch.zeros_like(y)
    )
    '''
    spike = (y >= torch.exp(neuron.x)).to(y)
    
    return spike


def spike_mechanism_div(neuron):     
    y = neuron.v
    x1, x2  = neuron.x.x, neuron.x.y

    y = y.to(x1)
    spike = torch.heaviside(y - torch.divide(x1,x2), torch.zeros_like(x1)) # TODO
    
    return spike