import torch
import torch.nn.functional as F

def spike_mechanism_gelu(neuron):     
    y = neuron.v
    
    y = y.to(neuron.x)
    spike = torch.heaviside( 
        y - neuron.x * torch.sigmoid(1.702 * neuron.x), 
        torch.zeros_like(y)
    )
    '''
    error = torch.abs(y - F.gelu(neuron.x))
    print("[GELU]", "Step:", neuron.timestep, " |X:", "[",neuron.x.mean(), "+-", neuron.x.std(), "(", neuron.x.max(),")]") 
    print("[GELU]", "Step:", neuron.timestep, " |Y:", "[",y.mean(), "+-", y.std(), "(", y.max(),")]") 
    print("error:", "[", error.mean() , error.std(),"]")
    '''
    
    return spike