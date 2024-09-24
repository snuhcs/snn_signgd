import torch

def spike_mechanism_square(neuron):     
    y = neuron.v

    '''
    error = torch.abs(y - neuron.x ** 2)
    print()
    print("[SQUARE]", "Step:", neuron.timestep, " |X:", "[",neuron.x.mean(), "+-", neuron.x.std(), "(", neuron.x.max(),")]") 
    print("[SQUARE]", "Step:", neuron.timestep, " |Y:", "[",y.mean(), "+-", y.std(), "(", y.max(),")]") 
    print("error:", "[", error.mean() , error.std(),"]")
    '''
    #spike = torch.heaviside(y.to(neuron.x) - neuron.x ** 2 , torch.zeros_like(neuron.x).to(neuron.x))
    spike = (y >= neuron.x ** 2).to(y)
    return spike


def spike_multiply_inverse_of_square_root(neuron):        
    y = neuron.v
    x1, x2  = neuron.x.x, neuron.x.y

    '''
    x1 = x1.float(); x2 = x2.float()
    y = y.float()
    error = torch.abs(y - torch.div(x1, torch.sqrt(torch.abs(x2) + 1e-6)))
    print("[MULTDIV]", "Step:", neuron.timestep, " |X1:", "[",x1.mean(), "+-", x1.std(), "(",x1.max(),")]") 
    print("[MULTDIV]", "Step:", neuron.timestep, " |X2:", "[",x2.mean(), "+-", x2.std(), "(",x2.max(),")]") 
    print("[MULTDIV]", "Step:", neuron.timestep, " |Y:", "[",y.mean(), "+-", y.std(), "(", y.max(),")]") 
    print("error:", "[", error.mean() , error.std(),"]")
    '''
    
    y = y.to(x1)
    spike = torch.heaviside( torch.sqrt(x2) * y - x1 , torch.zeros_like(x1))
    
    return spike