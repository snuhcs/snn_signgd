import torch
from torch import nn

#from pytorch_memlab import profile, set_target_gpu

class SGDModule:
    def __init__(self, lr=1e-3, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, inplace = False):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        
        self.config = dict(
            lr=lr, momentum=momentum, dampening=dampening,
            weight_decay=weight_decay, nesterov=nesterov,
            inplace = inplace)

    def reset(self, initializer):
        self.state = {
            'param': initializer,
            'momentum_buffer': None
        }

    def step(self, grad):

        state = self.sgd(
            param = self.state['param'],
            grad = grad,
            momentum_buffer = self.state['momentum_buffer'],
            weight_decay=self.config['weight_decay'],
            momentum=self.config['momentum'],
            lr=self.config['lr'],
            dampening=self.config['dampening'],
            nesterov=self.config['nesterov'],
            inplace = self.config['inplace'])

        self.state = state
        #print('state:',self.state, 'param:', state['param'])
        return state['param']
    
    #@profile
    def sgd(self,
            param, grad, momentum_buffer,
            weight_decay: float,
            momentum: float,
            lr: float,
            dampening: float,
            nesterov: bool,
            inplace: bool):

        d_p = grad
        buf = momentum_buffer

        
        if weight_decay != 0:
            if inplace:
                d_p.add_(param, alpha=weight_decay) # Inplace operation
            else:
                d_p = d_p + param * weight_decay # Not an inplace operation

        if momentum != 0:
            if buf is None:
                buf = torch.clone(d_p).detach()
            else:
                buf = momentum * buf + (1-dampening) * d_p
                #buf = torch.add(torch.mul(buf, momentum), d_p, alpha=1 - dampening)

            if nesterov:
                d_p = d_p + buf * momentum
                #d_p = torch.add(d_p, buf, alpha=momentum)
            else:
                d_p = buf

        #param = param - lr * d_p # Not an inplace operation
        #d_p = param - lr * d_p # Not an inplace operation
        if inplace:
            d_p.multiply_(-lr)
            param = d_p.add(param)
        else:
            #param = torch.add(param, d_p, alpha = -lr) #
            param = param - lr * d_p 

        state = {
            'param': param,
            'momentum_buffer': buf
        }
        return state
    
