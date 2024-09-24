import torch
from torch import nn, Tensor
from typing import List, Optional

class RpropModule:
    def __init__(self, 
        lr=1e-2,
        etas=(0.5, 1.2),
        step_sizes=(1e-6, 50),
        ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 < etas[0] < 1.0 < etas[1]:
            raise ValueError(f"Invalid eta values: {etas[0]}, {etas[1]}")
        
        self.config = dict(
            lr=lr,
            etas=etas,
            step_sizes=step_sizes,
        )

    def reset(self, param):
        self.state = {
            'param': param,
            'prev':  0.0, #torch.zeros_like(param, memory_format=torch.preserve_format),
            'step_size': self.config['lr']#torch.zeros_like(param).fill_(self.config['lr'])
        }

    def step(self, grad):

        etaminus, etaplus = self.config["etas"]
        step_size_min, step_size_max = self.config["step_sizes"]

        param = self.state['param']
        if isinstance(param, float): # x_float_to_tensor
            param = torch.full_like(grad, param)

        state = self.rprop(
                param = param,
                grad = grad,
                prev = self.state['prev'],
                step_size = self.state['step_size'],
                step_size_min=step_size_min,
                step_size_max=step_size_max,
                etaminus=etaminus,
                etaplus=etaplus,
            )

        self.state = state
        #print('state:',self.state, 'param:', state['param'])
        return state['param']

    def rprop(self,
            param, grad, 
            prev, step_size,
            step_size_min: float,
            step_size_max: float,
            etaminus: float,
            etaplus: float,
    ):
        sign = grad.mul(prev).sign()
        sign[sign.gt(0)] = etaplus
        sign[sign.lt(0)] = etaminus
        sign[sign.eq(0)] = 1

        step_size = torch.clamp(torch.mul(step_size, sign), min = step_size_min, max = step_size_max)

        grad[sign.eq(etaminus)] = 0

        param = torch.addcmul(param, grad.sign(), step_size, value = -1)

        state = {
            'param': param,
            'prev': torch.clone(grad),
            'step_size': step_size,
        }
        return state

    