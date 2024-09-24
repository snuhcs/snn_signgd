import torch
from torch import Tensor
from typing import List, Optional

class RMSpropModule:
    def __init__(
        self,
        lr=1e-2,
        alpha=0.99,
        eps=1e-8,
        weight_decay=0,
        momentum=0,
        centered=False,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= momentum:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= alpha:
            raise ValueError(f"Invalid alpha value: {alpha}")

        self.config = dict(
            lr=lr,
            momentum=momentum,
            alpha=alpha,
            eps=eps,
            centered=centered,
            weight_decay=weight_decay,
        )

    def reset(self, param):
        self.state = {
            'param': param,
            'square_avg': 0,
            'momentum_buffer': 0,
            'grad_avg': 0,
        }

    def step(self, grad):

        param = self.state['param']
        if isinstance(param, float): # x_float_to_tensor
            param = torch.full_like(grad, param)

        state = self.rmsprop(
            param = param,
            grad = grad,
            square_avg = self.state['square_avg'],
            grad_avg = self.state['grad_avg'],
            momentum_buffer = self.state['momentum_buffer'],
            lr=self.config["lr"],
            alpha=self.config["alpha"],
            eps=self.config["eps"],
            weight_decay=self.config["weight_decay"],
            momentum=self.config["momentum"],
            centered=self.config["centered"],
        )
        self.state = state

        return state['param']

    def rmsprop(self,
        param, grad,
        square_avg, grad_avg, momentum_buffer,
        lr: float,
        alpha: float,
        eps: float,
        weight_decay: float,
        momentum: float,
        centered: bool,
    ):

        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)

        square_avg = torch.mul(square_avg, alpha).to(grad)
        square_avg = torch.addcmul(square_avg, grad, grad, value=1 - alpha)

        if centered:
            grad_avg = torch.lerp(grad_avg, grad, 1 - alpha)
            avg = torch.sqrt(torch.addcmul(square_avg, grad_avg, grad_avg, value=-1))
        else:
            avg = torch.sqrt(square_avg)

        avg = torch.add(avg, eps)

        if momentum > 0:
            buf = momentum_buffer
            buf = torch.addcdiv(torch.mul(buf, momentum), grad, avg)
            param = torch.add(param, buf, alpha=-lr)
        else:
            param = torch.addcdiv(param, grad, avg, value = -lr)
        
        state = {
            'param': param,
            'square_avg': square_avg,
            'grad_avg': grad_avg,
            'momentum_buffer': momentum_buffer,
        }
        return state