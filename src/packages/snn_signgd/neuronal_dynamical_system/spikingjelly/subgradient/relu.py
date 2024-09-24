import torch

class Subgradient_ReLU: # ReLU(x-y) + 0.5 * y^2
    def __init__(self):
        pass
    def spike(self, y, x):
        spikes = torch.heaviside( x - y , torch.zeros_like(x))
        return spikes
    def compute(self, y, spikes):
        gradient = y - spikes
        return gradient
    def encode(self, y, x):
        spikes = x
        return spikes
