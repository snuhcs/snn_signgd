import torch
from .neuronal_dynamical_system.spikingjelly.dtype import TensorPair

def reduce_duplicates(tensor):
    if torch.is_tensor(tensor):
        for dim in tensor.shape:
            if torch.allclose(tensor[0], tensor[1:]):
                tensor = tensor[0]
            else:
                break
        tensor = tensor.clone()
    elif isinstance(tensor, TensorPair):
        x = reduce_duplicates(tensor.x)
        y = reduce_duplicates(tensor.y)
        tensor = TensorPair(x, y)
    return tensor