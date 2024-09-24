import torch
class TensorPair:
    def __init__(self, x, y):
        self.x = x
        self.y = y

        #torch._assert(x.shape == y.shape, "TensorPair: x and y must have the same shape")
        self.data = self.x

        self.device = self.x.device
        
    def __repr__(self):
        return f"TensorPair({self.x}, {self.y})"

    def __add__(self, other, inplace = False):
        if inplace:
            self.x = self.x + other.x
            self.y = self.y + other.y
            return self
        else:
            return TensorPair(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other, inplace = False):
        if inplace:
            self.x = self.x - other.x
            self.y = self.y - other.y
            return self
        else:
            return TensorPair(self.x - other.x, self.y - other.y)
    
    def __rmul__(self, other, inplace = False):
        if inplace:
            self.x = other * self.x
            self.y = other * self.y
            return self
        else:
            return TensorPair(other * self.x, other * self.y)
    
    def __getitem__(self, item):
        return TensorPair(self.x.__getitem__(item), self.y.__getitem__(item))