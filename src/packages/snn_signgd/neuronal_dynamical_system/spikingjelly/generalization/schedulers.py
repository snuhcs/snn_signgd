class ExponentialScheduler:
    def __init__(self, gamma: float, eager_evaluation: bool = False):
        self.gamma = gamma
        self.lr = None
        self.eager_evaluation = eager_evaluation
        
    def reset(self, moduleoptimizer):
        self.moduleoptimizer = moduleoptimizer
        if self.lr is None:
            self.lr = self.moduleoptimizer.config['lr']
        else:
            self.moduleoptimizer.config['lr'] = self.lr

        if self.eager_evaluation:
            M = 1024
            self.lrs = [self.lr * (self.gamma ** i) for i in range(M)]
        self.timestep = 1

    def schedule(self):
        if self.eager_evaluation:
            self.moduleoptimizer.config['lr'] = self.lrs[self.timestep]
        else:
            self.moduleoptimizer.config['lr'] *= self.gamma 
        #print("Timestep:", self.timestep, "Learning rate:", self.moduleoptimizer.config['lr'], "Original learning rate:", self.lr)
        self.timestep += 1

class InverseScheduler:
    def __init__(self, eager_evaluation: bool = False):
        self.lr = None
        self.eager_evaluation = eager_evaluation
        
    def reset(self, moduleoptimizer):
        self.moduleoptimizer = moduleoptimizer
        if self.lr is None:
            self.lr = self.moduleoptimizer.config['lr']
        else:
            self.moduleoptimizer.config['lr'] = self.lr

        if self.eager_evaluation:
            M = 1024
            self.lrs = [self.lr / (i + 1) for i in range(M)]
        self.timestep = 1

    def schedule(self):
        if self.eager_evaluation:
            self.moduleoptimizer.config['lr'] = self.lrs[self.timestep]
        else:
            self.moduleoptimizer.config['lr'] = self.lr / (self.timestep + 1)
        #print("Timestep:", self.timestep, "Learning rate:", self.moduleoptimizer.config['lr'], "Original learning rate:", self.lr)
        self.timestep += 1

class ConstantScheduler:
    def __init__(self, eager_evaluation: bool = False):
        self.lr = None
        
    def reset(self, moduleoptimizer):
        self.moduleoptimizer = moduleoptimizer
        if self.lr is None:
            self.lr = self.moduleoptimizer.config['lr']
        else:
            self.moduleoptimizer.config['lr'] = self.lr

    def schedule(self):
        #print("Timestep:", self.timestep, "Learning rate:", self.moduleoptimizer.config['lr'], "Original learning rate:", self.lr)
        return