import sys

def seed_torch(seed):
    import torch
    torch.manual_seed(seed)
def seed_random(seed):
    import random
    random.seed(seed)
def seed_numpy(seed):
    import numpy
    numpy.random.seed(seed)


seeders = {
    'torch': seed_torch,
    'random': seed_random,
    'numpy': seed_numpy,
}

def seed_all(seed):
    for modulename, seeder in seeders.items():
        if modulename in sys.modules:
            print("Seeding", modulename, "with seed", seed)
            seeder(seed)
        else:
            print("Failed to seed", modulename, "because it is not imported")
    return