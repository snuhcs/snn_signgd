import os
from munch import Munch

from torchvision.datasets import CIFAR10

from snn_signgd.functional_config import FunctionalConfig

rootdir = os.path.join("..","data", "CIFAR10")
num_classes = 10

dataset_gen = lambda mod: FunctionalConfig(
    module = CIFAR10,
    root = rootdir, 
    train = True if mod == "train" else False, 
    download = True,
    transform = lambda: (_ for _ in ()).throw(ValueError('Dataset Transforms must be defined')), 
)
config = Munch(
    train_dataset = dataset_gen("train"),
    valid_dataset = dataset_gen("val"),
    test_dataset =  dataset_gen("test"),
    num_classes = num_classes,
)