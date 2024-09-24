from torchvision.datasets import MNIST
import os
from munch import Munch
from snn_signgd.functional_config import FunctionalConfig

rootdir = os.path.join("..", "data", "MNIST")
dataset_gen = lambda mod: FunctionalConfig(
    module = MNIST,
    root = rootdir, 
    train = True if mod == "train" else False, 
    download = True,
    transform = lambda: (_ for _ in ()).throw(ValueError('Dataset Transforms must be defined')), 
)
config = Munch(
    train_dataset = dataset_gen("train"),
    valid_dataset = dataset_gen("val"),
    test_dataset =  dataset_gen("val"),
)