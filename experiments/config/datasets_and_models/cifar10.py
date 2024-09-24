import os
from munch import Munch

import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10

from snn_signgd.functional_config import FunctionalConfig
from src.dataset.cifar import Cutout, CIFAR10Policy
from src.model.torch_cifar.models import list_models, load_model

rootdir = os.path.join("..","data", "CIFAR10")
num_classes = 10

print(list_models(verbose=True))
model_config = FunctionalConfig(
    module = load_model,
    category_name = 'resnet', 
    vgg_default = 'VGG16',
    index = 0
)

transforms_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    CIFAR10Policy(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    Cutout(n_holes=1, length=16)
])
transforms_test = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

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

    preprocessors = Munch(
        train = transforms_train,
        test = transforms_test,
    ),

    num_classes = num_classes,

    model = model_config,
)