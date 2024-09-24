import os
from munch import Munch

from torchvision.datasets import CIFAR100
import torchvision.transforms as transforms

from snn_signgd.functional_config import FunctionalConfig
from src.dataset.cifar import Cutout, CIFAR10Policy
from src.model.torch_cifar.models import list_models, load_model

rootdir = os.path.join("..", "data", "CIFAR100")
num_classes = 100

model_config = FunctionalConfig(
    module = load_model,
    category_name = 'vit', 
    vgg_default = 'VGG16',
    index = 0,
    num_classes = num_classes
) 

transforms_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    CIFAR10Policy(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[n/255. for n in [129.3, 124.1, 112.4]], std=[n/255. for n in [68.2,  65.4,  70.4]]),
    Cutout(n_holes=1, length=8)
])
transforms_test = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize(mean=[n/255. for n in [129.3, 124.1, 112.4]], std=[n/255. for n in [68.2,  65.4,  70.4]])
])

dataset_gen = lambda mod: FunctionalConfig(
    module = CIFAR100,
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