import os
from munch import Munch

from  torchvision.datasets import ImageNet

from snn_signgd.functional_config import FunctionalConfig
from src.model.imagenet import list_models, load_model

rootdir = os.path.join("..", "data", "ImageNet")
num_classes = 1000

print(list_models(verbose=True))
model_config, preprocessors = load_model(
    category_name = 'vit', 
    index = 0,
    num_classes = num_classes
)

def dataset_gen(mod):
    assert mod in ['train', 'val'], "Dataset mod not in 'train', 'val'"
    return FunctionalConfig(
    module = ImageNet,
    root = rootdir, 
    split = mod,
    transform = lambda: (_ for _ in ()).throw(ValueError('Dataset Transforms must be defined')), 
)
config = Munch(
    train_dataset = dataset_gen("train"),
    valid_dataset = dataset_gen("val"),
    test_dataset =  dataset_gen("val"),
    num_classes = num_classes,

    preprocessors = Munch(
        train = preprocessors['train'],
        test = preprocessors['test'],
    ),

    model = model_config,
)