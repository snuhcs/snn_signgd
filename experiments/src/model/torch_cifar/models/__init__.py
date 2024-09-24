from .vgg import *
from .dpn import *
from .lenet import *
from .senet import *
from .pnasnet import *
from .densenet import *
from .googlenet import *
from .shufflenet import *
from .shufflenetv2 import *
from .resnet import *
from .resnext import *
from .preact_resnet import *
from .mobilenet import *
from .mobilenetv2 import *
from .efficientnet import *
from .regnet import *
from .dla_simple import *
from .dla import *
from .vit import ViTB16

categories = {
    'vgg': [VGG],  
    'dpn': [DPN26, DPN92], 
    'lenet': [LeNet], 
    'senet': [SENet18], 
    'pnasnet': [PNASNetA, PNASNetB], 
    'densenet':  [DenseNet121, DenseNet169, DenseNet201, DenseNet161, densenet_cifar], 
    'googlenet':  [GoogLeNet], 
    'shufflenet': [ ShuffleNetG2, ShuffleNetG3], 
    'shufflenetv2': [ShuffleNetV2], 
    'resnet': [ ResNet18, ResNet34, ResNet50, ResNet101, ResNet152], 
    'resnext': [ ResNeXt29_2x64d, ResNeXt29_4x64d, ResNeXt29_8x64d, ResNeXt29_32x4d], 
    'preact_resnet': [ PreActResNet18, PreActResNet34, PreActResNet50, PreActResNet101, PreActResNet152], 
    'mobilenet': [ MobileNet], 
    'mobilenetv2': [ MobileNetV2], 
    'efficientnet':  [ EfficientNetB0], 
    'regnet': [ RegNetX_200MF, RegNetX_400MF, RegNetY_400MF], 
    'dla_simple': [ SimpleDLA], 
    'dla': [ DLA],
    'vit': [ViTB16],
}
def list_models(verbose = False):
    if verbose:
        print("<Category Name>: [<Model Name>, ...] (enumerable)")
        for category, models in categories.items():
            print(f" {category}: [",','.join([model.__name__ for model in models]), "]")
    return list(categories.keys())

def load_model(category_name:str, index:int, vgg_default:str = 'VGG11', shufflenet_size_default:str = 0.5, num_classes = 10):
    print(f"\nLoading DNN model for CIFAR-{num_classes} dataset")
    
    assert category_name in categories, "DNN Category not available"
    
    print("Available model instances for category", category_name, ":")
    index = index % len(categories[category_name])
    for class_index, model_class in enumerate(categories[category_name]):
        prefix = "*" if class_index == index else " "
        print(f'\t{prefix}[{class_index}] {model_class.__name__}')
    
    model_class = categories[category_name][index]
    if category_name == 'vgg':
        kwargs = {'vgg_name' : vgg_default, 'num_classes': num_classes}
    elif category_name == 'shufflenetv2':
        kwargs = {'net_size':shufflenet_size_default, 'num_classes' : num_classes}       
    elif category_name in ['googlenet', 'lenet','densenet']:
        kwargs = {}
    else:
        kwargs = {'num_classes': num_classes}
    model =  model_class(**kwargs)
    return model