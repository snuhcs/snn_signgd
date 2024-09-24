from .torchvision import *
from .resmlp import *
from .mlp_mixer_wrap import *

categories = {
    'vgg': [VGG16_BN],  
    'resnet': [ ResNet18, ResNet34], 
    'mobilenetv2': [ MobileNetV2], 
    'resmlp':  [ResMLPBig24], 
    'mlpmixer': [ MLPMixerB16], 
    'convnext': [ ConvNeXt_small], 
    'vit':[ViT_B_16],
}
def list_models(verbose = False):
    if verbose:
        for category, models in categories.items():
            print(f" {category}: [",','.join([model.__name__ for model in models]), "]")
    return list(categories.keys())

def load_model(category_name:str, index:int, num_classes = 1000):
    print(f"\nLoading DNN model for ImageNet-{num_classes} dataset")
    
    assert category_name in categories, "DNN Category not available"
    
    print("Available model instances for category", category_name, ":")
    index = index % len(categories[category_name])
    for class_index, model_class in enumerate(categories[category_name]):
        prefix = "*" if class_index == index else " "
        print(f'\t{prefix}[{class_index}] {model_class.__name__}')
    
    model_class = categories[category_name][index]
    if category_name == 'mlpmixer':
        kwargs = {'num_classes': num_classes}     
    else:
        kwargs = {}
    model =  model_class(**kwargs)
    return model