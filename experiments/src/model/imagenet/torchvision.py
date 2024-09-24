from snn_signgd.functional_config import FunctionalConfig

def _torchvision_models_api_wrapper(model, weights):
    transforms = {
        'train': weights.transforms(),
        'test': weights.transforms(),
    }
    return FunctionalConfig(module = model, weights = weights), transforms

def ResNet34():
    from torchvision.models import resnet34, ResNet34_Weights
    return _torchvision_models_api_wrapper(resnet34, weights = ResNet34_Weights.DEFAULT)

def ResNet18():
    from torchvision.models import resnet18, ResNet18_Weights
    return _torchvision_models_api_wrapper(resnet18, weights = ResNet18_Weights.DEFAULT)

def MobileNetV2():
    from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
    return _torchvision_models_api_wrapper(mobilenet_v2, weights = MobileNet_V2_Weights.IMAGENET1K_V2)

def VGG16_BN():
    from torchvision.models import vgg16_bn, VGG16_BN_Weights
    return _torchvision_models_api_wrapper(vgg16_bn, weights = VGG16_BN_Weights.IMAGENET1K_V1)

def ConvNeXt_small():
    from torchvision.models import convnext_small, ConvNeXt_Small_Weights
    return _torchvision_models_api_wrapper(convnext_small, weights = ConvNeXt_Small_Weights.IMAGENET1K_V1)

def ViT_B_16():
    from torchvision.models import vit_b_16, ViT_B_16_Weights
    return _torchvision_models_api_wrapper(vit_b_16, weights = ViT_B_16_Weights.IMAGENET1K_V1)