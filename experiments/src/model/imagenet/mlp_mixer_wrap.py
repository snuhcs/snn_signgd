import os
from snn_signgd.functional_config import FunctionalConfig

def MLPMixerB16(num_classes):
    from .mlp_mixer import MLPMixer, mixer_transforms
    model_config = FunctionalConfig(
        module = MLPMixer,
        name = "Mixer-B_16", 
        weight_path = os.path.join('src','model','imagenet','mlp_mixer', "checkpoints","imagenet1k_Mixer-B_16.npz"),
        img_size = 224, 
        num_classes=num_classes, 
        patch_size=16, 
        zero_head=False,
    )
    transforms = {
        'train' : mixer_transforms,
        'test' : mixer_transforms,
    }
    return model_config, transforms