from snn_signgd.functional_config import FunctionalConfig
def ResMLPBig24():
    import timm
    model_name = 'resmlp_big_24_224.fb_in22k_ft_in1k'
    model_config = FunctionalConfig(
        module = timm.create_model,
        model_name = model_name, 
        pretrained=True
    )

    data_config = {
        'input_size': (3, 224, 224), 
        'interpolation': 'bicubic', 
        'mean': (0.485, 0.456, 0.406), 
        'std': (0.229, 0.224, 0.225), 
        'crop_pct': 0.875, 'crop_mode': 'center'
    }
    transforms = {
        'train': timm.data.create_transform(**data_config, is_training=True),
        'test': timm.data.create_transform(**data_config, is_training=False)
    }
    return model_config, transforms

