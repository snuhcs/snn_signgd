from torchvision.models import VisionTransformer
from snn_signgd.functional_config import FunctionalConfig
from torch import nn

class ViTB16(nn.Module):
    def __init__(self, 
            image_size=32,
            patch_size=4,
            num_layers=4,
            num_heads=8,
            hidden_dim=256,
            mlp_dim=256,
            num_classes = 10,
            dropout = 0.1,
        ) -> None:
        super().__init__()

        self.model = VisionTransformer(
            image_size = image_size, 
            patch_size=patch_size, 
            num_layers=num_layers, 
            num_heads=num_heads, 
            hidden_dim=hidden_dim, 
            mlp_dim=mlp_dim, 
            num_classes=num_classes,
            dropout = dropout,
        ) 

    def forward(self, x):
        return self.model(x)

