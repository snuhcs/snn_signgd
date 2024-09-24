from snn_signgd.functional_config import FunctionalConfig, Munch
import torch.optim
import torch.optim.lr_scheduler
config = Munch(
    optimizer = FunctionalConfig(
        module = torch.optim.AdamW,
        submodules = [
            FunctionalConfig(
                module = torch.optim.lr_scheduler.ExponentialLR,
                gamma=0.99, 
            )
        ],
        lr = 1e-4,
        weight_decay = 1e-4
    ),
)