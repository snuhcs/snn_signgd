from snn_signgd.functional_config import FunctionalConfig, Munch
import torch.optim
import torch.optim.lr_scheduler
config = Munch(
    optimizer = FunctionalConfig(
        module = torch.optim.Adam,
        lr = 1e-4,
    ),
    lr_scheduler = FunctionalConfig(
        module = torch.optim.lr_scheduler.CosineAnnealingLR,
        eta_min=0,
        T_max=300, 
    ),
)