from snn_signgd.functional_config import FunctionalConfig, Munch
import torch.optim
import torch.optim.lr_scheduler
config = Munch(
    optimizer = FunctionalConfig(
        module = torch.optim.Adam,
        lr = 1e-4,
    ),
    lr_scheduler = FunctionalConfig(
        module = torch.optim.lr_scheduler.LinearLR,
        start_factor=1.0, 
        end_factor=0.5,
        total_iters=100, 
    )
)