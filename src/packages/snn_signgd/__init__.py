from .conversion import convert
from .neuronal_dynamical_system.spikingjelly.generalization.sgd import SGDModule
from .neuronal_dynamical_system.spikingjelly.generalization.schedulers import ConstantScheduler, ExponentialScheduler, InverseScheduler
from .neuronal_dynamical_system.spikingjelly.generalization import construct_spiking_neurons_for_operators
from .neuronal_dynamical_system.spikingjelly import subgradient
from .functional_config import FunctionalConfig

from torch.utils.data import DataLoader

def setup(stage, config, model):
    if stage in ['test', 'predict'] and stage not in ['fit', 'validate']:

        train_dataset = config.train_dataset(transform = config.preprocessors.train)
        train_dataloader = DataLoader(
            train_dataset, batch_size= config.batch_size, 
            shuffle=True, drop_last=False, num_workers=4, pin_memory=True
        )
        snn_model, ported_ann_model, sample = convert(
            ann_model = model, 
            neuronal_dynamics = config.neuronal_dynamics,
            dynamics_type = config.dynamics_type,
            default_simulation_length = config.default_simulation_length, 
            activation_scale_dataloader = train_dataloader,
            max_activation_scale_iterations = config.max_activation_scale_iterations,
            scale_relu_with_max_activation = config.scale_relu_with_max_activation,
        )
        
        reference_ann_model, device = ported_ann_model, None
    
        return snn_model, ported_ann_model, model, device
    else:
        return model