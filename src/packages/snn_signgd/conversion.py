import torch
from torch.utils.data import DataLoader

from .snn import SpikingNeuralNetwork 
from .port_ann import porting 

from snn_signgd.pretty_printer import print

#def convert(config, ann_model, sample_size = None):
'''
    train_dataset = dataset(transform = data_preprocessor)
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, 
        shuffle=True, drop_last=False, num_workers=4, pin_memory=True
    )
'''
def convert(
        ann_model, 
        neuronal_dynamics, dynamics_type:str, default_simulation_length:int, 
        activation_scale_dataloader:DataLoader, max_activation_scale_iterations:int,
        scale_relu_with_max_activation:bool, sample_size = None
):
    assert dynamics_type in ['subgradient', 'signgd'], f"Unknown dynamics type: {dynamics_type}. Supported types are ['subgradient', 'signgd']"

    print("<cyan> Starting ANN to SNN Conversion Process </cyan>")
    ann_model.eval()
    
    if sample_size is not None:
        sample = torch.randn(sample_size)
    else:
        sample, _ = next(iter(activation_scale_dataloader))    

    sample = sample.to(next(ann_model.parameters()).device)
    
    print("<cyan> Porting ANN to Enable Conversion  </cyan>")
    snn_compatible_ann_model = porting(
        model = ann_model,
        dataloader = activation_scale_dataloader, 
        max_activation_iterations = max_activation_scale_iterations,
        scale_relu_with_max_activation = scale_relu_with_max_activation,
    )
    
    print("<cyan> Converting ANN to SNN </cyan>")
    snn_model = SpikingNeuralNetwork(
        ann_model = snn_compatible_ann_model, 
        config = neuronal_dynamics,
        default_simulation_length= default_simulation_length, 
        sample_data = sample, 
        dynamics_type = dynamics_type
    )
    
    print("<cyan> Finished ANN to SNN Conversion Process! </cyan>")

    return snn_model, snn_compatible_ann_model, sample