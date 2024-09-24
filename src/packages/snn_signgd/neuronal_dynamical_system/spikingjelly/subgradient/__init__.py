from .relu import Subgradient_ReLU

from snn_signgd.functional_config import FunctionalConfig, Munch
from .membrane_equations import Neuron, Codec, correction
from .learning_rate_scheduler import InverseLR, ConstantLR, ExponentialLR

def construct_spiking_neurons_for_operators(neuronal_dynamics_specification):
    output = Munch(
        neuron = FunctionalConfig(
                    module = Neuron,
                    subgradient_function = Subgradient_ReLU,
                    **neuronal_dynamics_specification
                ),
        codec = FunctionalConfig(
                    module = Codec,
                    choice = 'float', 
                    subgradient_function = Subgradient_ReLU,
                    **neuronal_dynamics_specification
                ),
        correction = FunctionalConfig(
            module = correction,
        ),
    )
    return output
