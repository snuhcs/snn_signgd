from snn_signgd.functional_config import FunctionalConfig, Munch
from snn_signgd import subgradient, setup

neuronal_dynamics = Munch(
    submodules = Munch(
        learning_rate_scheduler = FunctionalConfig(
            module = subgradient.InverseLR, #CONSTANT
        ),
    ),
    initial_learning_rate = 1.0, #CONSTANT
    momentum = 0.0, #CONSTANT

    nesterov_momentum = False,
    weight_decay = 0.0,
)
neuronal_dynamics_per_ops = subgradient.construct_spiking_neurons_for_operators(neuronal_dynamics)    


config = Munch(
    dynamics_type = 'subgradient',

    default_simulation_length = 32,
    max_activation_scale_iterations = 10,
    scale_relu_with_max_activation = True,

    neuronal_dynamics = neuronal_dynamics_per_ops,

    setup = setup,
)
