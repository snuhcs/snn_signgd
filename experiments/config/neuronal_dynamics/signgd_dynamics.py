from snn_signgd.functional_config import FunctionalConfig, Munch
from snn_signgd import SGDModule, ExponentialScheduler, construct_spiking_neurons_for_operators, setup #, ConstantScheduler, InverseScheduler
    
neuronal_dynamics_per_ops = construct_spiking_neurons_for_operators(
    moduleoptimizer_cfg = FunctionalConfig(
        module = SGDModule,
        lr = 0.15,
        inplace = False,
    ),
    lr_scheduler_cfg = FunctionalConfig(
        module = ExponentialScheduler,
        gamma = 0.95,
        eager_evaluation = True,
    ),
)

config = Munch(
    dynamics_type = 'signgd',

    default_simulation_length = 32,
    max_activation_scale_iterations = 10,
    scale_relu_with_max_activation = True,

    neuronal_dynamics = neuronal_dynamics_per_ops,

    setup = setup,
)
