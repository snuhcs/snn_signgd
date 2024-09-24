from snn_signgd.functional_config import FunctionalConfig, Munch, import_config_from_path
from src.task.downstream.image_classification_snn import ImageClassification

config = import_config_from_path(
    "config/datasets_and_models/cifar10.py", 
    "config/optimizers/optim_sgd.py",
    "config/neuronal_dynamics/signgd_dynamics.py"
)

config.update(Munch(
    seed=41,
    batch_size = 256,
    epochs = 300,

    noload = False,
    
    timestamps = [32,64,128],

    verbose = True,

    task = FunctionalConfig(
        module=ImageClassification
    ),
))
