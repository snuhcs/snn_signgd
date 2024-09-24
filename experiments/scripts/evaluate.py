import pytorch_lightning as pl
import argparse, os, glob
from snn_signgd.functional_config import import_config_from_path
from src.reproducibility import seed_all
import torch

def test(config, ckpt_path, devices, **kwargs): 
    trainer = pl.Trainer(
        precision=16, 
        accelerator="gpu", 
        devices = devices,
        limit_test_batches= 2,
        **kwargs
    )

    if config.noload:
        boilerplate = config.task(config = config)   
        boilerplate.model = boilerplate.model.to(f'cuda:{devices[0]}') 
    else:
        boilerplate = config.task.module.load_from_checkpoint(
            ckpt_path, config = config,
            map_location = torch.device(f'cuda:{devices[0]}')
        )
        
    trainer.test( model=boilerplate)
    return boilerplate, trainer

def lightning_setup(config, run_name):
    seed_all(config.seed)

    loggers = []    
    log_dir = os.path.join(".", "logs")

    loggers.append(
        pl.loggers.CSVLogger(
            log_dir,
            name = run_name,
            prefix = 'test'
        )
    )
    
    ckpt_dir = os.path.join("resources", "checkpoints", run_name)
    callbacks = []
    ckpt_path = max(glob.glob(os.path.join(ckpt_dir, "*.ckpt")), 
                    key=os.path.getctime, default = None)
    
    return loggers, callbacks, ckpt_path

def pipeline(run_name, config, devices):
    loggers, callbacks, ckpt_path = lightning_setup(config, run_name)
    test(config, ckpt_path, 
        devices = devices,
        logger = loggers,
        callbacks = callbacks,
    )   

def get_commandline_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True) 
    parser.add_argument("-gpu", "--gpu", type=str, required=True) 
    args = parser.parse_args()
    
    run_name = os.path.splitext(os.path.basename(args.config))[0]
    config = import_config_from_path(args.config)
    
    gpus = [int(n) for n in args.gpu.split(",")]
    return config, run_name, gpus


if __name__ == "__main__":
    config, run_name, gpus = get_commandline_arguments()
    pipeline(run_name = run_name, config = config, devices = gpus)