import pytorch_lightning as pl
import argparse, os, glob
from snn_signgd.functional_config import import_config_from_path
from src.reproducibility import seed_all

def train(config, ckpt_path, **kwargs): 
    trainer = pl.Trainer(
        precision=16, 
        max_epochs = 1000,
        accelerator="gpu", 
        #gradient_clip_val=1.0,
        num_sanity_val_steps=2,
        **kwargs
    )

    boilerplate = config.task(config = config)    
    trainer.fit( model=boilerplate , ckpt_path = ckpt_path )
    return boilerplate, trainer

def lightning_setup(config, run_name):
    seed_all(config.seed)

    loggers = []    
    log_dir = os.path.join(".", "logs")

    loggers.append(
        pl.loggers.WandbLogger(
            project = "snn_signgd",
            save_dir = log_dir,
            config = config,
            name = run_name,
            log_model = 'all',
        )
    )
    print("Config:", config)
    
    ckpt_dir = os.path.join("resources", "checkpoints", run_name)
    callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath = ckpt_dir,
            save_top_k = 2, 
            monitor = 'performance',
            filename = '{epoch:02d}-{performance:.4f}',
            mode = 'max',
        )
    ]
    ckpt_path = max(glob.glob(os.path.join(ckpt_dir, "*.ckpt")), 
                    key=os.path.getctime, default = None)
    
    return loggers, callbacks, ckpt_path

def pipeline(run_name, config, devices):
    loggers, callbacks, ckpt_path = lightning_setup(config, run_name)
    train(config, ckpt_path, 
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