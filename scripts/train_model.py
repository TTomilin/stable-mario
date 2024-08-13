import argparse
import sys

from utilities.train_parser import TrainParser
from utilities.environment_creator import RetroEnvCreator
from utilities.model_manager import ModelManager
from utilities.wandb_manager import WandbManager
from config import CONFIG
from callbacks import CustomEvalCallback

import os
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import wandb
import torch

def main(cfg: argparse.Namespace):
    # create logging directory:
    experiment_dir = Path(__file__).parent.parent.resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"{experiment_dir}/saves/{cfg.game}/{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    
    # select optimal training hardware:
    device = torch.device("cuda") if cfg.device == "cuda" and torch.cuda.is_available() else torch.device("cpu")
    
    # save the exact training command to a textfile in the logging directory:
    with open(f"{log_dir}/train_command.txt", "w") as commandFile:
        commandFile.write(' '.join(sys.argv[0:])) # save the training command (needed for model re-initialization)

    # initialize wandb run:
    if cfg.with_wandb:
        WandbManager.InitializeWandb(cfg, log_dir, timestamp)

    # create environment:
    env = RetroEnvCreator.create(cfg, log_dir, CONFIG)

    # Create a callback to save best model
    callback = None
    if cfg.save_best:
        callback = CustomEvalCallback(cfg=cfg, eval_env=env, 
                                                log_dir=log_dir,
                                                device=device,
                                                system_file_name=f"{cfg.game}_best",
                                                wandb_file_name=f"{cfg.game}_best",
                                                eval_freq=cfg.eval_freq)

    # Create the model
    model = ModelManager.create_model(cfg, env, device, log_dir)
    print(model.policy.mlp_extractor.value_net)

    # Determine number of timesteps
    timesteps = CONFIG[cfg.game]["timesteps"]
    if cfg.timesteps > 0:
        timesteps = cfg.timesteps

    # Train the model
    try:
        model.learn(total_timesteps=timesteps, callback=callback)
        model.save(f"{log_dir}/{cfg.game}.zip")
        # if training completes, upload last model to wandb.
        if cfg.with_wandb:
            wandb.save(f"{log_dir}/{cfg.game}.zip")
    except KeyboardInterrupt:
        model.save(f"{log_dir}/{cfg.game}-bak.zip")
        # upload last model to wandb:
        if cfg.with_wandb:
            wandb.save(f"{log_dir}/{cfg.game}-bak.zip")

if __name__ == '__main__':
    parser = TrainParser(arg_source=sys.argv[1:])
    args = parser.get_args()
    parser.validate_args(args)
    
    main(args)
