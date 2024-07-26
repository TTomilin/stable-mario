import argparse
import sys
from utilities.train_parser import TrainParser
from utilities.environment_creator import RetroEnvCreator
from utilities.model_manager import ModelManager
from utilities.wandb_manager import WandbManager
from config import CONFIG

import os
from copy import copy
from datetime import datetime
from pathlib import Path

import wandb
import torch
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

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
    eval_env = Monitor(copy(env))
    eval_callback = EvalCallback(eval_env, best_model_save_path=f"{log_dir}/checkpoints", log_path=f"{log_dir}/logs",
                                 eval_freq=cfg.store_every, deterministic=True, render=False)

    # Create the model
    model = ModelManager.create_model(cfg, env, device, log_dir)
    
    # Determine number of timesteps
    timesteps = CONFIG[cfg.game]["timesteps"]
    if cfg.timesteps > 0:
        timesteps = cfg.timesteps

    # Train the model
    try:
        model.learn(total_timesteps=timesteps)
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
    main(args)
