import argparse
import sys
import pathlib

from utilities.train_parser import TrainParser
from utilities.load_parser import LoadParser
from utilities.environment_creator import RetroEnvCreator
from utilities.model_manager import ModelManager
from utilities.wandb_manager import WandbManager
from callbacks import CustomEvalCallback
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
    # Load training parameters:
    train_command = None
    with open(f"{cfg.directory}/train_command.txt", 'r') as file:
        train_command = file.read()
    argv = train_command.split(" ") # convert arguments to list
    train_parser = TrainParser(arg_source=argv[1:]) # feed arguments into parser
    train_args = train_parser.get_args()
    
    # select optimal training hardware:
    device = torch.device("cuda") if train_args.device == "cuda" and torch.cuda.is_available() else torch.device("cpu")

    # initialize wandb run:
    path = pathlib.Path(cfg.directory)
    timestamp_original_run = path.stem # note: assumes we keep identifying logging folders by their timestamp
    if train_args.with_wandb:
        WandbManager.ResumeWandb(train_args, cfg.directory, timestamp_original_run)

    # create environment:
    env = RetroEnvCreator.create(train_args, cfg.directory, CONFIG)

    # Create callback:
    callback = None
    if train_args.save_best:
        callback = CustomEvalCallback(cfg=train_args, eval_env=env, 
                                                save_path=f"{cfg.directory}/best_model/",
                                                system_file_name=f"{train_args.game}_best",
                                                wandb_file_name=f"{train_args.game}_best",
                                                eval_freq=train_args.eval_freq)


    # Load the model:
    model = ModelManager.load_model(train_args.model, train_args.game, cfg.directory, env)
    
    # Determine number of timesteps
    timesteps = CONFIG[train_args.game]["timesteps"]
    if train_args.timesteps > 0:
        timesteps = train_args.timesteps

    # Train the model
    try:
        model.learn(total_timesteps=timesteps, reset_num_timesteps=False)
        model.save(f"{cfg.directory}/{train_args.game}.zip")
        # if training completes, upload last model to wandb.
        if cfg.with_wandb:
            wandb.save(f"{cfg.directory}/{train_args.game}.zip")
    except KeyboardInterrupt:
        model.save(f"{cfg.directory}/{train_args.game}-bak.zip")
        # upload last model to wandb:
        if cfg.with_wandb:
            wandb.save(f"{cfg.directory}/{train_args.game}-bak.zip")

if __name__ == '__main__':
    parser = LoadParser(arg_source=sys.argv[1:])
    args = parser.get_args()
    main(args)
