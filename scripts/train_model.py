import argparse
import sys
import stable_retro.data
from utilities.train_parser import TrainParser
from utilities.environment_creator import RetroEnvCreator
from utilities.model_creator import ModelCreator
from config import CONFIG

import os
import numpy as np
from copy import copy
from datetime import datetime
from pathlib import Path

import wandb
import torch
from gymnasium.wrappers import ResizeObservation, NormalizeObservation, RecordVideo, FrameStack, NormalizeReward, TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import ClipRewardEnv, MaxAndSkipEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from sb3_contrib import QRDQN

def main(cfg: argparse.Namespace):
    experiment_dir = Path(__file__).parent.parent.resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"{experiment_dir}/saves/{cfg.game}/{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    device = torch.device("cuda") if cfg.device == "cuda" and torch.cuda.is_available() else torch.device("cpu")
    game = cfg.game

    with open(f"{log_dir}/train_command.txt", "w") as commandFile:
        commandFile.write(' '.join(sys.argv[0:])) # save the training command (needed for model re-initialization)

    if cfg.with_wandb:
        init_wandb(cfg, log_dir, timestamp)

    # create environment:
    env = RetroEnvCreator.create(cfg, log_dir, CONFIG)

    # Create a callback to save best model
    eval_env = Monitor(copy(env))
    eval_callback = EvalCallback(eval_env, best_model_save_path=f"{log_dir}/checkpoints", log_path=f"{log_dir}/logs",
                                 eval_freq=cfg.store_every, deterministic=True, render=False)

    # Create the model
    model = ModelCreator.CreateModel(cfg, env, device, log_dir)
    
    # Determine number of timesteps
    timesteps = CONFIG[game]["timesteps"]
    if cfg.timesteps > 0:
        timesteps = cfg.timesteps

    # Train the model
    try:
        model.learn(total_timesteps=timesteps)
        model.save(f"{log_dir}/{game}.zip")
    except KeyboardInterrupt:
        model.save(f"{log_dir}/{game}-bak.zip")

    # upload best and most recent models to wandb:
    if cfg.with_wandb:
        wandb.save(f"{log_dir}/models/{game}-bak.zip")


def init_wandb(cfg: argparse.Namespace, log_dir: str, timestamp: str) -> None:
    if cfg.wandb_key:
        wandb.login(key=cfg.wandb_key)
    wandb_group = cfg.wandb_group if cfg.wandb_group is not None else cfg.game
    wandb_job_type = cfg.wandb_job_type if cfg.wandb_job_type is not None else "PPO"
    wandb_unique_id = f'{wandb_job_type}_{wandb_group}_{timestamp}'
    wandb.init(
        dir=log_dir,
        monitor_gym=True,
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        sync_tensorboard=True,
        id=wandb_unique_id,
        name=wandb_unique_id,
        group=wandb_group,
        job_type=wandb_job_type,
        tags=cfg.wandb_tags,
        resume=False,
        settings=wandb.Settings(start_method='fork'),
        reinit=True
    )
    wandb.define_metric(name='eval/mean_reward', step_metric='global_step')
    wandb.define_metric(name='eval/mean_ep_length', step_metric='global_step')

if __name__ == '__main__':
    parser = TrainParser(arg_source=sys.argv[1:])
    args = parser.get_args()
    main(args)
