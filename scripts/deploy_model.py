import sys

import argparse
from utilities.load_parser import LoadParser
from utilities.train_parser import TrainParser

import os
from copy import copy
from datetime import datetime
from pathlib import Path

import wandb
import torch
from gymnasium.wrappers import ResizeObservation, NormalizeObservation, RecordVideo, FrameStack, NormalizeReward
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import ClipRewardEnv, MaxAndSkipEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from sb3_contrib import QRDQN

import stable_retro
from config import CONFIG
from stable_retro.examples.discretizer import Discretizer
from stable_retro.examples.ppo import StochasticFrameSkip
from wrappers.observation import Rescale
from wrappers.observation import ShowObservation


def main(cfg: argparse.Namespace):
    # Create directory to store footage of trained model:
    load_directory = cfg.directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"{load_directory}/trained_footage/{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    
    # Load training parameters:
    train_command = None
    with open(f"{load_directory}/train_command.txt", 'r') as file:
        train_command = file.read()
    argv = train_command.split(" ") # convert arguments to list
    train_parser = TrainParser(arg_source=argv[1:]) # feed arguments into parser
    train_args = train_parser.get_args()

    # Create environment
    game = train_args.game
    state = train_args.load_state if train_args.load_state is not None else CONFIG[game]["state"]
    env = stable_retro.make(game=CONFIG[game]['game_env'], state=state, render_mode="human")

    if train_args.discretize:
        env = Discretizer(env, CONFIG[game]["actions"])
    if train_args.resize_observation:
        env = ResizeObservation(env, CONFIG[game]["resize"])
    if train_args.rescale:
        env = Rescale(env)
    if train_args.normalize_observation:
        env = NormalizeObservation(env)
    if train_args.normalize_reward:
        env = NormalizeReward(env)
    if train_args.show_observation:
        env = ShowObservation(env);
    if train_args.skip_frames:
        env = MaxAndSkipEnv(env, skip=train_args.n_skip_frames)
    if train_args.stack_frames:
        env = FrameStack(env, train_args.n_stack_frames)
    if CONFIG[game]["clip_reward"]:
        env = ClipRewardEnv(env)

    # Load the model
        model = None
    if train_args.model == "PPO":
        model = PPO.load(f"{load_directory}/model", env=env)
    elif train_args.model == "QR-DQN":
        model = QRDQN.load(f"{load_directory}/model", env=env)
    else:
        print("No model matching the model argument found. Aborting...")
        exit()

    # Show the model
    obs, _ = env.reset()
    while True:
        env.render()
        if train_args.discretize:
            action = model.predict(obs)[0] # Model's action are returned as tuple with one element. Corresponds to discretized action.
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            obs, _ = env.reset()

def init_wandb(cfg: argparse.Namespace, log_dir: str, timestamp: str) -> None:
    if args.wandb_key:
        wandb.login(key=args.wandb_key)
    wandb_group = cfg.wandb_group if cfg.wandb_group is not None else cfg.game
    wandb_job_type = cfg.wandb_job_type if cfg.wandb_job_type is not None else "PPO"
    wandb_unique_id = f'{wandb_job_type}_{wandb_group}_{timestamp}'
    wandb.init(
        dir=log_dir,
        monitor_gym=True,
        project=args.wandb_project,
        entity=args.wandb_entity,
        sync_tensorboard=True,
        id=wandb_unique_id,
        name=wandb_unique_id,
        group=wandb_group,
        job_type=wandb_job_type,
        tags=args.wandb_tags,
        resume=False,
        settings=wandb.Settings(start_method='fork'),
        reinit=True
    )


if __name__ == '__main__':
    parser = LoadParser(arg_source=sys.argv[1:])
    args = parser.get_args()

    main(args)
