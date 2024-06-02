import argparse
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

from sb3_contrib import QRDQN

import stable_retro
from config import CONFIG
from stable_retro.examples.discretizer import Discretizer
from stable_retro.examples.ppo import StochasticFrameSkip
from wrappers.observation import Rescale
from wrappers.observation import ShowObservation


def main(cfg: argparse.Namespace):
    experiment_dir = Path(__file__).parent.parent.resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"{experiment_dir}/trained_footage/{cfg.game}/{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    device = torch.device("cuda") if cfg.device == "cuda" and torch.cuda.is_available() else torch.device("cpu")
    
    # Create environment
    game = cfg.game
    state = cfg.load_state if cfg.load_state is not None else CONFIG[game]["state"]
    env = stable_retro.make(game=CONFIG[game]['game_env'], state=state, render_mode=cfg.render_mode)
    if cfg.discretize:
        env = Discretizer(env, CONFIG[game]["actions"])
    if cfg.resize_observation:
        env = ResizeObservation(env, CONFIG[game]["resize"])
    if cfg.rescale:
        env = Rescale(env)
    if cfg.normalize_observation:
        env = NormalizeObservation(env)
    if cfg.normalize_reward:
        env = NormalizeReward(env)
    if cfg.show_observation:
        env = ShowObservation(env);
    if cfg.skip_frames:
        env = MaxAndSkipEnv(env, skip=cfg.n_skip_frames)
    if cfg.stack_frames:
        env = FrameStack(env, cfg.n_stack_frames)
    if CONFIG[game]["clip_reward"]:
        env = ClipRewardEnv(env)
    if cfg.record:
        video_folder = f"{log_dir}/videos"
        env = RecordVideo(env=env, video_folder=video_folder, episode_trigger=lambda x: x % cfg.record_every == 0)

    # Create a callback to save best model
    eval_env = Monitor(copy(env))
    eval_callback = EvalCallback(eval_env, best_model_save_path=f"{log_dir}/checkpoints", log_path=f"{log_dir}/logs",
                                 eval_freq=cfg.store_every, deterministic=True, render=False)

    # Create the model
    model = None
    if cfg.model == "PPO":
        model = PPO(policy='CnnPolicy', env=env, device=device, ent_coef=cfg.ent_coeff,
                    learning_rate=cfg.learning_rate,verbose=True, tensorboard_log=f"{log_dir}/tensorboard/")
    elif cfg.model == "QR-DQN":
        model = QRDQN(policy='CnnPolicy', env=env, device=device,
                    learning_rate=cfg.learning_rate,verbose=True, tensorboard_log=f"{log_dir}/tensorboard/")
    else:
        print("No model matching the model argument found. Aborting...")
        exit()
    
    # Determine number of timesteps
    timesteps = CONFIG[game]["timesteps"]
    if cfg.timesteps > 0:
        timesteps = cfg.timesteps
        

    # Train the model
    try:
        model.learn(total_timesteps=timesteps, callback=eval_callback if cfg.store_model else None)
        model.save(f"{log_dir}/{game}")
    except KeyboardInterrupt:
        model.save(f"{log_dir}/{game}-bak.zip")

    # upload best and most recent models to wandb:
    if cfg.store_model and cfg.with_wandb:
        wandb.save(f"{log_dir}/checkpoints/*")
        wandb.save(f"{log_dir}/{game}*")

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
    def arg(*args, **kwargs):
        parser.add_argument(*args, **kwargs)


    parser = argparse.ArgumentParser()
    
    arg("--path", type=str, help="Path to model's zip file")
    arg("--game", type=str, default="broom_zoom", help="Name of the game")
    arg("--render_mode", default="rgb_array", choices=["human", "rgb_array"], help="Render mode")
    arg("--load_state", type=str, default=None, help="Path to the game save state to load")
    arg("--record", default=True, action='store_true', help="Whether to record gameplay videos")
    arg("--record_every", type=int, default=150, help="Record gameplay video every n episodes")

    args = parser.parse_args()
    main(args)
