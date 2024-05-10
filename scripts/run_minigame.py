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

import stable_retro
from config import CONFIG
from stable_retro.examples.discretizer import Discretizer
from stable_retro.examples.ppo import StochasticFrameSkip
from wrappers.observation import Rescale
from wrappers.observation import ShowObservation


def main(cfg: argparse.Namespace):
    experiment_dir = Path(__file__).parent.parent.resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"{experiment_dir}/saves/{cfg.game}/{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    device = torch.device("cuda") if cfg.device == "cuda" and torch.cuda.is_available() else torch.device("cpu")

    if cfg.with_wandb:
        init_wandb(cfg, log_dir, timestamp)

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
        env = FrameStack(env, cfg.n_frame_stack)
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
    model = PPO(policy='CnnPolicy', env=env, device=device, ent_coef=cfg.ent_coeff,
                learning_rate=cfg.learning_rate,verbose=True, tensorboard_log=f"{log_dir}/tensorboard/")
    
    # Determine number of timesteps
    timesteps = CONFIG[game]["timesteps"]
    if cfg.time_steps > 0:
        timesteps = cfg.time_steps
        

    # Train the model
    try:
        model.learn(total_timesteps=timesteps, callback=eval_callback if cfg.store_model else None)
        model.save(f"{log_dir}/{game}")
    except KeyboardInterrupt:
        model.save(f"{log_dir}/{game}-bak")


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
    
    arg("--device", default="cuda", type=str, choices=["cuda", "cpu"], help="Device to use")
    arg("--game", type=str, default="broom_zoom", help="Name of the game")
    arg("--render_mode", default="rgb_array", choices=["human", "rgb_array"], help="Render mode")
    arg("--load_state", type=str, default=None, help="Path to the game save state to load")
    arg("--record", default=False, action='store_true', help="Whether to record gameplay videos")
    arg("--record_every", type=int, default=50, help="Record gameplay video every n episodes")
    arg("--store_model", default=False, action='store_true', help="Whether to record gameplay videos")
    arg("--store_every", type=int, default=100, help="Save model every n episodes")
    arg("--skip_frames", default=False, action='store_true', help="Whether to skip frames")
    arg("--n_skip_frames", type=int, default=4, help="How many frames to skip")
    arg("--stack_frames", default=False, action='store_true', help="Whether to stack frames")
    arg("--n_stack_frames", type=int, default=4, help="How many frames to stack")
    arg("--show_observation", default=False, action='store_true', help="Show AI's observation.")
    arg("--normalize_reward", default=False, action='store_true', help="Normalize agent reward.")
    arg("--normalize_observation", default=False, action='store_true', help="Normalize agent observations.")
    arg("--resize_observation", default=False, action='store_true', help="Resize agent's observation to size specified in config.")
    arg("--rescale", default=False, action='store_true', help="Allow a modular transformation of the step and reset methods.")
    arg("--discretize", default=False, action='store_true', help="Limit agent's actions as specified in config.")
    arg("--learning_rate", type=float, default=0.0003, help="Set model's learning rate.")
    arg("--ent_coeff", type=float, default=0.05, help="Set entropy coefficient")
    arg("--timesteps", type=int, default=0, help="Number of timesteps the agent should train for.")

    # WandB
    arg('--with_wandb', default=False, action='store_true', help='Enables Weights and Biases')
    arg('--wandb_entity', default='automated-play', type=str, help='WandB username (entity).')
    arg('--wandb_project', default='Mario', type=str, help='WandB "Project"')
    arg('--wandb_group', default=None, type=str, help='WandB "Group". Name of the env by default.')
    arg('--wandb_job_type', default=None, type=str, help='WandB job type')
    arg('--wandb_tags', default=[], type=str, nargs='*', help='Tags can help finding experiments')
    arg('--wandb_key', default=None, type=str, help='API key for authorizing WandB')

    args = parser.parse_args()
    main(args)
