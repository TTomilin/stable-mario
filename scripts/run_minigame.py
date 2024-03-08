import argparse
from copy import copy
from pathlib import Path

from gymnasium.wrappers import ResizeObservation, NormalizeObservation, RecordVideo
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import ClipRewardEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

import stable_retro
from scripts.config import CONFIG
from stable_retro.examples.discretizer import Discretizer
from stable_retro.examples.ppo import StochasticFrameSkip
from wrappers.observation import Rescale


def main(cfg: argparse.Namespace):
    experiment_dir = Path(__file__).parent.resolve()

    # Create environment
    game = cfg.game
    state = cfg.load_state if cfg.load_state is not None else CONFIG[game]["state"]
    env = stable_retro.make(game=CONFIG[game]['game_env'], state=state, render_mode=cfg.render_mode)
    env = Discretizer(env, CONFIG[game]["actions"])
    env = ResizeObservation(env, CONFIG[game]["resize"])
    env = Rescale(env)
    env = NormalizeObservation(env)
    env = StochasticFrameSkip(env, n=4, stickprob=0.05)
    if CONFIG[game]["clip_reward"]:
        env = ClipRewardEnv(env)
    if cfg.record:
        video_folder = f"{experiment_dir}/saves/{game}/videos"
        env = RecordVideo(env=env, video_folder=video_folder, episode_trigger=lambda x: x % cfg.record_every == 0)

    # Create a callback to save best model
    eval_env = Monitor(copy(env))
    eval_callback = EvalCallback(eval_env, best_model_save_path=f"{experiment_dir}/saves/{game}/checkpoint",
                                 log_path=f"{experiment_dir}/saves/{game}/checkpoint", eval_freq=cfg.store_every,
                                 deterministic=True, render=False)

    # Create the model
    model = PPO(policy='CnnPolicy', env=env, verbose=True)

    # Train the model
    try:
        model.learn(total_timesteps=CONFIG[game]["timesteps"], callback= eval_callback if cfg.store_model else None)
        model.save(f"saves/{game}/{game}")
    except KeyboardInterrupt:
        model.save(f"saves/{game}/{game}-bak")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", type=str, default="broom_zoom", help="Name of the game")
    parser.add_argument("--render_mode", default="rgb_array", choices=["human", "rgb_array"], help="Render mode")
    parser.add_argument("--load_state", type=str, default=None, help="Path to the game save state to load")
    parser.add_argument("--record", default=False, action='store_true', help="Whether to record gameplay videos")
    parser.add_argument("--record_every", type=int, default=100, help="Record gameplay video every n episodes")
    parser.add_argument("--store_model", default=False, action='store_true', help="Whether to record gameplay videos")
    parser.add_argument("--store_every", type=int, default=100, help="Save model every n episodes")
    args = parser.parse_args()
    main(args)
