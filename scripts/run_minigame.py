import argparse
from copy import copy

from gymnasium.wrappers import ResizeObservation, NormalizeObservation
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import ClipRewardEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

import stable_retro
from scripts.config import CONFIG
from stable_retro.examples.discretizer import Discretizer
from stable_retro.examples.ppo import StochasticFrameSkip


class MiniGame(Discretizer):
    def __init__(self, env, actions):
        super().__init__(env=env, combos=actions)


def main(cfg: argparse.Namespace):
    # Create environment
    game = cfg.game
    state = cfg.state if cfg.state is not None else CONFIG[game]["state"]
    env = stable_retro.make(game=CONFIG[game]['game_env'], state=state, render_mode=cfg.render_mode)
    env = MiniGame(env, CONFIG[game]["actions"])
    env = ResizeObservation(env, CONFIG[game]["resize"])
    env = StochasticFrameSkip(env, n=4, stickprob=0.05)
    env = NormalizeObservation(env)
    if "clip_reward" in CONFIG[game] and CONFIG[game]["clip_reward"]:
        env = ClipRewardEnv(env)
    # env = RecordVideo(env=env, video_folder=f"./saves/{game}/videos", video_length=0, step_trigger=lambda x: x % 10000 == 0)

    # create callback to save best model found:
    eval_env = Monitor(copy(env))
    eval_callback = EvalCallback(eval_env, best_model_save_path=f"./saves/{game}/checkpoint",
                                 log_path=f"./saves/{game}/checkpoint", eval_freq=1000,
                                 deterministic=True, render=False)

    # create model:
    model = PPO(policy='CnnPolicy', env=env, verbose=True)

    # train model:
    try:
        model.learn(total_timesteps=CONFIG[game]["timesteps"])
        model.save(f"saves/{game}/{game}")
    except KeyboardInterrupt:
        model.save(f"saves/{game}/{game}-bak")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default="broom_zoom")
    parser.add_argument("--render_mode", default="rgb_array", choices=["human", "rgb_array"])
    parser.add_argument("--state", default=None)
    args = parser.parse_args()
    main(args)
