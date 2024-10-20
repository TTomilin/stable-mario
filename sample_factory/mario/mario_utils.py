from typing import Optional

import gymnasium as gym
from gymnasium.wrappers import ResizeObservation, FrameStack

import stable_retro
from sample_factory.envs.env_wrappers import (
    ClipRewardEnv,
    MaxAndSkipEnv,
    NoopResetEnv, PixelFormatChwWrapper,
)
from scripts.config import CONFIG
from stable_retro.examples.discretizer import Discretizer


class MarioSpec:
    def __init__(self, name, env_id, default_timeout=None):
        self.name = name
        self.env_id = env_id
        self.default_timeout = default_timeout


MARIO_ENVS = [
    MarioSpec("broom_zoom", "BroomZoom-v0"),
]


def mario_env_by_name(name):
    for cfg in MARIO_ENVS:
        if cfg.name == name:
            return cfg
    raise Exception("Unknown Mario env")


def make_mario_env(env_name, cfg, env_config, render_mode: Optional[str] = None):
    mario_spec = mario_env_by_name(env_name)

    game = CONFIG[cfg.game]
    state = cfg.load_state if cfg.load_state is not None else game["state"]
    env = stable_retro.make(game=game['game_env'], state=state, render_mode=render_mode)

    if cfg.discretize:
        env = Discretizer(env, game["actions"])

    if game["clip_reward"]:
        env = ClipRewardEnv(env)

    if mario_spec.default_timeout is not None:
        env._max_episode_steps = mario_spec.default_timeout

    # these are chosen to match Stable-Baselines3 and CleanRL implementations as precisely as possible
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=cfg.env_frameskip)
    env = ResizeObservation(env, game["resize"])
    env = PixelFormatChwWrapper(env)
    # env = FrameStack(env, cfg.env_framestack)  # TODO out of order
    return env
