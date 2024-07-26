import argparse
import pathlib

import numpy as np
import json

from gymnasium.wrappers import ResizeObservation, NormalizeObservation, RecordVideo, FrameStack, NormalizeReward, TimeLimit
from stable_baselines3.common.atari_wrappers import ClipRewardEnv, MaxAndSkipEnv

import stable_retro
import stable_retro.data
from stable_retro.examples.discretizer import Discretizer
from wrappers.observation import Rescale, ShowObservation, CenterCrop
from wrappers.logger import LogVariance, LogRewardSummary, StepRewardLogger

STEPS_PER_FRAME = 4
FRAMERATE = 60

class RetroEnvCreator:
    @staticmethod
    def create(cfg: argparse.Namespace, log_dir: str, config: dict):
        game = cfg.game
        state = cfg.load_state if cfg.load_state is not None else config[game]["state"]
        env = stable_retro.make(game=config[game]['game_env'], state=state, render_mode=cfg.render_mode)

        if cfg.discretize:
            env = Discretizer(env, config[game]["actions"])
        if cfg.crop:
            dim = np.array([int(num_str) for num_str in cfg.crop_dimension.split("x")])
            env = CenterCrop(env, dim=dim)
        if cfg.resize_observation:
            env = ResizeObservation(env, config[game]["resize"])
        if cfg.rescale:
            env = Rescale(env)
        if cfg.normalize_observation:
            env = NormalizeObservation(env)
        if cfg.normalize_reward:
            env = NormalizeReward(env)    
        if cfg.show_observation:
            env = ShowObservation(env)
        if cfg.skip_frames:
            env = MaxAndSkipEnv(env, skip=cfg.n_skip_frames * STEPS_PER_FRAME)
        if cfg.stack_frames:
            env = FrameStack(env, cfg.n_stack_frames)
        if config[game]["clip_reward"]:
            env = ClipRewardEnv(env)
        if cfg.record:
            video_folder = f"{log_dir}/videos"
            env = RecordVideo(env=env, video_folder=video_folder, episode_trigger=lambda x: x % cfg.record_every == 0)
        if cfg.time_limit != None:
            env = TimeLimit(env=env, max_episode_steps=cfg.time_limit * STEPS_PER_FRAME * FRAMERATE)
        if cfg.log_variance and cfg.with_wandb:
            env = LogVariance(env, cfg.variance_log_frequency)
        if cfg.log_reward_summary and cfg.with_wandb:
            env = LogRewardSummary(env, cfg.log_reward_summary_frequency)
        if cfg.log_step_rewards:
            env = StepRewardLogger(env, log_dir)

        return env

