from copy import copy

import retro
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import ClipRewardEnv, MaxAndSkipEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from retro.examples.discretizer import Discretizer

GAME_ENV = 'broom_zoom-MP'
STATE = 'Level1'
POLICY = 'CnnPolicy'

TIMESTEPS = 50000


# descretizer class for broom zoom:
class BroomZoomDiscretizer(Discretizer):
    def __init__(self, env):
        super().__init__(env=env, combos=[['UP'], ['DOWN'], []])


def rec(iter):
    if iter % 1000 == 0:
        return True
    else:
        return False


def main():
    # create env:
    env = retro.make(game=GAME_ENV, state=STATE, render_mode="rgb_array")
    env = BroomZoomDiscretizer(env)
    # env = WarpFrame(env)
    env = ClipRewardEnv(env)
    env = MaxAndSkipEnv(env, skip=10)
    env = RecordVideo(env=env, video_folder="./saves/broom_zoom/videos", video_length=1000,
                      step_trigger=lambda x: x % 10000 == 0)

    # create callback to save best model found:
    eval_env = Monitor(copy(env))
    eval_callback = EvalCallback(eval_env, best_model_save_path="./saves/broom_zoom/checkpoint",
                                 log_path="./saves/broom_zoom/checkpoint", eval_freq=1000,
                                 deterministic=True, render=False)

    # create model:
    model = PPO(policy=POLICY, env=env, verbose=True)

    # train model:
    try:
        model.learn(total_timesteps=TIMESTEPS)
        model.save("saves/broom-zoom/broom_zoom")
    except KeyboardInterrupt:
        model.save("saves/broom_zoom/broom_zoom-bak")


if __name__ == '__main__':
    main()
