import retro;
from copy import copy;
from retrotools.discretizer import Discretizer;
from stable_baselines3 import DQN;
from stable_baselines3 import PPO;
from stable_baselines3 import A2C;
from stable_baselines3.common.monitor import Monitor;
from stable_baselines3.common.callbacks import EvalCallback;
from stable_baselines3.common.atari_wrappers import WarpFrame, ClipRewardEnv, StickyActionEnv, MaxAndSkipEnv;
from gymnasium.wrappers import RecordVideo, resize_observation;
from pyglet.gl import *

GAME_ENV = 'spook_spike-MP';
STATE = 'Level1';
POLICY = 'CnnPolicy';

TIMESTEPS = 1000000;

# descretizer class for spook spike:
class SpookSpikeDiscretizer(Discretizer):
    def __init__(self, env):
        super().__init__(env=env, combos=[['UP'], ['DOWN'], ['LEFT'], ['RIGHT'], []]);

def main():
    # create env:
    print("running")
    env = retro.make(game = GAME_ENV, state = STATE, render_mode="human");
    env = SpookSpikeDiscretizer(env);
    #env = resize_observation.ResizeObservation(env, (80, 72));
    env = ClipRewardEnv(env);
    #env = RecordVideo(env=env, video_folder="./saves/spook_spike/videos", video_length=0, step_trigger=lambda x: x % 10000 == 0);

    # create callback to save best model found:
    eval_env = Monitor(copy(env));
    eval_callback = EvalCallback(eval_env, best_model_save_path="./saves/spook_spike/checkpoint",
                             log_path="./saves/spook_spike/checkpoint", eval_freq=1000,
                             deterministic=True, render=False)

    # create model:
    model = PPO(policy = POLICY, env = env, verbose=True);

    # train model:
    try:
        model.learn(total_timesteps=TIMESTEPS);
        model.save("saves/spook_spike/spook_spike");
    except KeyboardInterrupt:
        model.save("saves/spook_spike/spook_spike-bak");

if __name__ == '__main__':
    main();
