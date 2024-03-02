import numpy as np
import stable_retro
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import WarpFrame, ClipRewardEnv
from stable_baselines3.common.noise import NormalActionNoise

GAME_ENV = 'boo_bye-MP'
STATE = 'Level1'
POLICY = 'CnnPolicy'

TIMESTEPS = 100000


def main():
    # create env:
    env = stable_retro.make(game=GAME_ENV, state=STATE)  # for DQN: use_restricted_actions=stable_retro.Actions.DISCRETE
    env = WarpFrame(env)
    env = ClipRewardEnv(env)

    # create model:
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    model = PPO(policy=POLICY, env=env, verbose=True)

    # train model:
    try:
        model.learn(total_timesteps=TIMESTEPS)
        model.save("boo-bye")
    except KeyboardInterrupt:
        model.save("boo-bye-bak")


if __name__ == '__main__':
    main()
