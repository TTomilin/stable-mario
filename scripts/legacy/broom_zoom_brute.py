import pickle

import retro

from retro.examples.brute import Brute

GAME_ENV = 'broom_zoom-MP'
STATE = 'Level1'
STEP_LIMIT = 2000000
EPISODE_LIMIT = 2000000


def main():
    env = retro.make(game=GAME_ENV, state=STATE, use_restricted_actions=retro.Actions.DISCRETE)

    brute = Brute(env, max_episode_steps=EPISODE_LIMIT)
    timesteps = 0
    best_rew = float('-inf')
    try:
        while True:
            acts, rew = brute.run()
            timesteps += len(acts)

            if rew > best_rew:
                print("new best reward {} => {}".format(best_rew, rew))
                best_rew = rew

                print("saving best list of actions found...")
                out_file = open("saves/broom_zoom/brute.pkl", "wb")
                pickle.dump(acts, out_file)

                env.unwrapped.record_movie("best.bk2")
                env.reset()
                for act in acts:
                    env.step(act)
                env.unwrapped.stop_record()

            if timesteps > STEP_LIMIT:
                print("timestep limit exceeded")
                break
    except KeyboardInterrupt:
        print("aborting...")


if __name__ == '__main__':
    main()
