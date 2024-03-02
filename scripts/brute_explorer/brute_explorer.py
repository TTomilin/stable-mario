import gzip
import sys
import time

import numpy as np

import stable_retro

GAME_ENV = 'MarioParty-GbAdvance'
LOAD_STATE = 'Level1'
PLAYTIME = 0


def main():
    # create env:
    env = stable_retro.make(game=GAME_ENV,
                            state=LOAD_STATE,
                            use_restricted_actions=stable_retro.Actions.ALL,
                            # stable_retro.Actions.ALL needed to press start/select if needed
                            render_mode="none")  # rendering disabled for enhanced speed.

    env.reset()  # set environment to initial state
    t = time.time()  # initialize time
    start_time = t
    b = 10  # initialize bound

    # start taking random actions:
    while True:
        try:
            a = env.action_space.sample()
            a = filter_actions(a)

            elapsed = time.time() - t
            if elapsed >= b & b >= 9:
                print("Total playtime is " + str(np.round(time.time() - start_time))[:-2] + " seconds.")
                b = 1
                a[3] = 1
                t = time.time()
            elif elapsed >= b & b < 9:
                b = 9
                a[3] = 1
                t = time.time()

            _, _, _, _, _ = env.step(a)
        except KeyboardInterrupt:
            content = env.em.get_state()
            with gzip.open(SAVE_STATE, 'wb') as f:
                f.write(content)
                f.close()
            break

        # function filters the action vector used by the random agent.


# it ensures that start is pressed each 10 seconds and that start/select are not pressed at any other time
def filter_actions(a):
    a[2] = 0
    a[3] = 0
    return a


if __name__ == '__main__':
    LOAD_STATE = str(sys.argv[1])[:-6]
    SAVE_STATE = str(sys.argv[2])
    main()
