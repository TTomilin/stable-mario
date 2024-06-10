import gzip
import sys
import random

import numpy as np

import retro

GAME_ENV = 'MarioParty-GbAdvance'
LOAD_STATE = 'brute_explorer_15.state'
RENDER_MODE = 'none'
SAVE_STATE = 'example'
PLAYTIME = 0


def main():
    # create env:
    env = retro.make(game=GAME_ENV,
                            state=LOAD_STATE,
                            use_restricted_actions=retro.Actions.ALL,
                            # stable_retro.Actions.ALL needed to press start/select if needed
                            render_mode=RENDER_MODE)  # rendering disabled for enhanced speed.

    env.reset()  # set environment to initial state
    start_pressed = 0
    flip = False
    obs = {'in_menu': 1}

    # start taking random actions:
    while True:
        try:
            a = env.action_space.sample()
            a = filter_actions(a)

            if obs['in_menu'] == 1:
                for i in range(0, len(a)):
                    a[i] = 0
                if flip:
                    a[8] = 1 # if we are in main menu, press A immediately to re-enter shroom city
                flip = not flip

            if start_pressed == 1:
                print("pressing start again")
                a[3] = 1 # press start again if already pressed
                start_pressed = 0
            elif random.randint(0, 10**4) == 1:
                print("pressing start")
                a[3] = 1 # press start
                start_pressed = 1

            _, _, _, _, obs = env.step(a)
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
    a[3] = 0 #filter select
    return a


if __name__ == '__main__':
    LOAD_STATE = str(sys.argv[1])
    SAVE_STATE = str(sys.argv[2])
    RENDER_MODE = str(sys.argv[3])
    main()
