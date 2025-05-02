import gzip
import sys
import time
import MelonFolly
import numpy as np

import stable_retro

GAME_ENV = 'melon_folly-MP'
LOAD_STATE = 'before0.state'
PLAYTIME = 0


def main():
    # create env:
    env = stable_retro.make(game=GAME_ENV,
                            state=LOAD_STATE,
                            use_restricted_actions=stable_retro.Actions.ALL,
                            # stable_retro.Actions.ALL needed to press start/select if needed
                            render_mode=RENDER_MODE)  # rendering disabled for enhanced speed.

    env.reset()  # set environment to initial state
    t = time.time()  # initialize time
    start_time = t
    b = 10  # initialize bound
    # printed = False
    # start taking random actions:
    while True:
        try:
            a = env.action_space.sample()
            a = filter_actions(a)
            # print(a)
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
            ob, _, _, _, vals = env.step(a)
            if vals['Inlevel'] == 1:
                route = MelonFolly.main(ob) #Get the route
                inroute = 0
                score = vals['score']
                c = [0]*12
                c[0] = 1
                _, _, _, _, vals = env.step(c)
                while vals['Inlevel'] == 1:
                    
                    c = [0]*12
                    c[0] = 1
                    if vals['score']!=score:
                        score = vals['score']
                        try:
                            direction = int(route[inroute])- int(route[inroute+1])
                        except:
                            continue
                        if direction == 10:
                            c[4] = 1
                        elif direction == -10:
                            c[5] = 1
                        elif direction == 1:
                            c[6] = 1
                        elif direction == -1:
                            c[7] = 1
                        inroute += 1
                        _, _, _, _, vals = env.step(c)
                    _, _, _, _, vals = env.step(c)
                
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
    # LOAD_STATE = str(sys.argv[1])[:-6]
    # SAVE_STATE = str(sys.argv[2])
    # RENDER_MODE = str(sys.argv[3])
    SAVE_STATE = 'savemeehere'
    RENDER_MODE = 'human'
    main()
