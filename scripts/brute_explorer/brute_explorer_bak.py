import retro;
import sys;
import time;
import gzip;
import numpy as np;

GAME_ENV = 'MarioParty-GbAdvance';
LOAD_STATE = 'Level1';
PLAYTIME = 0;

def main():
    # create env:
    env = retro.make(game = GAME_ENV, 
                     state = LOAD_STATE, 
                     use_restricted_actions=retro.Actions.ALL, # retro.Actions.ALL needed to press start/select if needed  
                     render_mode="none"); # rendering disabled for enhanced speed. 

    env.reset(); # set environment to initial state
    t = time.time(); # initialize time
    b = 10; # initialize bound

    # start taking random actions:
    while True:
        try:
            a = env.action_space.sample();
            a, t, b = filter_actions(a, t, b); # filter button presses
            _, _, _, _, _ = env.step(a);
        except KeyboardInterrupt:
            content = env.em.get_state();
            with gzip.open(SAVE_STATE, 'wb') as f:
                f.write(content);
                f.close();
            break;    

# function filters the action vector used by the random agent.
# it ensures that start is pressed each 10 seconds and that start/select are not pressed at any other time
def filter_actions(a, t, b):
    a[2] = 0; 
    a[3] = 0;
    elapsed = time.time() - t; # compute elapsed time
    if elapsed >= b and b >= 9:
        b = 1
        a[3] = 1;
        t = time.time();
        print("still playing...");
    elif elapsed >= b and b < 9:
        b = 9;
        a[3] = 1
        t = time.time();
    return a, t, b;

if __name__ == '__main__':
    LOAD_STATE = str(sys.argv[1])[:-6];
    SAVE_STATE = str(sys.argv[2]);
    main();