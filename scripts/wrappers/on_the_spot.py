from __future__ import annotations


import gymnasium as gym
from gymnasium.wrappers.frame_stack import LazyFrames
from gymnasium.spaces import Box
from collections import deque
import numpy as np
import imageio
from utilities.imaging import ImageUtilities

class FindAndStoreColorWrapper(gym.ObservationWrapper, gym.utils.RecordConstructorArgs):
    def __init__(
        self,
        env: gym.Env,
        color: np.ndarray, 
        memory_depth: int, 
        cooldown: int
    ):
        """Observation wrapper that stacks the observations in a rolling manner.

        Args:
            env (Env): The environment to apply the wrapper
            num_stack (int): The number of frames to stack
            lz4_compress (bool): Use lz4 to compress the frames internally
        """
        gym.utils.RecordConstructorArgs.__init__(
            self, num_stack=memory_depth + 1, lz4_compress=False
        )
        gym.ObservationWrapper.__init__(self, env)

        self.stack_depth = memory_depth
        self.frames = deque(maxlen=self.stack_depth)
        self.ret_frames = deque(maxlen =self.stack_depth + 1)
        self.color = color
        self.counter = 0
        self.step_cooldown = cooldown

        low = np.tile(self.observation_space.low, (6,1,1))
        high = np.tile(self.observation_space.high, (6,1,1))
        self.observation_space = Box(
            low=low, high=high, dtype=self.observation_space.dtype
        )

    def observation(self, observation):
        # verify memory depth and n.o. elts in queue:
        assert len(self.frames) == self.stack_depth, (len(self.frames), self.stack_depth)
        
        # add the memories to ret_frames:
        self.ret_frames.extendleft(self.frames)
        # append current observations to ret_frames:
        self.ret_frames.append(observation)
        if len(self.ret_frames) >= 5:
            for frame in self.ret_frames:
                imageio.imsave(uri='/home/ctrl/AP_self/temp/test.png', im=frame, format="png")

        # convert the queue ret_frames to a single matrix and return:
        return np.concatenate(self.ret_frames, axis=0)
        # there is some (avoidable) overhead here, but there is a similar amount of overhead in VecFrameStack.

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        # note: we receive 'observation' as a single (colored) image

        # detect presence of color:
        color_found = (ImageUtilities.find_color(self.color, observation) != None)

        # store color if cooldown elapsed:
        if color_found and self.counter > self.step_cooldown:
            self.frames.append(observation)
            self.counter = 0
            print("Frame stored.")
        elif color_found:
            pass
        self.counter += 1

        return self.observation(observation), reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        # at first, we duplicate the first observation stack_depth times into memory to avoid mismatch in observation size
        [self.frames.append(obs) for _ in range(self.stack_depth)] 

        return self.observation(obs), info

class OnTheSpotWrapper(gym.ObservationWrapper, gym.utils.RecordConstructorArgs):
    """Show image that AI is fed during training.

    This wrapper works by displaying the image seen by the AI under all previous wrappers applied.
    Note that order matters: the wrapper will only display the effects of other wrappers that have
    been applied before it.

    Example:
        >>> env = ShowObservation(env)
    """

    def __init__(self, env: gym.Env, n_skip_frames, color: np.ndarray, observation_shape: tuple, memory_depth: int, cooldown: int) -> None:
        """Shows a graphical representation of the AIs observations

        Args:
            env: The environment to apply the wrapper
        """
        gym.utils.RecordConstructorArgs.__init__(self, num_stack=memory_depth + 1)
        gym.ObservationWrapper.__init__(self, env)

        self.color = color
        self.step_cooldown = cooldown
        self.observation_dimension = observation_shape
        print(observation_shape)
        self.memory_depth = memory_depth
        self.memory = np.zeros((self.memory_depth,) + self.observation_dimension)
        self.counter = 0

    def observation(self, observation):
        # detect presence of color:
        color_found = ImageUtilities.find_color(self.color, observation)

        # store color if cooldown elapsed:
        if color_found and self.counter > self.step_cooldown:
            self.memory = np.roll(a=self.memory, shift=1, axis=0)
            self.memory[0] = observation
            self.counter = 0
            print("Frame added.")
        elif color_found:
            pass

        # let returned observation be the memory, with the current frame added to it
        return_observation = np.concatenate((np.expand_dims(observation, axis=0), self.memory), axis=0)

        return return_observation