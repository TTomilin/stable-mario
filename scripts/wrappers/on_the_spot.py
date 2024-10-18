from __future__ import annotations

from collections import deque

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box

from utilities.imaging import ImageUtilities

class HackOnTheSpotWrapper(gym.ObservationWrapper, gym.utils.RecordConstructorArgs):
    def __init__(
        self,
        env: gym.Env,
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
        self.arrow_color = [26 * 8, 29 * 8, 16 * 8]
        self.glove_color = [16*8, 18*8, 22*8] #[19 * 8, 21 * 8, 25 * 8]
        self.counter = 0
        self.step_cooldown = cooldown
        self.ret_counter = 0

        self.act_cooldown = 20
        self.act_counter = 0
        self.relevant_frame = None

        low = np.tile(self.observation_space.low, (6,1,1))
        high = np.tile(self.observation_space.high, (6,1,1))
        self.observation_space = Box(
            low=low, high=high, dtype=self.observation_space.dtype
        )

    def observation(self, observation):
        # verify memory depth and n.o. elts in queue:
        assert len(self.frames) == self.stack_depth, (len(self.frames), self.stack_depth)
        
        # figure out the number of gloves on screen:
        glove_pixels = ImageUtilities.find_color(self.glove_color, observation, observation.shape[0] * observation.shape[1])

        if glove_pixels != None:
            num_gloves = len(glove_pixels) // 35
            if num_gloves > 0:
                print(num_gloves)
                self.relevant_frame = self.frames[self.stack_depth - num_gloves]
                imageio.imsave(uri=f"/home/ctrl/AP_self/temp/rel_frame.png", im=self.relevant_frame)


        # add the memories to ret_frames:
        self.ret_frames.extendleft(self.frames)
        # append current observations to ret_frames:
        self.ret_frames.append(observation)

        #for i in range(len(self.frames)):
        #    imageio.imsave(uri=f"/home/ctrl/AP_self/temp/test_{i}.png", im=self.frames[i])

        # convert the queue ret_frames to a single matrix and return:
        return np.concatenate(self.ret_frames, axis=0)
        # there is some (avoidable) overhead here, but there is a similar amount of overhead in VecFrameStack.

    def step(self, action):
        action = np.zeros(12)

        action_idx = self.get_action(self.relevant_frame)

        if self.act_counter == 0:
            action[action_idx] = 1 # 7
        self.act_counter += 1
        if self.act_counter > self.act_cooldown:
            self.act_counter = 0

        observation, reward, terminated, truncated, info = self.env.step(action)
        # note: we receive 'observation' as a single (colored) image

        # detect presence of color:
        color_position = ImageUtilities.find_color(self.arrow_color, observation, 1)

        # store color if cooldown elapsed:
        if color_position != None and self.counter > self.step_cooldown:
            print(f"color pos: {color_position}")
            self.frames.append(observation)
            self.counter = 0
            #print(f"Frame stored: {self.ret_counter}")
            #self.ret_counter = self.ret_counter + 1
        elif color_position:
            pass
        self.counter += 1

        return self.observation(observation), reward, terminated, truncated, info
    
    def get_action(self, frame: np.array) -> int:
        color_position = ImageUtilities.find_color(self.arrow_color, frame, 1)

        if color_position == None:
            return 1
        elif color_position[0] == [63,171]:
            return 7
        elif color_position[0] == [63,107]:
            return 6
        elif color_position[0] == [95,139]:
            return 5
        elif color_position[0] == [31,139]:
            return 4

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        # at first, we duplicate the first observation stack_depth times into memory to avoid mismatch in observation size
        [self.frames.append(obs) for _ in range(self.stack_depth)] 
        # also, we set rel_frame to first obs:
        self.relevant_frame = obs

        return self.observation(obs), info

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
        self.ret_frames = deque(maxlen=self.stack_depth + 1)
        self.color = color
        self.counter = 0
        self.step_cooldown = cooldown
        self.ret_counter = 0

        low = np.tile(self.observation_space.low, (6, 1, 1))
        high = np.tile(self.observation_space.high, (6, 1, 1))
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

        # for i in range(len(self.frames)):
        #    imageio.imsave(uri=f"/home/ctrl/AP_self/temp/test_{i}.png", im=self.frames[i])

        # convert the queue ret_frames to a single matrix and return:
        return np.concatenate(self.ret_frames, axis=0)
        # there is some (avoidable) overhead here, but there is a similar amount of overhead in VecFrameStack.

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        # note: we receive 'observation' as a single (colored) image

        # detect presence of color:
        color_position = (ImageUtilities.find_color(self.color, observation) != None)

        # store color if cooldown elapsed:
        if color_position != None and self.counter > self.step_cooldown:
            self.frames.append(observation)
            self.counter = 0
            #print(f"Frame stored: {self.ret_counter}")
            #self.ret_counter = self.ret_counter + 1
        elif color_position:
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

    def __init__(self, env: gym.Env, n_skip_frames, color: np.ndarray, observation_shape: tuple, memory_depth: int,
                 cooldown: int) -> None:
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
