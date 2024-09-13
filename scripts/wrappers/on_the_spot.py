from __future__ import annotations
from typing import Any, SupportsFloat

import gymnasium
from gymnasium.core import WrapperObsType, WrapperActType

import gymnasium as gym
import math
from utilities.imaging import ImageUtilities

class OnTheSpotWrapper(gym.ObservationWrapper, gym.utils.RecordConstructorArgs):
    """Show image that AI is fed during training.

    This wrapper works by displaying the image seen by the AI under all previous wrappers applied.
    Note that order matters: the wrapper will only display the effects of other wrappers that have
    been applied before it.

    Example:
        >>> env = ShowObservation(env)
    """

    def __init__(self, env: gym.Env, n_skip_frames) -> None:
        """Shows a graphical representation of the AIs observations

        Args:
            env: The environment to apply the wrapper
        """
        gym.utils.RecordConstructorArgs.__init__(self)
        gym.ObservationWrapper.__init__(self, env)

        self.color = [26 * 8, 29 * 8, 16 * 8]
        self.step_cooldown = math.ceil(25 / n_skip_frames * 4)
        self.temp_counter = 0

    def observation(self, observation):
        
        if ImageUtilities.find_color(self.color, observation) != None:
            print(self.temp_counter)
            self.temp_counter += 1

        return observation