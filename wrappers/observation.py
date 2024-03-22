from __future__ import annotations
from typing import Any, SupportsFloat

import gymnasium
from gymnasium.core import WrapperObsType, WrapperActType

import numpy as np
import gymnasium as gym
import pylab as pl;


class Rescale(gymnasium.Wrapper):
    """Rescale the observation space to [-1, 1]."""

    def __init__(self, env):
        gymnasium.Wrapper.__init__(self, env)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) \
            -> tuple[WrapperObsType, dict[str, Any]]:
        state, info = self.env.reset()
        return state / 255. * 2 - 1, info

    def step(self, action: WrapperActType) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        state, reward, done, truncated, info = self.env.step(action)
        return state / 255. * 2 - 1, reward, done, truncated, info
    
class ShowObservation(gym.ObservationWrapper, gym.utils.RecordConstructorArgs):
    """Resize the image observation.

    This wrapper works on environments with image observations. More generally,
    the input can either be two-dimensional (AxB, e.g. grayscale images) or
    three-dimensional (AxBxC, e.g. color images). This resizes the observation
    to the shape given by the 2-tuple :attr:`shape`.
    The argument :attr:`shape` may also be an integer, in which case, the
    observation is scaled to a square of side-length :attr:`shape`.

    Example:
        >>> import gymnasium as gym
        >>> from gymnasium.wrappers import ResizeObservation
        >>> env = gym.make("CarRacing-v2")
        >>> env.observation_space.shape
        (96, 96, 3)
        >>> env = ResizeObservation(env, 64)
        >>> env.observation_space.shape
        (64, 64, 3)
    """

    def __init__(self, env: gym.Env) -> None:
        """Shows a graphical representation of the AIs observations

        Args:
            env: The environment to apply the wrapper
        """
        gym.utils.RecordConstructorArgs.__init__(self)
        gym.ObservationWrapper.__init__(self, env)

    def observation(self, observation):
        """Shows a graphical representation of the AI's inputs under the applied wrappers

        Args:
            None

        Returns:
            Unchanged observations
        """
        try:
            self.animate_observations(observations=observation)
        except KeyboardInterrupt:
            raise KeyboardInterrupt;
        return observation

    def animate_observations(self, observations):
        pl.imshow(observations)
        try:
            pl.pause(10**-6)
        except KeyboardInterrupt:
            return;
        pl.draw()
