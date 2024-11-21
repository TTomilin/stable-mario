from __future__ import annotations

from typing import Any, SupportsFloat

import gymnasium
import gymnasium as gym
import numpy as np
import pylab as pl
import torch
from gymnasium.core import WrapperObsType, WrapperActType
from torchvision.transforms.functional import center_crop


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
    """Show image that AI is fed during training.

    This wrapper works by displaying the image seen by the AI under all previous wrappers applied.
    Note that order matters: the wrapper will only display the effects of other wrappers that have
    been applied before it.

    Example:
        >>> env = ShowObservation(env)
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
        self.animate_observations(observations=observation)
        return observation

    def animate_observations(self, observations):
        display = (observations - np.ones(shape=observations.shape) * np.min(observations)) * (
                    1 / (np.max(observations) - np.min(observations)))
        pl.clf()
        pl.imshow(X=display, interpolation="nearest")
        pl.pause(10 ** -6)
        pl.draw()


class CenterCrop(gym.ObservationWrapper, gym.utils.RecordConstructorArgs):
    """Crop image that AI is fed during training.

    This wrapper works by taking a tuple (W1, W2, ...) and showing the AI only
    the pixel (W1, W2, ...) around the center of its input image, cutting
    away all other pixels. E.g., if you input (48, 48), then only the 48x48 square
    around the input's center will be fed to the AI. 

    Example:
        >>> env = CenterCrop((48,48))
    """

    def __init__(self, env: gym.Env, dim: tuple[int, int] | int) -> None:
        """Crops the AIs observations

        Args:
            env: The environment to apply the wrapper
            dim: the dimensions of the image left after cropping
        """
        gym.utils.RecordConstructorArgs.__init__(self)
        gym.ObservationWrapper.__init__(self, env)
        self.__dim = dim

    def observation(self, observation):
        """Crops current observation to specified dimensions

        Args:
            None

        Returns:
            Cropped observations
        """
        cropped_tensor = center_crop(torch.from_numpy(observation.transpose()),
                                     output_size=self.__dim)  # get cropped tensor
        cropped_array = cropped_tensor.cpu().detach().numpy()  # convert tensor to numpy array
        cropped_array = cropped_array.transpose()  # transpose tensor: dimensions of tensor used by torch and gym are reversed

        return cropped_array


class FilterColors(gym.ObservationWrapper, gym.utils.RecordConstructorArgs):
    """
    """

    def __init__(self, env: gym.Env, colors: list[str]) -> None:
        """
        Removes all colors but the ones specified in a comma seperated list of extended hex.
        
        Arguments:
        env: the env
        colors: list of e-hex colors
        """
        gym.utils.RecordConstructorArgs.__init__(self)
        gym.ObservationWrapper.__init__(self, env)
        colorlist = []
        for color in colors:
            c = []
            for a in color:
                if a.isnumeric():
                    c.append(int(a) * 8)
                else:
                    c.append((int(ord(a)) - 55) * 8)
            colorlist.append(c)
        print(f"Showing {colorlist}")
        self.__colors = colorlist

    def observation(self, observation):
        for i in range(len(observation)):
            for j in range(len(observation[0])):
                for color in self.__colors:
                    if observation[i, j, 0] == color[0] and observation[i, j, 1] == color[1] and observation[i, j, 2] == \
                            color[2]:
                        break
                else:
                    observation[i, j] = [0, 0, 0]
        return observation


class Grabbit(gymnasium.Wrapper):
    """
    """

    def __init__(self, env: gym.Env, colors: list[str]) -> None:
        """
        Removes all colors but the ones specified in a comma seperated list of extended hex.
        Wrapper designed for Grabbit
        Arguments:
        env: the env
        colors: list of e-hex colors
        """
        gymnasium.Wrapper.__init__(self, env)
        colorlist = []
        for color in colors:
            c = []
            for a in color:
                if a.isnumeric():
                    c.append(int(a) * 8)
                else:
                    c.append((int(ord(a)) - 55) * 8)
            colorlist.append(c)
        print(f"Showing {colorlist}")
        self.__colors = colorlist

    def step(self, action: WrapperActType) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        state, reward, done, truncated, info = self.env.step(action)
        for i in range(len(state)):
            for j in range(len(state[0])):
                for color in self.__colors:
                    if state[i, j, 0] == color[0] and state[i, j, 1] == color[1] and state[i, j, 2] == color[2]:
                        if i >= 30 or j >= 50:
                            if set(color) == {248}:
                                if abs(i - 90) + abs(j - 120) > 15:
                                    reward += ((i - 90) ** 2 + (j - 120) ** 2) ** -2
                            break
                else:
                    state[i, j] = [0, 0, 0]
        return state, reward, done, truncated, info

class SledSlide(gymnasium.Wrapper):
    """
    """

    def __init__(self, env: gym.Env) -> None:
        """
        
        """
        gymnasium.Wrapper.__init__(self, env)

    def step(self, action: WrapperActType) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        state, reward, done, truncated, info = self.env.step(action)
        if info['progress'] != 14:
            state[0] = [info['speed']//5, 0, 0]
            state[1] = [info['speed']//5, 0, 0]
        return state, reward, done, truncated, info