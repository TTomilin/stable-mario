from __future__ import annotations
from typing import Any, SupportsFloat

import gymnasium
from gymnasium.core import WrapperObsType, WrapperActType

import gymnasium as gym
import pylab as pl
from torchvision.transforms.functional import center_crop
import torch

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
        pl.imshow(observations.astype('uint8'))
        pl.pause(10**-6)
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
        cropped_tensor = center_crop(torch.from_numpy(observation.transpose()), output_size=self.__dim) # get cropped tensor
        cropped_array = cropped_tensor.cpu().detach().numpy() # convert tensor to numpy array
        cropped_array = cropped_array.transpose() # transpose tensor: dimensions of tensor used by torch and gym are reversed

        return cropped_array
