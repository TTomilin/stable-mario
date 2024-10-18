"""Set of wrappers for normalizing observations, using saved/loaded obs_rms."""
import scipy

import gymnasium as gym
from gymnasium.wrappers.normalize import RunningMeanStd


class RestoreObsRms(gym.Wrapper, gym.utils.RecordConstructorArgs):
    """This wrapper will normalize observations s.t. each coordinate is centered with unit variance.

    Note:
        The normalization depends on past trajectories and observations will not be normalized correctly if the wrapper was
        newly instantiated or the policy was changed recently.
    """

    def __init__(self, env: gym.Env, load_dir: str):
        """This wrapper will normalize observations s.t. each coordinate is centered with unit variance.

        Args:
            env (Env): The environment to apply the wrapper
            epsilon: A stability parameter that is used when scaling the observations.
        """
        gym.utils.RecordConstructorArgs.__init__(self)
        gym.Wrapper.__init__(self, env)

        self.load_dir = load_dir

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        obs_rms_dir = f"{self.load_dir}/obs_rms"
        obs_rms_handle = self.env.get_wrapper_attr("obs_rms")
        obs_rms_handle.var = scipy.io.loadmat(f'{obs_rms_dir}/var.mat')['out']
        obs_rms_handle.mean = scipy.io.loadmat(f'{obs_rms_dir}/mean.mat')['out']
        obs_rms_handle.count = scipy.io.loadmat(f'{obs_rms_dir}/count.mat')['out'][0,0]

        return obs, info
