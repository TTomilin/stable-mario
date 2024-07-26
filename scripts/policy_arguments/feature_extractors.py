from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch.nn import BatchNorm2d
import torch

class BatchNorm(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        features_dim = observation_space.shape[0]
        super(BatchNorm, self).__init__(observation_space, features_dim)
        self.batch_norm = BatchNorm2d(num_features=features_dim)

    def forward(self, observations):
        return self.batch_norm(observations)