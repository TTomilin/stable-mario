from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import gymnasium as gym
import torch as th
import torch.nn.functional as F
from torch import nn

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy

class NetworkExample(nn.Module):
    '''
    Based on https://stable-baselines3.readthedocs.io/en/v1.0/guide/custom_policy.html
    '''

    def __init__(
            self,
            feature_dim: int,
            device: Union[th.device, str],
            last_layer_dim_pi: int = 64,
            last_layer_dim_vf: int = 64
    ):
        super(NetworkExample, self).__init__()

        # save output dimensions, used to create distributions:
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Create policy network:
        self.policy_net = nn.Sequential(
            nn.Linear(feature_dim, last_layer_dim_pi), nn.ReLU()
        ).to(device)

        # Create Value network:
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, last_layer_dim_vf), nn.ReLU()
        ).to(device)

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor), latent_policy, latent_value of network.
            If all layers are shared, then ''latent_policy == latent_value''
            
            Intuitively, 'latent_policy' is what to do, 'latent_value' is how good the
            current state is (to my understanding). 
        """
        return self.policy_net(features), self.value_net(features)
    
    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)
    
class RNN(nn.Module):
    '''
    Recurrent policy, based on https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html.
    Features two MLP layers, n units each for both policy and value networks (n configurable, with n=64 default).
    Uses Softmax activation function at output layer.
    '''

    def __init__(
            self,
            feature_dim: int,
            device: Union[th.device, str],
            hidden_size: int = 64
    ):
        super(RNN, self).__init__()

        # save output dimensions, used to create distributions:
        self.latent_dim_pi = hidden_size
        self.latent_dim_vf = hidden_size
        self.hidden_size = hidden_size

        self.policy_net = nn.Sequential(
            nn.Linear(feature_dim, hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.Softmax()
        ).to(device)

        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.Softmax()
        ).to(device)

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        return self.policy_net(features), self.value_net(features)
    
    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)

    def initHidden(self):
        return th.zeros(1, self.hidden_size)
    
class RecurrentPolicy(ActorCriticPolicy):
    '''
    Based on https://stable-baselines3.readthedocs.io/en/v1.0/guide/custom_policy.html
    '''

    def __init__(
            self,
            observation_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            lr_schedule: Callable[[float], float],
            net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
            activation_fn: Type[nn.Module] = nn.Tanh,
            *args,
            **kwargs
    ):
        super(RecurrentPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            *args,
            **kwargs
        )
        # Disable orthogonal initialization:
        self.ortho_init = False

    # override the creation of the policy and value networks:
    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = RNN(self.features_dim, self.device)