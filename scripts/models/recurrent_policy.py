from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import gymnasium as gym
import torch as th
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable

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
    Features three MLP layers, n units each for both policy and value networks (n configurable, with n=64 default).
    Uses Softmax activation function at output layer.
    '''

    def __init__(
            self,
            feature_dim: int,
            device: Union[th.device, str],
            hidden_size: int = 64,
            n_rollout_steps: int = 64
    ):
        super(RNN, self).__init__() # run constructor of parent

        # note: memory_size = n_epochs * buffer_size... -> n_rollout_steps wordt memory_size

        # save output dimensions, used to create distributions:
        self.latent_dim_pi = hidden_size
        self.latent_dim_vf = hidden_size
        self.hidden_size = hidden_size

        # set actor and critic's hidden values to 0 initially...
        self.actor_hidden = th.zeros(1, hidden_size) 
        self.critic_hidden = th.zeros(1, hidden_size)
        
        # enable collection of hidden states...
        self.n_rollout_steps = n_rollout_steps
        self.past_actor_hidden = th.zeros(n_rollout_steps, hidden_size) 
        self.past_critic_hidden = th.zeros(n_rollout_steps, hidden_size)
        self.elapsed_rollout_steps_actor = 0
        self.elapsed_rollout_steps_critic = 0

        if th.cuda.is_available(): # if cuda device is available...
            self.actor_hidden = self.actor_hidden.cuda() # ensure they are on cuda device by invoking 'cuda'
            self.critic_hidden = self.critic_hidden.cuda()
            self.past_actor_hidden = self.past_actor_hidden.cuda()
            self.past_critic_hidden = self.past_critic_hidden.cuda()

        self.policy_net = nn.Sequential( # create policy network...
            nn.Linear(feature_dim, hidden_size), # add linear layer
            nn.Linear(hidden_size, hidden_size), # add linear layer
            nn.Linear(hidden_size, hidden_size), # add linear layer
            nn.LogSoftmax(dim=1) # add softmax activation before returning outputs
        ).to(device)

        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, hidden_size), # add linear layer
            nn.Linear(hidden_size, hidden_size), # add linear layer
            nn.Linear(hidden_size, hidden_size), # add linear layer
            nn.LogSoftmax(dim=1) # add softmax activation before returning outputs
        ).to(device)

        self.count = 0

    """
    Note to self:
    in principle, the hidden state for each observation should be collected during rollout collection (see 'on_policy_algorithm').
    This will likely require an additional property added to 'RolloutBuffer'.
    Then, they should be passed in together with the observations to 'evaluate_actions', so that each observation can be evaluated
    together with its actual hidden state instead of just the current hidden state.

    Right now, you evaluate each action with the current hidden state instead of the hidden state that was present at the time
    the action was taken. This seems wrong on an intuitive level. Hence, implementing the recurrent policy will require a new
    version of PPO and a new rollout buffer type added to PPO. We can add the rollout buffer type in the same .py file...
    """

    def forward_actor(self, features) -> th.Tensor: # note: shape of features changes from 1x17280 to 64 x 17280
        if features.shape[0] > 1: # if multiple observations are passed simultaneously (as in evaluate action)
            print(self.count)
            self.count += 1
            self.actor_hidden = self.past_actor_hidden

        self.actor_hidden = F.tanh(self.policy_net[0](features) + self.policy_net[1](self.actor_hidden)) # sum linear outputs, apply element-wise tanh
        output = self.policy_net[2](self.actor_hidden) # apply linear layer
        output = self.policy_net[3](output) # apply activation function

        self.actor_hidden = self.actor_hidden.detach()
        if features.shape[0] > 1: # if multiple observations were passed...
            self.actor_hidden = self.actor_hidden[self.actor_hidden.shape[0] - 1] # remove recalled hidden states...
            self.past_actor_hidden = th.zeros(self.n_rollout_steps, self.hidden_size).cuda() # flush memory
            self.elapsed_rollout_steps_actor = 0 # reset elapsed
        else:
            self.past_actor_hidden[self.elapsed_rollout_steps_actor % self.n_rollout_steps] = self.actor_hidden # store this hidden state
            self.elapsed_rollout_steps_actor += 1 # increment elapsed hidden states

        return output
    
    def forward_critic(self, features) -> th.Tensor:
        if features.shape[0] > 1:
            self.critic_hidden = self.critic_hidden.repeat(features.shape[0], 1)

        self.critic_hidden = F.tanh(self.value_net[0](features) + self.value_net[1](self.critic_hidden))
        output = self.value_net[2](self.critic_hidden)
        output = self.value_net[3](output)
        
        self.critic_hidden = self.critic_hidden.detach()
        if features.shape[0] > 1:
            self.critic_hidden = self.critic_hidden[0]

        return output

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        return self.forward_actor(features), self.forward_critic(features)
    
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