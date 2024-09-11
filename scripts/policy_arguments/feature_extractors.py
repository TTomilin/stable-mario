from typing import Dict, List, Tuple, Type, Union

import gymnasium as gym
import torch as th
from gymnasium import spaces
from torch import nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.preprocessing import is_image_space

class BatchNorm(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        features_dim = observation_space.shape[0]
        super(BatchNorm, self).__init__(observation_space, features_dim)
        self.batch_norm = nn.BatchNorm2d(num_features=features_dim)

    def forward(self, observations):
        return self.batch_norm(observations)
    
class NatureRNN(BaseFeaturesExtractor):
    """
    CNN from DQN Nature paper:
        Mnih, Volodymyr, et al.
        "Human-level control through deep reinforcement learning."
        Nature 518.7540 (2015): 529-533.

    :param observation_space:
    :param features_dim: Number of features extracted.
        This corresponds to the number of unit for the last layer.
    :param normalized_image: Whether to assume that the image is already normalized
        or not (this disables dtype and bounds checks): when True, it only checks that
        the space is a Box and has 3 dimensions.
        Otherwise, it checks that it has expected dtype (uint8) and bounds (values in [0, 255]).
    """

    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 512,
        lstm_size: int = 512,
        normalized_image: bool = False,
        hidden_size: int = 64,
        batch_size: int = 64,
        n_epochs: int = 10,
        buffer_size: int = 2048
    ) -> None:
        assert isinstance(observation_space, spaces.Box), (
            "NatureCNN must be used with a gym.spaces.Box ",
            f"observation space, not {observation_space}",
        )
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        assert is_image_space(observation_space, check_channels=False, normalized_image=normalized_image), (
            "You should use NatureCNN "
            f"only with images not with {observation_space}\n"
            "(you are probably using `CnnPolicy` instead of `MlpPolicy` or `MultiInputPolicy`)\n"
            "If you are using a custom environment,\n"
            "please check it using our env checker:\n"
            "https://stable-baselines3.readthedocs.io/en/master/common/env_checker.html.\n"
            "If you are using `VecNormalize` or already normalized channel-first images "
            "you should pass `normalize_images=False`: \n"
            "https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html"
        )
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # required for bookkeeping:
        self.buffer_size = buffer_size
        self.n_epochs = n_epochs
        self.n_rollout_steps = n_epochs * buffer_size

        # save output dimensions, used to create distributions:
        self.latent_dim_pi = hidden_size
        self.latent_dim_vf = hidden_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.lstm_size = lstm_size

        # set actor and critic's hidden values to 0 initially...
        self.actor_hidden = [th.zeros(1, lstm_size), th.zeros(1, lstm_size)]
        
        # enable collection of hidden states...
        self.past_actor_hidden = th.zeros(n_epochs, (buffer_size // batch_size), batch_size, lstm_size) #th.zeros(n_epochs, buffer_size // batch_size, batch_size, hidden_size) 
        self.past_critic_hidden = th.zeros(n_epochs, (buffer_size // batch_size), batch_size, lstm_size) #th.zeros(n_epochs, buffer_size // batch_size, batch_size, hidden_size)
        # first axis: epochs, i.e. n.o. buffers collected;
        # second axis: batch in current buffer;
        # third axis: number of hidden states in current batch;
        # fourth axis: the numbers in the current hidden state.

        self.idx_store_actor = 0
        self.idx_retrieve_actor = 0
        self.idx_store_critic = 0
        self.idx_retrieve_critic = 0

        if th.cuda.is_available(): # if cuda device is available...
            self.actor_hidden[0] = self.actor_hidden[0].cuda() # ensure they are on cuda device by invoking 'cuda'
            self.actor_hidden[1] = self.actor_hidden[1].cuda() # ensure they are on cuda device by invoking 'cuda'
            #self.critic_hidden = self.critic_hidden.cuda()
            self.past_actor_hidden = self.past_actor_hidden.cuda()
            self.past_critic_hidden = self.past_critic_hidden.cuda()

        self.lstm = nn.LSTM(input_size=1920, hidden_size=512)

    @staticmethod
    def get_list(a, b, c, d):
        lst = [[[ [ ['', ''] for col in range (a) ] for col in range (b) ] for col in range (c)] for col in range (d)]


    def forward(self, observations: th.Tensor) -> th.Tensor:
        
        if observations.shape[0] > 1 and self.idx_retrieve_actor < self.n_rollout_steps: # if multiple observations are passed and not all states have been retrieved
            hidden_batch_actor = self.past_actor_hidden[self.idx_retrieve_actor // self.buffer_size, (self.idx_retrieve_actor % self.buffer_size) // self.batch_size]
            hidden_batch_critic = self.past_critic_hidden[self.idx_retrieve_actor // self.buffer_size, (self.idx_retrieve_actor % self.buffer_size) // self.batch_size] # retrieve a batch of states
            self.idx_retrieve_actor += self.batch_size # increment number of states retrieved
            
            output = th.zeros(64, 1, self.lstm_size).cuda() # create something to store the outputs
            for i in range(self.batch_size):
                single_observation = observations[i].unsqueeze(0) # get i-th row of batch observations
                hidden_state = hidden_batch_actor[i].unsqueeze(0) # get hidden state
                cell_state = hidden_batch_critic[i].unsqueeze(0) # get cell state
                output[i], _ = self.lstm(self.cnn(single_observation), (hidden_state, cell_state))
        else:
            output, self.actor_hidden = self.lstm(self.cnn(observations), self.actor_hidden)
        
        if observations.shape[0] <= 1 and self.idx_store_actor < self.n_rollout_steps: # if one observation is passed an not all states have been stored
            self.past_actor_hidden[self.idx_store_actor // self.buffer_size, (self.idx_store_actor % self.buffer_size) // self.batch_size, self.idx_store_actor % self.hidden_size] = self.actor_hidden[0] # store hidden state
            self.past_critic_hidden[self.idx_store_actor // self.buffer_size, (self.idx_store_actor % self.buffer_size) // self.batch_size, self.idx_store_actor % self.hidden_size] = self.actor_hidden[1]
            self.idx_store_actor += 1 # increment number of stored states
        if observations.shape[0] <= 1 and self.idx_store_actor >= self.n_rollout_steps: # if one observation is passed and all states have now been stored
            self.idx_store_actor = 0 # reset store index
        if observations.shape[0] > 1 and self.idx_retrieve_actor >= self.n_rollout_steps: # if multiple observations are passed and all states have been retrieved...
            self.actor_hidden = (self.past_actor_hidden[self.past_actor_hidden.shape[0] - 1, self.past_actor_hidden.shape[1] - 1, self.past_actor_hidden.shape[2] - 1].unsqueeze(0), 
                                self.past_critic_hidden[self.past_actor_hidden.shape[0] - 1, self.past_actor_hidden.shape[1] - 1, self.past_actor_hidden.shape[2] - 1].unsqueeze(0)) # restore hidden state to most recent state
            self.past_actor_hidden = th.zeros(self.n_epochs, self.buffer_size // self.batch_size, self.batch_size, self.lstm_size).cuda() # flush hidden state memory
            self.past_critic_hidden = th.zeros(self.n_epochs, self.buffer_size // self.batch_size, self.batch_size, self.lstm_size).cuda() # flush hidden state memory
            self.idx_retrieve_actor = 0 # reset retrieval index

        return output


def create_mlp(
    input_dim: int,
    output_dim: int,
    net_arch: List[int],
    activation_fn: Type[nn.Module] = nn.ReLU,
    squash_output: bool = False,
    with_bias: bool = True,
) -> List[nn.Module]:
    """
    Create a multi layer perceptron (MLP), which is
    a collection of fully-connected layers each followed by an activation function.

    :param input_dim: Dimension of the input vector
    :param output_dim:
    :param net_arch: Architecture of the neural net
        It represents the number of units per layer.
        The length of this list is the number of layers.
    :param activation_fn: The activation function
        to use after each layer.
    :param squash_output: Whether to squash the output using a Tanh
        activation function
    :param with_bias: If set to False, the layers will not learn an additive bias
    :return:
    """

    if len(net_arch) > 0:
        modules = [nn.Linear(input_dim, net_arch[0], bias=with_bias), activation_fn()]
    else:
        modules = []

    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1], bias=with_bias))
        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(nn.Linear(last_layer_dim, output_dim, bias=with_bias))
    if squash_output:
        modules.append(nn.Tanh())
    return modules