import warnings
from sample_factory.model.actor_critic import ActorCriticSharedWeights, ActorCritic
from sample_factory.algo.utils.context import global_model_factory
from stable_retro.examples.discretizer import Discretizer
from gymnasium.wrappers import ResizeObservation, RecordEpisodeStatistics
from gymnasium.spaces import Dict
from sample_factory.envs.env_wrappers import NoopResetEnv, MaxAndSkipEnv, PixelFormatChwWrapper
import torch

import gymnasium as gym
import numpy as np

import stable_retro
from stable_retro.retro_env import RetroEnv
from scripts.config import CONFIG

class OverworldEnv(gym.Env):
    """
    Overworld environment class.

    Allows an overworld model to navigate Mario Party's gameboard, whilst minigame playing is handled by other agents.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "video.frames_per_second": 60.0}
    DEFAULT_MODEL_PATH = ""

    def __init__(
        self,
        game,
        state=stable_retro.State.DEFAULT,
        scenario=None,
        info=None,
        use_restricted_actions=stable_retro.Actions.FILTERED,
        record=False,
        players=1,
        inttype=stable_retro.data.Integrations.STABLE,
        obs_type=stable_retro.Observations.IMAGE,
        render_mode="human",
        cfg=None
    ):
        # create retro environment:
        self.retro_env = RetroEnv(game=game, state=state, scenario=scenario, info=info,
                                  use_restricted_actions=use_restricted_actions, record=record,
                                  players=players, inttype=inttype, obs_type=obs_type, render_mode=render_mode)
        self.action_space = self.retro_env.action_space
        self.observation_space = self.retro_env.observation_space
        self.buttons = self.retro_env.buttons
        self.cfg = cfg

    def step(self, a):
        ob, rew, done, truncated, info = self.retro_env.step(a)

        if info['game_ram_num'] != 0: # if a minigame is encountered...
            self.__handle_minigame(ob, info) # handle it

        ## TODO Check for minigame activation.
        ## TODO Find correct model for given minigame.
        ## TODO divert control to appropriate model until RAM flag for minigame has been set off again.

        return ob, rew, done, truncated, info
    
    def __handle_minigame(self, ob, info):
        # determine what game is being played:
        current_game_dict = self.get_game_dict(info)
        
        # find correct model to load:
        if current_game_dict != None:
            model_path = current_game_dict['model_path']
        else:
            model_path = OverworldEnv.DEFAULT_MODEL_PATH
        
        # load the model:
        model = self.load_model(model_path, current_game_dict)
        # prepare the environment:
        self.retro_env = self.set_env_to_game(self.retro_env, current_game_dict)
        # play the game:
        self.complete_minigame(model, ob, info)
        # restore environment:
        self.retro_env = self.set_env_to_game(self.retro_env, CONFIG['overworld'])

    def complete_minigame(self, model: ActorCritic, ob, info):
        # have the model play the game until the game is over:
        while info['game_ram_num'] != 0:
            result = model.forward(ob) # model processes observation
            action = result['actions'] # determine action from result
            ob, _, _, _, info = self.retro_env(action) # take action, get next observation


    def set_env_to_game(self, retro_env, game_dict):
        retro_env = retro_env.unwrapped # unwrap the environment completely

        # Just defaults for now. TODO: make this work based on a loaded CLI-command with which env was made.
        retro_env = Discretizer(retro_env, game_dict["actions"]) # add discretizer
        retro_env = RecordEpisodeStatistics(retro_env)
        retro_env = NoopResetEnv(retro_env, noop_max=30)
        retro_env = MaxAndSkipEnv(retro_env, skip=4)
        retro_env = ResizeObservation(retro_env, game_dict["resize"])
        retro_env = PixelFormatChwWrapper(retro_env)
        return retro_env

    def load_model(self, path, game_dict):
        if path.endswith('.pth'):
            # determine the observation and action space:
            resolution = game_dict['resize']
            n_actions = len(game_dict['actions'])
            model_obs_space = Dict({"obs": gym.spaces.Box(0, 255, (3, resolution[0], resolution[1]), np.uint8)})
            model_act_space = gym.spaces.Discrete(n_actions)

            # create corresponding AC:            
            model_factory = global_model_factory()
            model = ActorCriticSharedWeights(model_factory=model_factory, obs_space=model_obs_space, 
                                            action_space=model_act_space, cfg=self.cfg)
            # load weights into this AC:
            checkpoint = torch.load(path)
            model.load_state_dict(checkpoint['model'])

            return model
        else:
            raise ValueError("File extension of {0} not recognized.".format(path))
        
    def get_game_dict(self, info):
        current_game_dict = None
        game_ram_num = info['game_ram_num']
        for game_dict in CONFIG.values():
            if game_dict['game_ram_num'] == game_ram_num:
                current_game_dict = game_dict
                break
        if current_game_dict == None:
            warnings.warn("Unkown minigame encountered. Resorting to default model.")
        return current_game_dict

    def reset(self, seed=None, options=None):
        obs = self.retro_env.reset()
        return obs, {}

    def render(self):
        img = self.retro_env.render()
        return img

    def close(self):
        self.retro_env.close()

    def load_state(self, statename, inttype=stable_retro.data.Integrations.DEFAULT):
        self.retro_env.load_state(statename=statename, inttype=inttype)
