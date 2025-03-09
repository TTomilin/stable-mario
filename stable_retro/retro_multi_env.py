import gc
import gzip
import json
import os

import gymnasium as gym
import numpy as np

import stable_retro
import stable_retro.data
from stable_retro import RetroEnv
from stable_retro.enums import State

__all__ = ["RetroEnv"]


class RetroMultiEnv(RetroEnv):
    """
    Multi-environment Retro Gym class

    Allow multi-task learning in retro game environments
    """

    def __init__(
        self,
        game_list,
        state_list=None,
        scenario=None,
        info=None,
        use_restricted_actions=stable_retro.Actions.FILTERED,
        record=False,
        players=1,
        inttype=stable_retro.data.Integrations.STABLE,
        obs_type=stable_retro.Observations.IMAGE,
        render_mode="human",
    ):
        if not hasattr(self, "spec"):
            self.spec = None
        self._obs_type = obs_type
        self.img = None
        self.ram = None
        self.viewer = None
        self.game_list = game_list
        self.initial_state = None
        self.players = players

        self.inttype = inttype
        self.use_restricted_actions = use_restricted_actions
        self.record = record
        self.render_mode = render_mode
        self.scenario = scenario
        self.info = info

        # Don't return multiple rewards in multiplayer mode by default
        # as stable-baselines3 vectorized environments doesn't support it
        self.multi_rewards = False

        self.metadata_list = []
        self.rom_paths = []
        if state_list == None:
            self.state_list = [State.DEFAULT] * len(self.game_list)
        else:
            self.state_list = state_list

        for i, game in enumerate(game_list):
            rom_path = stable_retro.data.get_romfile_path(game, inttype)
            metadata_path = stable_retro.data.get_file_path(game, "metadata.json", inttype)
            self.rom_paths.append(rom_path)

            if state_list[i] == stable_retro.State.NONE:
                self.state_list[i] = None
            elif state_list[i] == stable_retro.State.DEFAULT:
                self.state_list[i] = None
                try:
                    with open(metadata_path) as f:
                        metadata = json.load(f)
                    if "default_player_state" in metadata and self.players <= len(
                        metadata["default_player_state"],
                    ):
                        self.state_list[i] = metadata["default_player_state"][self.players - 1]
                    elif "default_state" in metadata:
                        self.state_list[i] = metadata["default_state"]
                    else:
                        self.state_list[i] = None
                except (OSError, json.JSONDecodeError):
                    pass

        # select & load random:
        N = len(self.game_list)
        self.current_game = np.random.randint(0, N)
        self.mean_episode_rewards_per_game = [0] * N
        self.episodes_per_game = [0] * N
        self.current_episode_reward = 0
        self.load_game(self.current_game, self.inttype, self.use_restricted_actions, 
                       self.record, self.players, self.render_mode, self.scenario, self.info)
        
    def load_game(self, idx, inttype, use_restricted_actions, record, players, render_mode,
                  scenario=None, info=None):
        if self.state_list[idx]:
            self.load_state(idx, inttype)

        self.data = stable_retro.data.GameData()

        if info is None:
            info = "data"

        if info.endswith(".json"):
            # assume it's a path
            info_path = info
        else:
            info_path = stable_retro.data.get_file_path(self.game_list[idx], info + ".json", inttype)

        if scenario is None:
            scenario = "scenario"

        if scenario.endswith(".json"):
            # assume it's a path
            scenario_path = scenario
        else:
            scenario_path = stable_retro.data.get_file_path(self.game_list[idx], scenario + ".json", inttype)

        self.system = stable_retro.get_romfile_system(self.rom_paths[idx])

        # We can't have more than one emulator per process. Before creating an
        # emulator, ensure that unused ones are garbage-collected
        gc.collect()
        if hasattr(self, "em"):
            pass
            # del self.em
            # self.em = stable_retro.RetroEmulator(self.rom_paths[idx])
        else:
            self.em = stable_retro.RetroEmulator(self.rom_paths[idx])

        self.em.configure_data(self.data)
        self.em.step()

        core = stable_retro.get_system_info(self.system)
        self.buttons = core["buttons"]
        self.num_buttons = len(self.buttons)

        try:
            assert self.data.load(
                info_path,
                scenario_path,
            ), "Failed to load info ({}) or scenario ({})".format(
                info_path,
                scenario_path,
            )
        except Exception:
            del self.em
            raise

        self.button_combos = self.data.valid_actions()
        if use_restricted_actions == stable_retro.Actions.DISCRETE:
            combos = 1
            for combo in self.button_combos:
                combos *= len(combo)
            self.action_space = gym.spaces.Discrete(combos**players)
        elif use_restricted_actions == stable_retro.Actions.MULTI_DISCRETE:
            self.action_space = gym.spaces.MultiDiscrete(
                [len(combos) for combos in self.button_combos] * players,
            )
        else:
            self.action_space = gym.spaces.MultiBinary(self.num_buttons * players)

        if self._obs_type == stable_retro.Observations.RAM:
            shape = self.get_ram().shape
        else:
            img = [self.get_screen(p) for p in range(players)]
            shape = img[0].shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=shape,
            dtype=np.uint8,
        )

        self.use_restricted_actions = use_restricted_actions
        self.movie = None
        self.movie_id = 0
        self.movie_path = None
        if record is True:
            self.auto_record()
        elif record is not False:
            self.auto_record(record)

        self.render_mode = render_mode

    def step(self, a):
        if self.img is None and self.ram is None:
            raise RuntimeError("Please call env.reset() before env.step()")
        
        for p, ap in enumerate(self.action_to_array(a)):
            if self.movie:
                for i in range(self.num_buttons):
                    self.movie.set_key(i, ap[i], p)
            self.em.set_button_mask(ap, p)

        if self.movie:
            self.movie.step()
        self.em.step()
        self.data.update_ram()
        ob = self._update_obs()
        rew, done, info = self.compute_step()

        self.current_episode_reward += rew
        if done:
            self.episodes_per_game[self.current_game] += 1
            N = self.episodes_per_game[self.current_game]
            self.mean_episode_rewards_per_game[self.current_game] += (1/N) * self.current_episode_reward
            self.current_episode_reward = 0
            for i, game in enumerate(self.game_list):
                key = "{0}_reward".format(game)
                value = self.mean_episode_rewards_per_game[i]
                info[key] = value
            print(info)

        if self.render_mode == "human":
            self.render()

        return ob, rew, bool(done), False, dict(info)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if options != None and "games" in options and "game_probabilities" in options:
            game_probabilities = options["game_probabilities"]
            games = options["games"]
            choice = np.random.choice(games, p=game_probabilities)
            self.current_game = self.game_list.index(choice)
            self.load_game(self.current_game, self.inttype, self.use_restricted_actions, 
                       self.record, self.players, self.render_mode, self.scenario, self.info)
        else:
            N = len(self.game_list)
            choice = np.random.randint(0, N)
            self.current_game = choice
            self.load_game(self.current_game, self.inttype, self.use_restricted_actions, 
                       self.record, self.players, self.render_mode, self.scenario, self.info)

        if self.initial_state:
            self.em.set_state(self.initial_state)
        for p in range(self.players):
            self.em.set_button_mask(np.zeros([self.num_buttons], np.uint8), p)
        self.em.step()
        if self.movie_path is not None:
            rel_statename = os.path.splitext(os.path.basename(self.state_list[self.current_game]))[0]
            self.record_movie(
                os.path.join(
                    self.movie_path,
                    "%s-%s-%06d.bk2" % (self.gamename, rel_statename, self.movie_id),
                ),
            )
            self.movie_id += 1
        if self.movie:
            self.movie.step()
        self.data.reset()
        self.data.update_ram()

        if self.render_mode == "human":
            self.render()

        return self._update_obs(), {}
    
    def load_state(self, idx, inttype=stable_retro.data.Integrations.DEFAULT):
        if not self.state_list[idx].endswith(".state"):
            self.state_list[idx] += ".state"

        with gzip.open(
            stable_retro.data.get_file_path(self.game_list[idx], self.state_list[idx], inttype),
            "rb",
        ) as fh:
            self.initial_state = fh.read()