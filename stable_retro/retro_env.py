import gc
import gzip
import json
import os

import gymnasium as gym
import numpy as np

import stable_retro
import stable_retro.data

__all__ = ["RetroEnv"]


class RetroEnv(gym.Env):
    """
    Gym Retro environment class

    Provides a Gym interface to classic video games
    """

    metadata = {"render_modes": ["human", "rgb_array"], "video.frames_per_second": 60.0}

    # get_action_meanings ADDED BY FABRICE:
    def get_action_meanings(self):
        return ["NOOP", 0];

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
    ):
        if not hasattr(self, "spec"):
            self.spec = None
        self._obs_type = obs_type
        self.img = None
        self.ram = None
        self.viewer = None
        self.gamename = game
        self.statename = state
        self.initial_state = None
        self.players = players

        # Don't return multiple rewards in multiplayer mode by default
        # as stable-baselines3 vectorized environments doesn't support it
        self.multi_rewards = False

        metadata = {}
        rom_path = stable_retro.data.get_romfile_path(game, inttype)
        metadata_path = stable_retro.data.get_file_path(game, "metadata.json", inttype)

        if state == stable_retro.State.NONE:
            self.statename = None
        elif state == stable_retro.State.DEFAULT:
            self.statename = None
            try:
                with open(metadata_path) as f:
                    metadata = json.load(f)
                if "default_player_state" in metadata and self.players <= len(
                    metadata["default_player_state"],
                ):
                    self.statename = metadata["default_player_state"][self.players - 1]
                elif "default_state" in metadata:
                    self.statename = metadata["default_state"]
                else:
                    self.statename = None
            except (OSError, json.JSONDecodeError):
                pass

        if self.statename:
            self.load_state(self.statename, inttype)

        self.data = stable_retro.data.GameData()

        if info is None:
            info = "data"

        if info.endswith(".json"):
            # assume it's a path
            info_path = info
        else:
            info_path = stable_retro.data.get_file_path(game, info + ".json", inttype)

        if scenario is None:
            scenario = "scenario"

        if scenario.endswith(".json"):
            # assume it's a path
            scenario_path = scenario
        else:
            scenario_path = stable_retro.data.get_file_path(game, scenario + ".json", inttype)

        self.system = stable_retro.get_romfile_system(rom_path)

        # We can't have more than one emulator per process. Before creating an
        # emulator, ensure that unused ones are garbage-collected
        gc.collect()
        self.em = stable_retro.RetroEmulator(rom_path)
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

    def _update_obs(self):
        if self._obs_type == stable_retro.Observations.RAM:
            self.ram = self.get_ram()
            return self.ram
        elif self._obs_type == stable_retro.Observations.IMAGE:
            self.img = self.get_screen()
            return self.img
        else:
            raise ValueError(f"Unrecognized observation type: {self._obs_type}")

    def action_to_array(self, a):
        actions = []
        for p in range(self.players):
            action = 0
            if self.use_restricted_actions == stable_retro.Actions.DISCRETE:
                for combo in self.button_combos:
                    current = a % len(combo)
                    a //= len(combo)
                    action |= combo[current]
            elif self.use_restricted_actions == stable_retro.Actions.MULTI_DISCRETE:
                ap = a[self.num_buttons * p : self.num_buttons * (p + 1)]
                for i in range(len(ap)):
                    buttons = self.button_combos[i]
                    action |= buttons[ap[i]]
            else:
                ap = a[self.num_buttons * p : self.num_buttons * (p + 1)]
                for i in range(len(ap)):
                    action |= int(ap[i]) << i
                if self.use_restricted_actions == stable_retro.Actions.FILTERED:
                    action = self.data.filter_action(action)
            ap = np.zeros([self.num_buttons], np.uint8)
            for i in range(self.num_buttons):
                ap[i] = (action >> i) & 1
            actions.append(ap)
        return actions

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

        if self.render_mode == "human":
            self.render()

        return ob, rew, bool(done), False, dict(info)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.initial_state:
            self.em.set_state(self.initial_state)
        for p in range(self.players):
            self.em.set_button_mask(np.zeros([self.num_buttons], np.uint8), p)
        self.em.step()
        if self.movie_path is not None:
            rel_statename = os.path.splitext(os.path.basename(self.statename))[0]
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

    def render(self):
        mode = self.render_mode

        img = self.get_screen() if self.img is None else self.img
        if mode == "rgb_array":
            return img
        elif mode == "human":
            if self.viewer is None:
                from stable_retro.rendering import SimpleImageViewer

                self.viewer = SimpleImageViewer()
            self.viewer.imshow(img)
            return self.viewer.isopen

    def close(self):
        if hasattr(self, "em"):
            del self.em
        if self.viewer:
            self.viewer.close()

    def get_action_meaning(self, act):
        actions = []
        for p, action in enumerate(self.action_to_array(act)):
            actions.append(
                [self.buttons[i] for i in np.extract(action, np.arange(len(action)))],
            )
        if self.players == 1:
            return actions[0]
        return actions

    def get_ram(self):
        blocks = []
        for offset in sorted(self.data.memory.blocks):
            arr = np.frombuffer(self.data.memory.blocks[offset], dtype=np.uint8)
            blocks.append(arr)
        return np.concatenate(blocks)

    def get_screen(self, player=0):
        img = self.em.get_screen()
        x, y, w, h = self.data.crop_info(player)
        if not w or x + w > img.shape[1]:
            w = img.shape[1]
        else:
            w += x
        if not h or y + h > img.shape[0]:
            h = img.shape[0]
        else:
            h += y
        if x == 0 and y == 0 and w == img.shape[1] and h == img.shape[0]:
            return img
        return img[y:h, x:w]

    def load_state(self, statename, inttype=stable_retro.data.Integrations.DEFAULT):
        if not statename.endswith(".state"):
            statename += ".state"

        with gzip.open(
            stable_retro.data.get_file_path(self.gamename, statename, inttype),
            "rb",
        ) as fh:
            self.initial_state = fh.read()

        self.statename = statename

    def compute_step(self):
        if self.players > 1 and self.multi_rewards:
            reward = [self.data.current_reward(p) for p in range(self.players)]
        else:
            reward = self.data.current_reward()
        done = self.data.is_done()
        return reward, done, self.data.lookup_all()

    def record_movie(self, path):
        self.movie = stable_retro.Movie(path, True, self.players)
        self.movie.configure(self.gamename, self.em)
        if self.initial_state:
            self.movie.set_state(self.initial_state)

    def stop_record(self):
        self.movie_path = None
        self.movie_id = 0
        if self.movie:
            self.movie.close()
            self.movie = None

    def auto_record(self, path=None):
        if not path:
            path = os.getcwd()
        self.movie_path = path
