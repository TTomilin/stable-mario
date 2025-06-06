import os
from typing import Optional

from gymnasium.wrappers import ResizeObservation, RecordEpisodeStatistics

import stable_retro
from sample_factory.envs.env_wrappers import (
    ClipRewardEnv,
    MaxAndSkipEnv,
    NoopResetEnv, PixelFormatChwWrapper,
)
from sample_factory.mario.record_video import RecordVideo
from sample_factory.utils.utils import experiment_dir
from scripts.config import CONFIG
from stable_retro.examples.discretizer import Discretizer
from scripts.wrappers.observation import CustomFrameStack


class MarioSpec:
    def __init__(self, name, env_id, default_timeout=None):
        self.name = name
        self.env_id = env_id
        self.default_timeout = default_timeout


MARIO_ENVS = [
    MarioSpec("broom_zoom", "BroomZoom-v0"),
    MarioSpec("spook_spike", "SpookSpike-v0"),
    MarioSpec("flippin_out", "FlippinOut-v0"),
    MarioSpec("on_the_spot", "OnTheSpot-v0"),
    MarioSpec("amplifried", "Amplifried-v0"),
    MarioSpec("bill_bounce", "BillBounce-v0"),
    MarioSpec("bunny_belt", "BunnyBelt-v0"),
    MarioSpec("pest_aside", "PestAside-v0"),
    MarioSpec("match-em", "MatchEm-v0"),
    MarioSpec("hammergeddon", "Hammergeddon-v0"),
    MarioSpec("sort_stack", "SortStack-v0"),
    MarioSpec("stompbot_xl", "StompbotXL-v0"),
    MarioSpec("fling_shot", "FlingShot-v0"),
    MarioSpec("big_popper", "BigPopper-v0"),
    MarioSpec("melon_folly", "MelonFolly-v0"),
    MarioSpec("cloud_climb", "CloudClimb-v0"),
    MarioSpec("grabbit", "Grabbit-v0"),
    MarioSpec("forest_jump", "ForestJump-v0"),
    MarioSpec("drop_em", "DropEm-v0"),
    MarioSpec("barrel_peril", "BarrelPeril-v0"),
    MarioSpec("bob_ooom", "BobOoom-v0"),
    MarioSpec("boo_bye", "BooBye-v0"),
    MarioSpec("chicken", "Chicken-v0"),
    MarioSpec("chomp_walker", "ChompWalker-v0"),
    MarioSpec("crushed_ice", "CrushedIce-v0"),
    MarioSpec("dreadmill", "Dreadmill-v0"),
    MarioSpec("floor_it", "FloorIt-v0"),
    MarioSpec("go_go_pogo", "GoGoPogo-v0"),
    MarioSpec("koopa_crunch", "KoopaCrunch-v0"),
    MarioSpec("outta_my_way", "OuttaMyWay-v0"),
    MarioSpec("reel_cheep", "ReelCheep-v0"),
    MarioSpec("see_monkey", "SeeMonkey-v0"),
    MarioSpec("shell_stack", "ShellStack-v0"),
    MarioSpec("sled_slide", "SledSlide-v0"),
    MarioSpec("splatterball", "Splatterball-v0"),
    MarioSpec("stop_em", "StopEm-v0"),
    MarioSpec("tankdown", "Tankdown-v0"),
    MarioSpec("switch_way", "SwitchWay-v0"),
    MarioSpec("trap_floor","TrapFloor-v0"),
    MarioSpec("stair_scare","StairScare-v0"),
    MarioSpec("chainsaw","ChainSaw-v0"),
    MarioSpec("scratch-em","ScratchEm-v0"),
    MarioSpec("koopa_kurl","KoopaKurl-v0"),
    MarioSpec("pair-em","PairEm-v0"),
    MarioSpec("watch-em","WatchEm-v0"),
    MarioSpec("slammer", "Slammer-v0"),
    MarioSpec("much_rush","MuchRush-v0"),
    MarioSpec("peek_n_sneak","PeakNSneak-v0"),
    MarioSpec("overworld", "Overworld-v0"),
    MarioSpec("melon_folly0", "MelonFolly0-v0"),
    MarioSpec("melon_folly1", "MelonFolly1-v0"),
    MarioSpec("melon_folly2", "MelonFolly2-v0"),
    MarioSpec("volley_bomb", "VolleyBomb-v0"),
    MarioSpec("koopa_kappa", "KoopaKappa-v0"),
]



def mario_env_by_name(name):
    for cfg in MARIO_ENVS:
        if cfg.name == name:
            return cfg
    raise Exception("Unknown Minigame")


def make_mario_env(env_name, cfg, env_config, render_mode: Optional[str] = None):
    mario_spec = mario_env_by_name(env_name)
    if cfg.game_list == None:
        # initialize single-game environment:
        game_config = CONFIG[cfg.game]
        state = cfg.load_state if cfg.load_state is not None else game_config["state"]
        if cfg.game == 'overworld':
            env = stable_retro.make_overworld(cfg=cfg, game=game_config['game_env'], state=state, render_mode=render_mode)
        else:
            env = stable_retro.make(game=game_config['game_env'], state=state, render_mode=render_mode)
        env.metadata["render_fps"] = cfg.render_fps
    else:
        # initialize multi-game environment:
        game_list = []
        state_list = []
        for i, cfg_game in enumerate(cfg.game_list):
            game_config = CONFIG[cfg_game]
            game_list.append(game_config["game_env"])
            if len(state_list) == len(game_list):
                state_list.append(cfg.state_list[i])
            else:
                state_list.append(game_config["state"])
        env = stable_retro.make_multi(game_list=game_list, state_list=state_list, render_mode=render_mode, min_task_repeat=cfg.min_task_repeat)
        env.metadata["render_fps"] = cfg.render_fps

    if cfg.discretize:
        env = Discretizer(env, game_config["actions"])

    if game_config["clip_reward"]:
        env = ClipRewardEnv(env)

    if cfg.record:
        video_folder = os.path.join(experiment_dir(cfg), cfg.video_dir)
        env = RecordVideo(env, video_folder=video_folder, step_trigger=lambda step: not step % cfg.record_every,
                          name_prefix='mario', video_length=cfg.video_length, dummy_env=env_config is None)

    if mario_spec.default_timeout is not None:
        env._max_episode_steps = mario_spec.default_timeout

    if cfg.stack_frames:
        env = CustomFrameStack(env, cfg.n_stack_frames)

    # these are chosen to match Stable-Baselines3 and CleanRL implementations as precisely as possible
    env = RecordEpisodeStatistics(env)
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=cfg.env_frameskip)
    env = ResizeObservation(env, game_config["resize"])
    env = PixelFormatChwWrapper(env)
    # env = FrameStack(env, cfg.env_framestack)  # TODO out of order

    return env
