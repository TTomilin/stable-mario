import sys
import os

sample_factory_path = '/'.join(os.getcwd().split("/")[:-2])
sys.path.append(sample_factory_path)

from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.envs.env_utils import register_env
from sample_factory.mario.mario_params import mario_override_defaults
from sample_factory.mario.mario_utils import MARIO_ENVS, MarioSpec, make_mario_env
from sample_factory.train import run_rl


def register_mario_envs():
    for env in MARIO_ENVS:
        register_env(env.name, make_mario_env)

def add_multi_env(cfg):
    name = ",".join(cfg.game_list)
    MARIO_ENVS.append(MarioSpec(name, "{0}-v0".format(name)))

def register_mario_components(cfg):
    if cfg.game_list != None:
        add_multi_env(cfg)
    register_mario_envs()


def parse_mario_args(argv=None, evaluation=False):
    parser, partial_cfg = parse_sf_args(argv=argv, evaluation=evaluation)
    mario_override_defaults(partial_cfg.game, parser)
    final_cfg = parse_full_cfg(parser, argv)
    return final_cfg


def main():  # pragma: no cover
    """Script entry point."""
    cfg = parse_mario_args()
    register_mario_components(cfg)
    status = run_rl(cfg)
    return status


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
