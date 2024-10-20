import sys

from sample_factory.enjoy import enjoy
from sample_factory.mario.train_mario import register_mario_components, parse_mario_args


def main():
    """Script entry point."""
    register_mario_components()
    cfg = parse_mario_args(evaluation=True)

    status = enjoy(cfg)
    return status


if __name__ == "__main__":
    sys.exit(main())
