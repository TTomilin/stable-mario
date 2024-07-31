import sys

import argparse
from utilities.load_parser import LoadParser
from utilities.train_parser import TrainParser
from utilities.environment_creator import RetroEnvCreator
from utilities.wandb_manager import WandbManager

import os
from datetime import datetime

import wandb
from stable_baselines3 import PPO

from sb3_contrib import QRDQN

from config import CONFIG


def main(cfg: argparse.Namespace):
    # Create directory to store footage of trained model:
    load_directory = cfg.directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"{load_directory}/trained_footage/{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    
    # Load training parameters:
    train_command = None
    with open(f"{load_directory}/train_command.txt", 'r') as file:
        train_command = file.read()
    argv = train_command.split(" ") # convert arguments to list
    train_parser = TrainParser(arg_source=argv[1:]) # feed arguments into parser
    reinit_env_args = train_parser.get_args()
    reinit_env_args_dict = vars(reinit_env_args)

    # if needed, sync with wandb
    if cfg.with_wandb:
        WandbManager.InitializeWandb(reinit_env_args, log_dir + "load_at_", timestamp)

    # set render-mode based and some environment arguments based on args passed over cli:
    render_mode = None
    if cfg.record:
        reinit_env_args_dict['render_mode'] = "rgb_array"
        reinit_env_args_dict['record'] = True
        reinit_env_args_dict['record_every'] = cfg.record_every
    else:
        reinit_env_args_dict['record'] = False
        reinit_env_args_dict['render_mode'] = "human"

    # disable time limit:
    reinit_env_args_dict["time_limit"] = None

    # Create environment
    env = RetroEnvCreator.create(reinit_env_args, log_dir, CONFIG)

    # Load the model
    model = None
    if reinit_env_args.model == "PPO":
        model = try_load_model(load_directory, [reinit_env_args.game, f"{reinit_env_args.game}-bak"], PPO, env)
    elif reinit_env_args.model == "QR-DQN":
        model = try_load_model(load_directory, [reinit_env_args.game, f"{reinit_env_args.game}-bak"], QRDQN, env)
    else:
        print("No model matching the model argument found. Aborting...")
        exit()

    # Get determinism setting:
    deterministic = cfg.deterministic

    # Show the model
    obs, _ = env.reset()
    while True:
        env.render()
        
        if reinit_env_args.discretize:
            action = model.predict(obs, deterministic=deterministic)[0] # Model's action are returned as tuple with one element. Corresponds to discretized action.
        else:
            action = model.predict(obs, deterministic)

        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            obs, _ = env.reset()

def try_load_model(directory, names, model_type, env):
    model = None
    for name in names:
        try:
            model = model_type.load(f"{directory}/{name}", env=env)
            print(f"loaded {name}")
            break
        except FileNotFoundError:
            pass
    if model == None:
        print("Could not find model's zipfile. Please check if the file is present and whether its name is <game_name>.zip/<game_name>-bak.zip")
    return model

if __name__ == '__main__':
    parser = LoadParser(arg_source=sys.argv[1:])
    args = parser.get_args()

    main(args)
