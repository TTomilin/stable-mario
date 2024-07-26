import argparse

from policy_arguments.feature_extractors import BatchNorm

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.type_aliases import GymEnv
from sb3_contrib import QRDQN
from torch import device

class ModelManager:
    @staticmethod
    def create_model(cfg: argparse.Namespace, env: GymEnv, device: device, log_dir: str):
        if cfg.model == "PPO":
            return ModelManager.__create_PPO(cfg, env, device, log_dir)
        elif cfg.model == "QRDQN":
            return ModelManager.__create_QRDQN(cfg, env, device, log_dir)
        else:
            return ValueError("No model matching the model argument found.")

    @staticmethod
    def __create_PPO(cfg: argparse.Namespace, env: GymEnv, device: device, log_dir: str):
        return PPO(policy='CnnPolicy', env=env, device=device, ent_coef=cfg.ent_coeff,
                    learning_rate=cfg.learning_rate,verbose=True, tensorboard_log=f"{log_dir}/tensorboard/")

    @staticmethod
    def __create_QRDQN(cfg: argparse.Namespace, env: GymEnv, device: device, log_dir: str):

        policy_args = dict()

        if cfg.batch_norm:
            policy_args.update(features_extractor_class = BatchNorm)
        else:
            policy_args = None # if no value added, set it to default 'None'

        return QRDQN(policy='CnnPolicy', env=env, device=device,
                    learning_rate=cfg.learning_rate,verbose=True, tensorboard_log=f"{log_dir}/tensorboard/",
                    policy_kwargs=policy_args) # outperforms DQN, see https://arxiv.org/pdf/1710.10044
                    
    @staticmethod
    def load_model(model_type: str, game: str, load_directory: str, env):
        model = None
        if model_type == "PPO":
            model = ModelManager.__try_load_model(load_directory, [game, f"{game}-bak"], PPO, env)
        elif model_type == "QR-DQN":
            model = ModelManager.__try_load_model(load_directory, [game, f"{game}-bak"], QRDQN, env)
        else:
            return ValueError("No model matching the model argument found. Aborting...")
        
        return model

    @staticmethod
    def __try_load_model(load_directory, names, model_type, env):
        model = None
        for name in names:
            try:
                model = model_type.load(f"{load_directory}/{name}", env=env)
                print(f"loaded {name}")
                break
            except FileNotFoundError:
                pass
        if model == None:
            print("Could not find model's zipfile. Please check if the file is present and whether its name is <game_name>.zip/<game_name>-bak.zip")
        return model