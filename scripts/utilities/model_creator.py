import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.type_aliases import GymEnv
from sb3_contrib import QRDQN
from torch import device

class ModelCreator:
    @staticmethod
    def CreateModel(cfg: argparse.Namespace, env: GymEnv, device: device, log_dir: str):
        if cfg.model == "PPO":
            return ModelCreator.__CreatePPO(cfg, env, device, log_dir)
        elif cfg.model == "QRDQN":
            return ModelCreator.__CreateQRDQN(cfg, env, device, log_dir)
        else:
            return ValueError("No model matching the model argument found.")

    @staticmethod
    def __CreatePPO(cfg: argparse.Namespace, env: GymEnv, device: device, log_dir: str):
        return PPO(policy='CnnPolicy', env=env, device=device, ent_coef=cfg.ent_coeff,
                    learning_rate=cfg.learning_rate,verbose=True, tensorboard_log=f"{log_dir}/tensorboard/")

    @staticmethod
    def __CreateQRDQN(cfg: argparse.Namespace, env: GymEnv, device: device, log_dir: str):
        return QRDQN(policy='CnnPolicy', env=env, device=device,
                    learning_rate=cfg.learning_rate,verbose=True, tensorboard_log=f"{log_dir}/tensorboard/")