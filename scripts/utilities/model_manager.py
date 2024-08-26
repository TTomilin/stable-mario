import argparse
import numpy as np

from policy_arguments.feature_extractors import BatchNorm, NatureRNN

from stable_baselines3 import PPO
from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.common.torch_layers import NatureCNN, FlattenExtractor
from sb3_contrib import QRDQN, RecurrentPPO
from models.recurrent_policy import RecurrentPolicy
from torch import device, nn

class ModelManager:
    @staticmethod
    def create_model(cfg: argparse.Namespace, env: GymEnv, device: device, log_dir: str):
        shared_policy_args = ModelManager.__get_shared_policy_args(cfg)

        if cfg.model == "PPO":
            return ModelManager.__create_PPO(cfg, env, device, log_dir, shared_policy_args)
        elif cfg.model == "QRDQN":
            return ModelManager.__create_QRDQN(cfg, env, device, log_dir, shared_policy_args)
        elif cfg.model == "recurrent_PPO":
            return ModelManager.__create_recurrent_PPO(cfg, env, device, log_dir, shared_policy_args)
        elif cfg.model == "custom_recurrent_PPO":
            return ModelManager.__create_custom_recurrent_PPO(cfg, env, device, log_dir, shared_policy_args)
        else:
            raise ValueError("No model matching the model argument found.")

    @staticmethod
    def __create_PPO(cfg: argparse.Namespace, env: GymEnv, device: device, log_dir: str, policy_args: dict):

        ModelManager.__get_net_arch(cfg, policy_args)

        return PPO(policy='CnnPolicy', env=env, device=device, ent_coef=cfg.ent_coeff,
                    learning_rate=cfg.learning_rate,verbose=True, tensorboard_log=f"{log_dir}/tensorboard/",
                    policy_kwargs=policy_args, n_epochs=cfg.n_epochs, batch_size=cfg.batch_size, n_steps=cfg.n_steps)
    
    @staticmethod
    def __create_recurrent_PPO(cfg: argparse.Namespace, env: GymEnv, device: device, log_dir: str, policy_args: dict):

        ModelManager.__get_net_arch(cfg, policy_args)

        return RecurrentPPO(policy='CnnLstmPolicy', env=env, device=device, ent_coef=cfg.ent_coeff,
                    learning_rate=cfg.learning_rate,verbose=True, tensorboard_log=f"{log_dir}/tensorboard/",
                    policy_kwargs=policy_args, n_epochs=cfg.n_epochs, batch_size=cfg.batch_size, n_steps=cfg.n_steps)
    
    @staticmethod
    def __create_custom_recurrent_PPO(cfg: argparse.Namespace, env: GymEnv, device: device, log_dir: str, policy_args: dict):

        ModelManager.__get_net_arch(cfg, policy_args)

        RecurrentPolicy.batch_size = cfg.batch_size
        RecurrentPolicy.n_epochs = cfg.n_epochs
        RecurrentPolicy.buffer_size = cfg.n_steps # note, in PPO: buffer_size = n_envs * n_steps

        return PPO(policy=RecurrentPolicy, env=env, device=device, ent_coef=cfg.ent_coeff,
                    learning_rate=cfg.learning_rate,verbose=True, tensorboard_log=f"{log_dir}/tensorboard/",
                    policy_kwargs=policy_args)

    @staticmethod
    def __create_QRDQN(cfg: argparse.Namespace, env: GymEnv, device: device, log_dir: str, policy_args: dict):
        
        if cfg.batch_norm and policy_args != None:
            policy_args.update(features_extractor_class = BatchNorm)

        return QRDQN(policy='CnnPolicy', env=env, device=device,
                    learning_rate=cfg.learning_rate,verbose=True, tensorboard_log=f"{log_dir}/tensorboard/",
                    policy_kwargs=policy_args, batch_size=cfg.batch_size) # outperforms DQN, see https://arxiv.org/pdf/1710.10044
                    
    @staticmethod
    def load_model(model_type: str, game: str, load_directory: str, env):
        model = None
        if model_type == "PPO":
            model = ModelManager.__try_load_model(load_directory, [game, f"{game}-bak"], PPO, env)
        elif model_type == "QR-DQN":
            model = ModelManager.__try_load_model(load_directory, [game, f"{game}-bak"], QRDQN, env)
        elif model_type == "recurrent_PPO":
            model = ModelManager.__try_load_model(load_directory, [game, f"{game}-bak"], RecurrentPPO, env)
        else:
            raise ValueError("No model matching the model argument found. Aborting...")
        
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
    
    @staticmethod
    def __get_net_arch(cfg: argparse.Namespace, policy_args: dict):
        if cfg.pi == None or cfg.vf == None:
            default_net_arch = dict()

            default_pi = [256, 256]
            default_vf = [256, 256]
            default_net_arch.update(pi = default_pi)
            default_net_arch.update(vf = default_vf)

            policy_args.update(net_arch = default_net_arch)
        else:
            inputted_net_arch = dict()

            inputted_pi = np.array([int(num_str) for num_str in cfg.pi.split(",")])
            inputted_vf = np.array([int(num_str) for num_str in cfg.vf.split(",")])
            inputted_net_arch.update(pi = inputted_pi)
            inputted_net_arch.update(vf = inputted_vf)

            policy_args.update(net_arch = inputted_net_arch)
    
    @staticmethod
    def __get_shared_policy_args(cfg: argparse.Namespace):
        policy_args = dict()

        if cfg.activation_function == "tanh":
            policy_args.update(activation_fn=nn.Tanh)
        elif cfg.activation_function == "relu":
            policy_args.update(activation_fn=nn.ReLU)
        elif cfg.activation_function == "leaky_relu":
            policy_args.update(activation_fn=nn.LeakyReLU)
        else:
            raise ValueError("Invalid activation function specified.")

        if cfg.features_extractor == "NatureCNN":
            policy_args.update(features_extractor_class=NatureCNN)
        elif cfg.features_extractor == "NatureRNN":
            policy_args.update(features_extractor_class=NatureRNN)
        elif cfg.features_extractor == "Flatten":
            policy_args.update(features_extractor_class=FlattenExtractor)
        else:
            raise ValueError("Invalid feature extractor specified.")

        if cfg.features_extractor_dim != None:
            policy_args.update(features_extractor_kwargs=dict(features_dim=cfg.features_extractor_dim))

        if bool(policy_args) == False: # if dict left empty...
            policy_args = None # set to default None 

        return policy_args