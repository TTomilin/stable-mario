import argparse
import json
import os
import statistics
import sys

import gymnasium
import wandb
from colorist import Color
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from torch import device

from utilities.model_manager import ModelManager

RESUME_COMMAND = 'resume_training.py'
REWARD_KEY = 'mean_reward'
N_EPISODES_KEY = 'number_of_eval_episodes'
DETERMINISTIC_KEY = 'deterministic'


class CustomEvalCallback(BaseCallback):
    """
    A custom callback that evaluates the agent and saves the model if it was better than the previous agent.
    """

    def __init__(self, cfg: argparse.Namespace, eval_env: gymnasium.Env, log_dir: str, device: device,
                 system_file_name: str,
                 wandb_file_name: str, eval_freq: int = 300, n_eval_episodes: int = 1, deterministic: bool = True,
                 verbose: int = 0,
                 eval_metric: str = None):
        super().__init__(verbose)
        self.__n_eval_episodes = n_eval_episodes
        self.__eval_env = eval_env
        self.__log_dir = log_dir
        self.__system_file_name = system_file_name
        self.__wandb_file_name = wandb_file_name
        self.__cfg = cfg
        self.__device = device
        self.__eval_freq = eval_freq
        self.__deterministic = deterministic
        self.__ep_completed_since_update = 0
        self.__eval_metric = eval_metric

        # create local copy of model:
        self.__local_model = ModelManager.create_model(cfg=self.__cfg, env=self.__eval_env,
                                                       device=self.__device, log_dir=self.__log_dir)

        # initialize the previous best reward:
        self.__previous_best_total_reward = float('-inf')

        if os.path.isfile(f"{log_dir}/best_model/{system_file_name}_info.json"):
            with open(f"{log_dir}/best_model/{system_file_name}_info.json", 'r') as json_file:
                save_dict = json.load(json_file)
                try:
                    self.__previous_best_total_reward = save_dict[REWARD_KEY]
                except KeyError:
                    pass
        if self.__previous_best_total_reward == float('-inf') and sys.argv[0] == RESUME_COMMAND:
            print(
                f"{Color.RED}WARNING: could not find valid json file for best model.\nSetting best previous evaluation reward to default -infinity.{Color.OFF}")

    def _on_step(self) -> bool:
        if self.locals["dones"][0]:
            self.__ep_completed_since_update += 1
        # interesting note: directly accessing 'self.model.ep_info_buffer' in this function will result in constant rewards each episode

        if self.__ep_completed_since_update >= self.__eval_freq:
            if self.__eval_metric == None:
                mean_reward = self.__get_average_reward()
            else:
                mean_reward = self.__get_average_RAM_value(self.__eval_metric)
            print(f"{Color.RED}mean evaluation reward: {mean_reward}{Color.OFF}")

            if mean_reward > self.__previous_best_total_reward:
                self.__previous_best_total_reward = mean_reward
                self.__save_new_best_model(mean_reward)

            self.__ep_completed_since_update = 0

        return True  # always continue training

    def __save_new_best_model(self, mean_reward):
        self.model.save(f"{self.__log_dir}/best_model/{self.__system_file_name}.zip")

        save_dict = {REWARD_KEY: mean_reward,
                     DETERMINISTIC_KEY: self.__deterministic,
                     N_EPISODES_KEY: self.__n_eval_episodes}
        with open(f"{self.__log_dir}/best_model/{self.__system_file_name}_info.json", 'w') as json_file:
            json.dump(save_dict, json_file)

        if self.__cfg.with_wandb:
            wandb.save(f"{self.__log_dir}/best_model/{self.__wandb_file_name}.zip")
            wandb.save(f"{self.__log_dir}/best_model/{self.__wandb_file_name}_info.json")

    def __get_average_reward(self):
        total_episode_rewards = []
        if self.__deterministic:
            actor_params = self.model.policy.state_dict()
            self.__local_model.policy.load_state_dict(actor_params)
            total_episode_rewards, total_episode_lengths = evaluate_policy(model=self.__local_model,
                                                                           env=self.__eval_env,
                                                                           n_eval_episodes=1,
                                                                           deterministic=True,
                                                                           return_episode_rewards=True)
        else:
            total_episode_rewards, total_episode_lengths = evaluate_policy(model=self.model,
                                                                           env=self.__eval_env,
                                                                           n_eval_episodes=self.__n_eval_episodes,
                                                                           deterministic=False,
                                                                           return_episode_rewards=True)

        mean_reward = statistics.fmean(total_episode_rewards)
        return mean_reward

    def __get_average_RAM_value(self, ram_id):
        episode_ram_values = []
        actor_params = self.model.policy.state_dict()
        self.__local_model.policy.load_state_dict(actor_params)

        if self.__deterministic:
            obs, _ = self.__eval_env.reset()
            while True:
                if self.__cfg.discretize:
                    action = self.__local_model.predict(obs, deterministic=True)[0]
                else:
                    action = self.__local_model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.__eval_env.step(action)

                if terminated or truncated:
                    episode_ram_values.append(info[ram_id])
                    obs, _ = self.__eval_env.reset()
                    action = self.__local_model.predict(obs,
                                                        deterministic=False)  # reset determinism. Perhaps this would not be needed if we set 'eval_env' to an environment the same as 'env' instead of to 'env' itself
                    break
        else:
            obs, _ = self.__eval_env.reset()
            i = 0
            while i < self.__n_eval_episodes:
                if self.__cfg.discretize:
                    action = self.__local_model.predict(obs, deterministic=False)[0]
                else:
                    action = self.__local_model.predict(obs, deterministic=False)
                obs, reward, terminated, truncated, info = self.__eval_env.step(action)

                if terminated or truncated:
                    episode_ram_values.append(info[ram_id])
                    i += 1
                    obs, _ = self.__eval_env.reset()

        return statistics.fmean(episode_ram_values)
