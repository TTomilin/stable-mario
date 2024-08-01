from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from utilities.model_manager import ModelManager
from torch import device

import gymnasium
import wandb
import statistics
import argparse

class CustomEvalCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, cfg: argparse.Namespace, eval_env: gymnasium.Env, log_dir: str, device: device, system_file_name: str, 
                 wandb_file_name: str, eval_freq: int = 300, n_eval_episodes: int = 1, deterministic: bool = True, verbose: int = 0):
        super().__init__(verbose)
        self.__previous_best_total_reward = float('-inf')
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

        self.__local_model = ModelManager.create_model(cfg=self.__cfg, env=self.__eval_env, 
                                                       device=self.__device, log_dir=self.__log_dir)

    def _on_step(self) -> bool:
        if self.locals["dones"][0]:
            self.__ep_completed_since_update += 1
        # interesting note: directly accessing 'self.model.ep_info_buffer' in this function will result in constant rewards each episode

        if self.__ep_completed_since_update >= self.__eval_freq:
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
            if mean_reward > self.__previous_best_total_reward:
                self.__previous_best_total_reward = mean_reward
                self.__save_new_best_model(mean_reward)

            self.__ep_completed_since_update = 0

        return True # always continue training        

    def __save_new_best_model(self, mean_reward):
        self.model.save(f"{self.__log_dir}/best_model/{self.__system_file_name}.zip")
        with open(f"{self.__log_dir}/best_model/{self.__system_file_name}_info.txt", 'w') as text_file:
            text_file.write(f"{self.__system_file_name}.zip had mean reward {mean_reward} over {self.__n_eval_episodes} " + \
                            f"evaluation episode(s).\nDeterminism was set to {self.__deterministic}.")
            
        if self.__cfg.with_wandb:
            wandb.save(f"{self.__log_dir}/best_model/{self.__wandb_file_name}.zip")
            wandb.save(f"{self.__log_dir}/best_model/{self.__wandb_file_name}_info.txt")