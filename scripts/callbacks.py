from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy

import gymnasium
import wandb
import statistics
import argparse

class CustomEvalCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, cfg: argparse.Namespace, eval_env: gymnasium.Env, save_path: str, system_file_name: str, 
                 wandb_file_name: str, eval_freq: int = 300, n_eval_episodes: int = 1, deterministic: bool = False, verbose: int = 0):
        super().__init__(verbose)
        self.__previous_best_total_reward = float('-inf')
        self.__n_eval_episodes = n_eval_episodes
        self.__eval_env = eval_env
        self.__save_path = save_path
        self.__system_file_name = system_file_name
        self.__wandb_file_name = wandb_file_name
        self.__cfg = cfg
        self.__eval_freq = eval_freq
        self.deterministic = deterministic
        self.__ep_completed_since_update = 0

    def _on_step(self) -> bool:
        if self.locals["dones"][0]:
            self.__ep_completed_since_update += 1
        # interesting note: directly accessing 'self.model.ep_info_buffer' in this function will result in constant rewards each episode

        if self.__ep_completed_since_update >= self.__eval_freq - 1:
            episode_rewards, _ = evaluate_policy(
                self.model,
                self.__eval_env,
                n_eval_episodes=self.__n_eval_episodes,
                return_episode_rewards=True,
                deterministic=self.deterministic
            ) # note: discarded output of evaluate_policy contains episode lengths.

            mean_reward = statistics.fmean(episode_rewards)
            if mean_reward > self.__previous_best_total_reward:
                self.__previous_best_total_reward = mean_reward
                self.__save_new_best_model(mean_reward)

        return True # always continue training        

    def __save_new_best_model(self, mean_reward):
        self.model.save(f"{self.__save_path}/{self.__system_file_name}.zip")
        with open(f"{self.__save_path}/{self.__system_file_name}_info.txt", 'w') as text_file:
            text_file.write(f"{self.__system_file_name}.zip had mean reward {mean_reward} over {self.__n_eval_episodes} " + \
                            f"evaluation episode(s).\nDeterminism was set to {self.deterministic}.")
            
        if self.__cfg.with_wandb:
            wandb.save(f"{self.__save_path}/{self.__wandb_file_name}.zip")
            wandb.save(f"{self.__save_path}/{self.__wandb_file_name}_info.txt")