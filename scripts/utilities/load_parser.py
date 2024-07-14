import argparse
from utilities.parser import BaseParser

class LoadParser(BaseParser):
    def set_args(self):                
        self.arg("--directory", type=str, default=None, help="Directory from which to load model. Must contain model zip and textfile with train command.")
        self.arg("--record", default=False, action='store_true', help="Whether to record the trained model. Will disable human rendering.")
        self.arg("--record_every", type=int, default=1, help="Record trained model every n episodes")
        self.arg("--with_wandb", default=False, action='store_true', help="Whether to sync logs to wandb.")
        self.arg("--log_reward_summary", default=False, action='store_true', help="Whether to log the min, mean, max reward each n episodes.")
        self.arg('--log_reward_summary_frequency', type=int, default=5, help="Frequency with which to log the smallest episode reward, measured in episodes.")
