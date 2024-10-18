from colorist import Color
import argparse

from utilities.parser import BaseParser

class TrainParser(BaseParser):

    @staticmethod
    def validate_args(cfg: argparse.Namespace):
        if [cfg.pi, cfg.vf].count(None) == 1:
            raise ValueError(f"{Color.RED}Invalid input. Specify either both cfg.pi and cfg.vf or neither. Aborting...{Color.OFF}")

    def set_args(self):                
        self.arg("--device", default="cuda", type=str, choices=["cuda", "cpu"], help="Device to use")
        self.arg("--game", type=str, default="broom_zoom", help="Name of the game")
        self.arg("--render_mode", default="rgb_array", choices=["human", "rgb_array"], help="Render mode")
        self.arg("--load_state", type=str, default=None, help="Path to the game save state to load")
        self.arg("--record", default=False, action='store_true', help="Whether to record gameplay videos")
        self.arg("--record_every", type=int, default=150, help="Record gameplay video every n episodes")
        self.arg("--store_every", type=int, default=100, help="Save model every n episodes")
        self.arg("--skip_frames", default=False, action='store_true', help="Whether to skip frames")
        self.arg("--n_skip_frames", type=int, default=4, help="How many frames to skip")
        self.arg("--stack_frames", default=False, action='store_true', help="Whether to stack frames")
        self.arg("--n_stack_frames", type=int, default=4, help="How many frames to stack")
        self.arg("--show_observation", default=False, action='store_true', help="Show AI's observation.")
        self.arg("--normalize_reward", default=False, action='store_true', help="Normalize agent reward.")
        self.arg("--normalize_observation", default=False, action='store_true', help="Normalize agent observations.")
        self.arg("--resize_observation", default=True, action='store_true', help="Resize agent's observation to size specified in config.")
        self.arg("--rescale", default=False, action='store_true', help="Allow a modular transformation of the step and reset methods.")
        self.arg("--discretize", default=True, action='store_true', help="Limit agent's actions as specified in config.")
        self.arg("--learning_rate", type=float, default=0.00003, help="Set model's learning rate.")
        self.arg("--ent_coeff", type=float, default=0.05, help="Set entropy coefficient. Defaults to 0.05.")
        self.arg("--timesteps", type=int, default=0, help="Number of timesteps the agent should train for.")
        self.arg("--model", type=str, default="PPO", help="The specific RL model to be used.")
        self.arg("--time_limit", type=int, default=None, help="Max number of seconds agent is allowed to take before episode is truncated")
        self.arg("--crop", default=False, action='store_true', help="Crop the agent's observations (defaults to leaving 80x80) pixels at center")
        self.arg("--crop_dimension", type=str, default="256x256", help="The rectangular dimension of the center crop to be applied (e.g. 64x64).")
        self.arg("--log_step_rewards", action='store_true', help="Records step rewards in a textfile found at the root of the log directory")
        self.arg("--batch_norm", action="store_true", help="Normalizes inputs over each batch. Only available for QRDQN and DQN.")
        self.arg("--pi", type=str, default=None, help="Comma-separated numbers of units per HIDDEN layer of the model's actor. PPO default is 256,256")
        self.arg("--vf", type=str, default=None, help="Comma-separated numbers of units per HIDDEN layer of the model's critic. PPO default is 256,256")
        self.arg("--gray_scale", action="store_true", help="transf orms model's observations to grayscale.")
        self.arg("--save_best", action="store_true", default=False, help="Evaluates model each eval_freq episodes and stores best performing one.")
        self.arg("--eval_freq", type=int, default=100, help="How often to measure and potentially save the model, measured in number of episodes.")
        self.arg("--n_epochs", type=int, default=10, help="number of epochs per rollout (update)")
        self.arg("--n_steps", type=int, default=2048, help="number of timesteps per epoch")
        self.arg("--batch_size", type=int, default=64, help="size of the mini batches in which the rollout buffers are processed")
        self.arg("--features_extractor", type=str, default="NatureCNN", help="The type of feature extractor used by the model. Defaults to NatureCNN.")
        self.arg("--features_extractor_dim", type=int, default=None, help="Number of features to be extracted by the features extractor class.")
        self.arg("--activation_function", type=str, default="tanh", help="Which activation function to include between hidden layers, defaults to tanh.")
        self.arg("--eval_metric", type=str, default=None, help="Which RAM-value (name as specified in data.json) to use as evaluation metric for deciding on best model. Uses value in RAM at end of episode as metric. If none specified, the reward is used.")
        self.arg("--on_the_spot_wrapper", action="store_true", default=False, help="Wrapper designed to train on_the_spot.")
        self.arg("--on_the_spot_hack", action="store_true", default=False, help="Wrapper designed to train on_the_spot.")
        self.arg("--grabbit", type=str, default=None, help="Wrapper designed to train grabbit.")
        self.arg("--delay", action="store_true", default=False, help="Slow down model to more clearly see the actions.")
        self.arg("--delay_time", type=int, default=500, help="Amount of time to delay each timestep if --delay is used, measured in ms.")
        self.arg("--filter_colors", type=str, default=None, help="Comma seperated list of colors to see. i.e. '4H3,4JD' corresponds to only seeing the colors (4, 17, 3) and (4, 19, 13) in 15-bit RGB")
        # WandB
        self.arg('--with_wandb', default=False, action='store_true', help='Enables Weights and Biases')
        self.arg('--wandb_entity', default='automated-play', type=str, help='WandB username (entity).')
        self.arg('--wandb_project', default='Mario', type=str, help='WandB "Project"')
        self.arg('--wandb_group', default=None, type=str, help='WandB "Group". Name of the env by default.')
        self.arg('--wandb_job_type', default=None, type=str, help='WandB job type')
        self.arg('--wandb_tags', default=[], type=str, nargs='*', help='Tags can help finding experiments')
        self.arg('--wandb_key', default=None, type=str, help='API key for authorizing WandB')
        self.arg('--log_variance', default=False, action='store_true', help="Whether or not to log the variance in the model's reward each n episodes.")
        self.arg('--variance_log_frequency', type=int, default=5, help="Frequency with which to log variance in episode rewards, measured in episodes.")
        self.arg('--log_reward_summary', default=False, action='store_true', help="Whether or not to log the smallest episode reward each n episodes.")
        self.arg('--log_reward_summary_frequency', type=int, default=5, help="Frequency with which to log the smallest episode reward, measured in episodes.")