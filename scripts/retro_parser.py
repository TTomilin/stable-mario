import argparse

class RetroParser:
    def get_args():
        def arg(*args, **kwargs):
            parser.add_argument(*args, **kwargs)
        
        parser = argparse.ArgumentParser()
        
        arg("--device", default="cuda", type=str, choices=["cuda", "cpu"], help="Device to use")
        arg("--game", type=str, default="broom_zoom", help="Name of the game")
        arg("--render_mode", default="rgb_array", choices=["human", "rgb_array"], help="Render mode")
        arg("--load_state", type=str, default=None, help="Path to the game save state to load")
        arg("--record", default=True, action='store_true', help="Whether to record gameplay videos")
        arg("--record_every", type=int, default=150, help="Record gameplay video every n episodes")
        arg("--store_model", default=False, action='store_true', help="Whether to record gameplay videos")
        arg("--store_every", type=int, default=100, help="Save model every n episodes")
        arg("--skip_frames", default=True, action='store_true', help="Whether to skip frames")
        arg("--n_skip_frames", type=int, default=4, help="How many frames to skip")
        arg("--stack_frames", default=False, action='store_true', help="Whether to stack frames")
        arg("--n_stack_frames", type=int, default=4, help="How many frames to stack")
        arg("--show_observation", default=False, action='store_true', help="Show AI's observation.")
        arg("--normalize_reward", default=False, action='store_true', help="Normalize agent reward.")
        arg("--normalize_observation", default=True, action='store_true', help="Normalize agent observations.")
        arg("--resize_observation", default=True, action='store_true', help="Resize agent's observation to size specified in config.")
        arg("--rescale", default=False, action='store_true', help="Allow a modular transformation of the step and reset methods.")
        arg("--discretize", default=True, action='store_true', help="Limit agent's actions as specified in config.")
        arg("--learning_rate", type=float, default=0.00003, help="Set model's learning rate.")
        arg("--ent_coeff", type=float, default=0.05, help="Set entropy coefficient")
        arg("--timesteps", type=int, default=0, help="Number of timesteps the agent should train for.")
        arg("--model", type=str, default="PPO", help="The specific RL model to be used.")

        # WandB
        arg('--with_wandb', default=True, action='store_true', help='Enables Weights and Biases')
        arg('--wandb_entity', default='automated-play', type=str, help='WandB username (entity).')
        arg('--wandb_project', default='Mario', type=str, help='WandB "Project"')
        arg('--wandb_group', default=None, type=str, help='WandB "Group". Name of the env by default.')
        arg('--wandb_job_type', default=None, type=str, help='WandB job type')
        arg('--wandb_tags', default=[], type=str, nargs='*', help='Tags can help finding experiments')
        arg('--wandb_key', default=None, type=str, help='API key for authorizing WandB')

        args = parser.parse_args()

        return args