import wandb
import argparse
import os

class WandbManager():
    @staticmethod 
    def InitializeWandb(cfg: argparse.Namespace, log_dir: str, timestamp: str):
        if cfg.wandb_key:
            wandb.login(key=cfg.wandb_key)
        wandb_group = cfg.wandb_group if cfg.wandb_group is not None else cfg.game
        wandb_job_type = cfg.wandb_job_type if cfg.wandb_job_type is not None else "PPO"
        wandb_unique_id = f'{wandb_job_type}_{wandb_group}_{timestamp}'
        wandb.init(
            dir=log_dir,
            monitor_gym=True,
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            sync_tensorboard=True,
            id=wandb_unique_id,
            name=wandb_unique_id,
            group=wandb_group,
            job_type=wandb_job_type,
            tags=cfg.wandb_tags,
            resume="allow",
            settings=wandb.Settings(start_method='fork'),
            reinit=True
        )
        wandb.save("utilities/train_parser.py") # save default settings (could be useful for running model if defaults change)
        wandb.define_metric(name='eval/mean_reward', step_metric='global_step')
        wandb.define_metric(name='eval/mean_ep_length', step_metric='global_step')

    @staticmethod
    def ResumeWandb(cfg, log_dir, timestamp_original_run):
        if cfg.wandb_key:
            wandb.login(key=cfg.wandb_key)
        wandb_group = cfg.wandb_group if cfg.wandb_group is not None else cfg.game
        wandb_job_type = cfg.wandb_job_type if cfg.wandb_job_type is not None else "PPO"
        wandb_unique_id = f'{wandb_job_type}_{wandb_group}_{timestamp_original_run}'
        wandb.init(
            tensorboard=True,
            sync_tensorboard=True,
            dir=log_dir,
            monitor_gym=True,
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            id=wandb_unique_id,
            resume="must",
            settings=wandb.Settings(start_method='fork'),
            reinit=True
        )
        wandb.define_metric(name='eval/mean_reward', step_metric='global_step')
        wandb.define_metric(name='eval/mean_ep_length', step_metric='global_step')