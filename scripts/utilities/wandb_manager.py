import wandb
import argparse

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
            resume=False,
            settings=wandb.Settings(start_method='fork'),
            reinit=True
        )
        wandb.define_metric(name='eval/mean_reward', step_metric='global_step')
        wandb.define_metric(name='eval/mean_ep_length', step_metric='global_step')