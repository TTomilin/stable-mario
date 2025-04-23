from __future__ import annotations
from typing import List, Dict, Any

from signal_slot.signal_slot import EventLoop, EventLoopProcess

from sample_factory.algo.learning.learner_worker import init_learner_process
from sample_factory.utils.dicts import iterate_recursively
from collections import deque
import numpy as np
import wandb
from sample_factory.algo.runners.runner import Runner
from sample_factory.algo.sampling.sampler import ParallelSampler
from sample_factory.algo.utils.context import sf_global_context
from sample_factory.algo.utils.misc import ExperimentStatus
from sample_factory.algo.utils.multiprocessing_utils import get_mp_ctx
from sample_factory.utils.typing import StatusCode, PolicyID
from sample_factory.utils.utils import log
from sample_factory.mario.task_selectors import default_task_selector, enhanced_task_selector
from sample_factory.algo.utils.misc import EPISODIC
from scripts.config import CONFIG


class ParallelRunner(Runner):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.processes: List[EventLoopProcess] = []

    def init(self) -> StatusCode:
        status = super().init()
        if status != ExperimentStatus.SUCCESS:
            return status

        mp_ctx = get_mp_ctx(self.cfg.serial_mode)

        for policy_id in range(self.cfg.num_policies):
            batcher_event_loop = EventLoop("batcher_evt_loop")
            self.batchers[policy_id] = self._make_batcher(batcher_event_loop, policy_id)
            batcher_event_loop.owner = self.batchers[policy_id]

            learner_proc = EventLoopProcess(f"learner_proc{policy_id}", mp_ctx, init_func=init_learner_process)
            self.processes.append(learner_proc)

            self.learners[policy_id] = self._make_learner(
                learner_proc.event_loop,
                policy_id,
                self.batchers[policy_id],
            )
            learner_proc.event_loop.owner = self.learners[policy_id]
            learner_proc.set_init_func_args((sf_global_context(), self.learners[policy_id]))

        self.sampler = self._make_sampler(ParallelSampler, self.event_loop)

        self.connect_components()
        return status

    def _on_start(self):
        self._start_processes()
        super()._on_start()

    def _start_processes(self):
        log.debug("Starting all processes...")
        for p in self.processes:
            log.debug(f"Starting process {p.name}")
            p.start()
            self.event_loop.process_events()

    def _on_everything_stopped(self):
        for p in self.processes:
            log.debug(f"Waiting for process {p.name} to stop...")
            p.join()

        self.sampler.join()
        super()._on_everything_stopped()

class MultiParallelRunner(ParallelRunner):
    def __init__(self, cfg, task_selector=enhanced_task_selector):
        super().__init__(cfg)
        self.task_selector = task_selector
        self.task_properties = dict()
        for i, cfg_game in enumerate(cfg.game_list):
            game_dict = dict()
            game_dict['n_episodes'] = 0
            game_dict['total_reward'] = 0
            game_dict['training_progress'] = 0
            if cfg.reward_list != None:
                game_dict['target_reward'] = cfg.reward_list[i]
                if game_dict['target_reward'] <= 0:
                    game_dict['target_reward'] = 0.5
            else:
                game_dict['target_reward'] = 0.5
            game_dict['record'] = game_dict['target_reward']
            game_config = CONFIG[cfg_game]
            try:
                game_dict['start_reward'] = game_config['random_reward']
            except:
                print(f"{cfg_game} does not have a random reward in the config. Random reward is set to 0.")
                game_dict['start_reward'] = 0
            game = game_config["game_env"]
            self.task_properties[game] = game_dict
            if self.cfg.with_wandb and (wandb.run != None):
                wandb.define_metric(name=f"{game} Reward", step_metric="global_step")
                wandb.define_metric(name=f"{game} Runs", step_metric="global_step")
                wandb.define_metric(name=f"{game} Progress", step_metric="global_step")
        N = len(cfg.game_list)
        self.task_probabilities = dict()
        for key in self.task_properties:
            self.task_probabilities[key] = 1 / N # initialize to uniform random

    @staticmethod
    def _episodic_stats_handler(runner: MultiParallelRunner, msg: Dict, policy_id: PolicyID) -> None:
        # Do changing of state probabilities here
        s = msg[EPISODIC]
        for _, key, value in iterate_recursively(s):
            if isinstance(value, str): # in case the value is of type string...
                continue # then we don't want to include it in episodic stat reporting

            if key not in runner.policy_avg_stats:
                max_len = runner.cfg.heatmap_avg if key == 'heatmap' else runner.cfg.stats_avg
                runner.policy_avg_stats[key] = [
                    deque(maxlen=max_len) for _ in range(runner.cfg.num_policies)
                ]

            if isinstance(value, np.ndarray) and value.ndim > 0 and key != 'heatmap':
                if len(value) > runner.policy_avg_stats[key][policy_id].maxlen:
                    # increase maxlen to make sure we never ignore any stats from the environments
                    runner.policy_avg_stats[key][policy_id] = deque(maxlen=len(value))

                runner.policy_avg_stats[key][policy_id].extend(value)
            else:
                runner.policy_avg_stats[key][policy_id].append(value)
        
        episode_extra_stats = s['episode_extra_stats']
        game_dict = runner.task_properties[episode_extra_stats["completed_task"]]
        game_dict["n_episodes"] += 1
        game_dict["total_reward"] += episode_extra_stats["episode_reward"]
        if episode_extra_stats["episode_reward"] > game_dict["record"]:
            game_dict["record"] = episode_extra_stats["episode_reward"]
        game_dict["training_progress"] = 0.99 * game_dict["training_progress"] + 0.01 * (episode_extra_stats["episode_reward"]-game_dict["start_reward"])
        if runner.cfg.with_wandb:
            try:
                wandb.log({f"{episode_extra_stats['completed_task'][:-3]} Reward": episode_extra_stats["episode_reward"],f"{episode_extra_stats['completed_task'][:-3]} Runs": game_dict["n_episodes"],f"{episode_extra_stats['completed_task'][:-3]} Progress": game_dict["training_progress"]/(game_dict["record"]-game_dict["start_reward"])})
            except:
                pass
        task_probabilities = runner.task_selector(runner.task_properties, 
                                                  runner.cfg.random_task_probability, 
                                                  runner.cfg.episode_weight)
        runner.task_probabilities = task_probabilities

    def print_stats(self, fps, sample_throughput, total_env_steps):
        if hasattr(self, "task_probabilities"):
            log.debug("Task probabilities: %s", str(self.task_probabilities))
        if hasattr(self, "task_properties"):    
            log.debug("Task properties: %s", str(self.task_properties))
        return super().print_stats(fps, sample_throughput, total_env_steps)

    def _propagate_training_info(self):
        training_info: Dict[PolicyID, Dict[str, Any]] = dict()
        for policy_id in range(self.cfg.num_policies):
            training_info[policy_id] = dict({"task_probabilities": self.task_probabilities})
        self.update_training_info.emit(training_info)
