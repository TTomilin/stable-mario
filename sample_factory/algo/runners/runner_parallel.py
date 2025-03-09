from __future__ import annotations
from typing import List, Dict

from signal_slot.signal_slot import EventLoop, EventLoopProcess

from sample_factory.algo.learning.learner_worker import init_learner_process
from sample_factory.algo.runners.runner import Runner
from sample_factory.algo.sampling.sampler import ParallelSampler, ParallelMultiSampler
from sample_factory.algo.utils.context import sf_global_context
from sample_factory.algo.utils.misc import ExperimentStatus
from sample_factory.algo.utils.multiprocessing_utils import get_mp_ctx
from sample_factory.utils.typing import StatusCode, PolicyID
from sample_factory.utils.utils import log
from sample_factory.mario.task_selectors import default_task_selector
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
    def __init__(self, cfg, task_selector=default_task_selector):
        super().__init__(cfg)
        self.task_selector = default_task_selector
        self.task_properties = dict()
        for i, cfg_game in enumerate(cfg.game_list):
            game_dict = dict()
            game_dict['n_episodes'] = 0
            game_dict['total_reward'] = 0
            if cfg.target_reward_list != None:
                game_dict['target_reward'] = cfg.reward_list[i]
            game_config = CONFIG[cfg_game]
            game = game_config["game_env"]
            self.task_properties[game] = game_dict
    
    def init(self) -> StatusCode:
        status = Runner.init(self)
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

        self.sampler = self._make_sampler(ParallelMultiSampler, self.event_loop)

        self.connect_components()
        return status

    @staticmethod
    def _episodic_stats_handler(runner: MultiParallelRunner, msg: Dict, policy_id: PolicyID) -> None:
        super()._episodic_stats_handler(runner, msg, policy_id)
        s = msg[EPISODIC]
        episode_extra_stats = s['episode_extra_stats']
        game_dict = runner.task_properties[episode_extra_stats["completed_task"]]
        game_dict["n_episodes"] += 1
        game_dict["total_reward"] += episode_extra_stats["episode_reward"]

        task_probabilities = runner.task_selector(runner.task_properties, 
                                                  runner.cfg.random_task_probability, 
                                                  runner.cfg.episode_weight)
        runner.sampler.update_task_probabilities(task_probabilities)
