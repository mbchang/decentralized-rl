from collections import OrderedDict
import numpy as np
import time
import torch

from starter_code.organism.base_agent import BaseAgent
from starter_code.sampler.sampler import collect_train_samples_serial, collect_train_samples_parallel
from starter_code.sampler.hierarchy_utils import is_hierarchical

def distribute_paths(organism, episode_datas):
    for episode_data in episode_datas:
        distribute_paths_recursive(organism, episode_data)


def distribute_paths_recursive(organism, episode_data):
    if organism.trainable:
        organism.store_path(episode_data)
    for step in episode_data:
        if not step.hierarchy_info.leaf:
            sub_organism = organism.transformations[step.hierarchy_info.organism]
            distribute_paths_recursive(
                organism=sub_organism,
                episode_data=step.hierarchy_info.path_data)


def low_level_visualize(organism, episode_data):
    all_returns = 0
    all_frames = []
    all_bids=[]
    for step in episode_data:
        if not step.hierarchy_info.leaf:
            sub_organism = organism.transformations[step.hierarchy_info.organism]
            frames_lower, bids_lower, return_lower= low_level_visualize(sub_organism, step.hierarchy_info.path_data)
            all_frames.extend(frames_lower)
            all_bids.extend(bids_lower)
            all_returns += sum([e.reward for e in episode_data])
        else:
            all_frames.append(step.frame)
            all_bids.append(step.action_dist)
    return all_frames, all_bids, all_returns


class Learner():
    def __init__(self, organism, rl_alg, logger, device, args):
        self.organism = organism
        self.rl_alg = rl_alg
        self.logger = logger
        self.device = device
        self.args = args
        self.parallel_collect = args.parallel_collect
        self.logger.printf(self.organism)

        self.steps = 0
        self.min_return = np.inf
        self.max_return = -np.inf

    def collect_samples(self, epoch, env_manager):
        ######################################################
        # initialize
        ######################################################
        self.logger.printf('Collecting Samples...')
        t0 = time.time()
        collector = collect_train_samples_parallel if self.parallel_collect else collect_train_samples_serial

        ######################################################
        # collect
        ######################################################
        self.organism.to('cpu')
        stats_collector = collector(
            epoch=epoch,
            max_steps=self.rl_alg.num_samples_before_update,
            objects=dict(
                max_episode_length=env_manager.max_episode_length,
                env=env_manager.env,
                stats_collector_builder=self.stats_collector_builder,
                sampler=self.sampler,
                organism=self.organism,
                seed=self.args.seed,
                printer=self.logger.printf,
                ),
            )
        self.organism.to(self.device)

        ######################################################
        # store into replay buffer
        ######################################################
        stats = stats_collector.bundle_batch_stats()
        distribute_paths(self.organism, stats_collector.data['episode_datas'])

        ######################################################
        # update metrics
        ######################################################
        self.steps += stats['total_steps']
        if stats['min_return'] < self.min_return:
            self.min_return = stats['min_return']
        if stats['max_return'] > self.max_return:
            self.max_return = stats['max_return']

        self.logger.printf('Epoch {}: Time to Collect Samples: {}'.format(epoch, time.time()-t0))
        return stats

    def update(self, epoch):
        if self.args.param_verbose:
            print('Before')
            self.organism.visualize_parameters(self.logger.printf)

        t0 = time.time()
        self.organism.update(self.rl_alg)
        self.logger.printf('Epoch {}: Time to Update: {}'.format(epoch, time.time()-t0))
        self.organism.clear_buffer()
        torch.cuda.empty_cache()

        if epoch >= self.args.anneal_policy_lr_after:
            self.organism.step_optimizer_schedulers(self.logger.printf)

        if self.args.param_verbose:
            print('After')
            self.organism.visualize_parameters(self.logger.printf)

    def test(self, epoch, env_manager, num_test):
        stats_collector = self.stats_collector_builder()
        for i in range(num_test):
            with torch.no_grad():
                env_manager.env.seed(int(1e8)*self.args.seed+epoch)
                episode_data = self.sampler.sample_episode(
                    env=env_manager.env,
                    max_steps_this_episode=env_manager.max_episode_length,
                    render=True)
                if epoch % self.args.visualize_every == 0:
                    if i == 0 and self.organism.discrete:
                        self.get_qualitative_output(
                            env_manager, self.sampler, episode_data, epoch, i)
            stats_collector.append(episode_data)
        stats = stats_collector.bundle_batch_stats()
        return stats

    def get_qualitative_output(self, env_manager, sampler, episode_data, epoch, i):
        if env_manager.visual:
            bids = sampler.get_bids_for_episode(episode_data)
            returns = sum([e.reward for e in episode_data])
            frames = [e.frame for e in episode_data]

            # last frame
            frames.append(episode_data[-1].next_frame)
            for agent_id, agent_bids in bids.items():
                agent_bids.append(0)
            env_manager.save_video(epoch, i, bids, returns, frames)

            if is_hierarchical(episode_data):
                frames_low_level, bids_low_level, returns_low_level = low_level_visualize(self.organism, episode_data)
                bids_dict={}
                for bidlst in bids_low_level:
                    for j, bid in enumerate(bidlst):
                        if j not in bids_dict:
                            bids_dict[j]=[]
                        bids_dict[j].append(bid)
                env_manager.save_video(epoch, i , bids_dict, returns_low_level, frames_low_level, '_low_level')
