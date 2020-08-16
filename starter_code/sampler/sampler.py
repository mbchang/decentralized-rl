from collections import defaultdict
import dill
import numpy as np
import time
import torch.multiprocessing as mp
mp.set_sharing_strategy('file_system')

from starter_code.infrastructure.log import renderfn
from starter_code.sampler.hierarchy_utils import flatten_rewards, build_interval_tree, set_transformation_ids, get_subreturns_matrix, redistribute_rewards_recursive, visualize_episode_data, visualize_hrl_finish_episode
from starter_code.interfaces.interfaces import StepOutput, PolicyTransformParams
from starter_code.organism.domain_specific import preprocess_state_before_forward

def collect_train_samples_serial(epoch, max_steps, objects, pid=0, queue=None):
    """
        Purpose: collect rollouts for max_steps steps
        Return: stats_collector
    """
    env = objects['env']
    stats_collector = objects['stats_collector_builder']()
    sampler = objects['sampler']
    max_episode_length = objects['max_episode_length']
    seed = int(1e6)*objects['seed'] + pid

    env.seed(seed)
    start = time.time()
    num_steps = 0
    while num_steps < max_steps:
        max_steps_this_episode = min(max_steps - num_steps, max_episode_length)
        episode_data = sampler.sample_episode(env=env, max_steps_this_episode=max_steps_this_episode)
        stats_collector.append(episode_data)
        num_steps += len(episode_data)  # this is actually the number of high level timesteps
    end = time.time()

    objects['printer']('PID: {} Time to collect samples: {}'.format(pid, end-start))
    if queue is not None:
        queue.put([pid, stats_collector.data])
    else:
        return stats_collector

def collect_train_samples_parallel(epoch, max_steps, objects, num_workers=10):
    """
        Purpose: collect rollouts for max_steps steps using num_workers workers
        Return: stats_collector
    """
    num_steps_per_worker = max_steps // num_workers
    num_residual_steps = max_steps - num_steps_per_worker * num_workers

    queue = mp.Manager().Queue()
    workers = []
    for i in range(num_workers):
        worker_steps = num_steps_per_worker + num_residual_steps if i == 0 else num_steps_per_worker
        worker_kwargs = dict(
            epoch=epoch,
            max_steps=worker_steps,
            objects=objects,
            pid=i+1,
            queue=queue)
        workers.append(mp.Process(target=collect_train_samples_serial, kwargs=worker_kwargs))
    for j, worker in enumerate(workers):
        worker.start()

    start = time.time()
    master_stats_collector = objects['stats_collector_builder']()
    for j, worker in enumerate(workers):
        worker_pid, worker_stats_data = queue.get()
        master_stats_collector.extend(worker_stats_data)
    end = time.time()
    objects['printer']('Time to extend master_stats_collector: {}'.format(end-start))

    for j, worker in enumerate(workers):
        worker.join()

    assert master_stats_collector.get_total_steps() == max_steps
    return master_stats_collector


def step_agent(env, organism, state, step_info_builder, transform_params):
    render = transform_params.render
    if render: frame = renderfn(env=env, scale=1)

    processed_state = preprocess_state_before_forward(state)
    organism_output = organism.forward(processed_state, deterministic=transform_params.deterministic) 

    transform_params = transform_params if organism_output.action.is_subpolicy else None
    transform_output = organism_output.action.transform(
        state=state,
        env=env, 
        transform_params=transform_params)

    step_info = step_info_builder(
        state=state,
        organism_output=organism_output,
        next_state=transform_output.next_state,
        info=transform_output.transform_node
        )

    if render:
        step_info.frame = frame
    step_info.mask = 0 if transform_output.done else 1
    step_output = StepOutput(
        done=transform_output.done, 
        step_info=step_info, 
        option_length=transform_output.transform_node.get_length())
    return transform_output.next_state, step_output


class Sampler():
    def __init__(self, organism, step_info, deterministic):
        self.organism = organism
        self.deterministic = deterministic
        self.step_info_builder = step_info

    def begin_episode(self, env):
        state = env.reset()
        return state

    def finish_episode(self, state, episode_data, env):
        # 1. flatten reward
        reward_chain = flatten_rewards(episode_data)

        # 2. identify the index of the start and end of its chain
        interval_tree = build_interval_tree(episode_data)

        # 3. Set the index of the agents for t and t+1
        set_transformation_ids(interval_tree)

        # 4. get subreturns matrix
        subreturns_matrix = get_subreturns_matrix(reward_chain, self.organism.args.gamma)

        if self.organism.args.hrl_verbose:
            visualize_hrl_finish_episode(episode_data, interval_tree, reward_chain, subreturns_matrix)

        # 5. re-distribute rewards
        redistribute_rewards_recursive(episode_data, subreturns_matrix)
        return episode_data

    def trim_step_infos(self, episode_data):
        for step in episode_data:
            if not step.hierarchy_info.leaf:
                setattr(step.hierarchy_info, 'organism', step.hierarchy_info.organism.id_num)
                self.trim_step_infos(step.hierarchy_info.path_data)
        return episode_data

    def get_bids_for_episode(self, episode_data):
        episode_bids = defaultdict(lambda: [])
        for step in episode_data:
            probs = step['action_dist']
            for index, prob in enumerate(probs):
                episode_bids[index].append(prob)
        return episode_bids

    def step_through_episode(self, state, env, max_steps_this_episode, render):
        episode_data = []
        global_clock = 0
        while global_clock < max_steps_this_episode:
            max_steps_this_option = max_steps_this_episode - global_clock
            state, step_output = step_agent(
                env=env, 
                organism=self.organism, 
                state=state,
                step_info_builder=self.step_info_builder, 
                transform_params=PolicyTransformParams(
                    max_steps_this_option=max_steps_this_option, 
                    deterministic=self.deterministic, 
                    render=render)
                )
            episode_data.append(step_output.step_info)
            if step_output.done: 
                break
            global_clock += step_output.option_length
        step_output.step_info.next_frame = renderfn(env=env, scale=1)  # render last frame
        if not step_output.done:
            assert global_clock == max_steps_this_episode
        return state, episode_data

    def sample_episode(self, env, max_steps_this_episode, render=False):
        state = self.begin_episode(env)
        state, episode_data = self.step_through_episode(
            state, env, max_steps_this_episode, render)
        episode_data = self.finish_episode(state, episode_data, env)
        episode_data = self.trim_step_infos(episode_data)
        if self.organism.args.hrl_verbose:
            visualize_episode_data(episode_data)
        return episode_data


