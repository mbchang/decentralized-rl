
import copy
import math
import numpy as np
import pickle
import sys
import torch

from collections import defaultdict
from starter_code.infrastructure.utils import AttrDict
from starter_code.rl_update.gae import estimate_advantages_by_episode
from starter_code.rl_update.replay_buffer import StoredTransition

eps = np.finfo(np.float32).eps.item()

def rlalg_switch(alg_name):
    rlalgs = {
        'ppo': PPO,
    }
    return rlalgs[alg_name]

class OnPolicyRLAlg():
    def __init__(self, device, max_buffer_size):
        self.device = device
        self.max_buffer_size = max_buffer_size
        self.num_samples_before_update = max_buffer_size

class PPO(OnPolicyRLAlg):
    def __init__(self, device, args):
        super(PPO, self).__init__(device=device, max_buffer_size=args.max_buffer_size)
        self.device = device
        self.args = args

        self.gamma = self.args.gamma
        self.entropy_coeff = args.entropy_coeff
        self.tau = 0.95
        self.l2_reg = 1e-3
        self.clip_epsilon = 0.2
        self.optim_epochs = 1
        self.optim_batch_size = args.optim_batch_size
        self.optim_value_iternum = 1

        self.reset_record()

    def record(self, minibatch_log, epoch, iter):
        self.log[epoch][iter] = minibatch_log

    def reset_record(self):
        self.log = defaultdict(dict)

    def assign_last_value(self, stacked_everything, values, path_lengths, agent):
        split_values = torch.split(values, path_lengths)
        zipped_paths = list(zip(*map(lambda x: torch.split(x, path_lengths), stacked_everything)))

        paths = []
        for episode_data, value in zip(zipped_paths, split_values):
            episode_data = AttrDict(StoredTransition(*episode_data)._asdict())
            assert (episode_data.state[1:]-episode_data.next_state[:-1]).norm() == 0
            episode_data.value = value
            last_state = episode_data.next_state[-1].unsqueeze(0)
            with torch.no_grad():
                if agent.is_subpolicy:
                    next_transformation = agent.get_transformation_by_id(
                        int(episode_data.next_transformation_id[-1].item()))
                    episode_data.last_value = next_transformation.valuefn(last_state)
                else:
                    episode_data.last_value = agent.valuefn(last_state)
            paths.append(episode_data)
        return paths

    def improve(self, agent):
        self.reset_record()
        """sample data"""
        paths, path_lengths = agent.replay_buffer.sample(self.device)
        with torch.no_grad():
            values = agent.valuefn(paths.state)
            action_dist = agent.policy.get_action_dist(paths.state)
            fixed_log_probs = agent.policy.get_log_prob(
                action_dist, paths.action)
        states = paths.state
        actions = paths.action

        """get advantage estimation from the trajectories"""
        paths = self.assign_last_value(paths, values, path_lengths, agent)
        advantages, returns = estimate_advantages_by_episode(
            paths, self.gamma, self.tau, self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std()+eps)

        """perform mini-batch PPO update"""
        optim_iter_num = int(math.ceil(advantages.shape[0] / self.optim_batch_size))

        for j in range(self.optim_epochs):
            perm = np.arange(advantages.shape[0])
            np.random.shuffle(perm)
            perm = torch.LongTensor(perm).to(self.device)

            states, actions, returns, advantages, fixed_log_probs = \
                states[perm].clone(), actions[perm].clone(), returns[perm].clone(), advantages[perm].clone(), fixed_log_probs[perm].clone()

            for i in range(optim_iter_num):
                ind = slice(i * self.optim_batch_size, min((i + 1) * self.optim_batch_size, states.shape[0]))
                states_b, actions_b, advantages_b, returns_b, fixed_log_probs_b = \
                    states[ind], actions[ind], advantages[ind], returns[ind], fixed_log_probs[ind]

                minibatch_log = self.ppo_step(agent, states_b, actions_b, returns_b, advantages_b, fixed_log_probs_b)
                self.record(minibatch_log=minibatch_log, epoch=j, iter=i)

        agent.replay_buffer.clear_buffer()

    def ppo_step(self, agent, states, actions, returns, advantages, fixed_log_probs):
        """update critic"""
        for _ in range(self.optim_value_iternum):
            values_pred = agent.valuefn(states)
            value_loss = (values_pred - returns).pow(2).mean()
            for param in agent.valuefn.parameters():
                value_loss += param.pow(2).sum() * self.l2_reg
            agent.optimizers['valuefn'].zero_grad()
            value_loss.backward()
            agent.optimizers['valuefn'].step()

        """update policy"""
        action_dist = agent.policy.get_action_dist(states)
        log_probs = agent.policy.get_log_prob(action_dist, actions)
        ratio = torch.exp(log_probs - fixed_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
        policy_surr = -torch.min(surr1, surr2).mean()
        entropy = agent.policy.get_entropy(action_dist).mean()
        policy_loss = policy_surr - self.entropy_coeff*entropy
        agent.optimizers['policy'].zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.policy.parameters(), 40)
        agent.optimizers['policy'].step()

        """log"""
        num_clipped = (surr1-surr2).nonzero().size(0)
        ratio_clipped = num_clipped / states.size(0)
        log = {}
        log['num_clipped'] = num_clipped
        log['ratio_clipped'] = ratio_clipped
        log['entropy'] = entropy
        log['bsize'] = states.size(0)
        log['value_loss'] = value_loss.item()
        log['policy_surr'] = policy_surr.item()
        log['policy_loss'] = policy_loss.item()
        return log
