import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from starter_code.infrastructure.utils import from_np
from starter_code.interfaces.interfaces import CentralizedOutput, TransformOutput
from starter_code.rl_update.replay_buffer import StoredTransition
from starter_code.organism.organism import Organism
from starter_code.organism.transformations import LiteralActionTransformation
from starter_code.organism.domain_specific import preprocess_state_before_store

class BaseAgent(nn.Module):
    def __init__(self, networks, args):
        super(BaseAgent, self).__init__()
        self.args = args
        self.is_subpolicy = False  # default value
        self.bundle_networks(networks)

    def bundle_networks(self, networks):
        self.networks = networks

    def initialize_optimizer(self, lrs):
        self.optimizers = {}
        for name in lrs:
            self.optimizers[name] = optim.Adam(
                self.networks[name].parameters(), lr=lrs[name])

    def initialize_optimizer_schedulers(self, args):
        if not self.args.anneal_policy_lr: assert self.args.anneal_policy_lr_gamma == 1
        self.schedulers = {}
        for name, optimizer in self.optimizers.items():
            self.schedulers[name] = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=args.anneal_policy_lr_step,
                gamma=args.anneal_policy_lr_gamma,
                last_epoch=-1)

    def step_optimizer_schedulers(self, pfunc):
        for name in self.schedulers:
            before_lr = self.optimizers[name].state_dict()['param_groups'][0]['lr']
            self.schedulers[name].step()
            after_lr = self.optimizers[name].state_dict()['param_groups'][0]['lr']

    def get_state_dict(self):
        state_dict = dict()
        for name in self.networks:
            state_dict[name] = self.networks[name].state_dict()
        for name in self.optimizers:
            state_dict['{}_optimizer'.format(name)] = self.optimizers[name].state_dict()
        return state_dict

    def load_state_dict(self, agent_state_dict, reset_optimizer=True):
        for name in self.networks:
            self.networks[name].load_state_dict(agent_state_dict[name])
        if not reset_optimizer:
            for name in self.optimizers:
                self.optimizers[name].load_state_dict(
                    agent_state_dict['{}_optimizer'.format(name)])

class BaseRLAgent(BaseAgent, Organism):
    def __init__(self, networks, replay_buffer, args):
        BaseAgent.__init__(self, networks, args)
        self.replay_buffer = replay_buffer
        self.set_trainable(True)
        self.transformation_type = 'LiteralActionTransformation'

    def bundle_networks(self, networks):
        BaseAgent.bundle_networks(self, networks)
        self.policy = networks['policy']

    def forward(self, state, deterministic):
        with torch.no_grad():
            if isinstance(state, np.ndarray):
                state = from_np(state, 'cpu')
            action, dist_params = self.policy.select_action(state, deterministic)
            return CentralizedOutput(action=LiteralActionTransformation(action), dist_params=dist_params)

    def update(self, rl_alg):
        rl_alg.improve(self)

    def store_path(self, path):
        processed_path = []
        for step in path:
            step = preprocess_state_before_store(step)
            processed_path.append(
                StoredTransition(
                    state=step.state,
                    action=step.action,
                    next_state=step.next_state,
                    mask=step.mask,
                    reward=step.reward,
                    start_time=step.start_time,
                    end_time=step.end_time,
                    current_transformation_id=step.current_transformation_id,
                    next_transformation_id=step.next_transformation_id,
                    ))
        self.replay_buffer.add_path(processed_path)

    def clear_buffer(self):
        self.replay_buffer.clear_buffer()

    @property
    def discrete(self):
        return self.policy.discrete
    
    def set_trainable(self, trainable):
        self.trainable = trainable

    def can_be_updated(self):
        return self.trainable


class BaseHRLAgent(BaseRLAgent):
    def __init__(self, networks, transformations, replay_buffer, args):
        BaseRLAgent.__init__(self, networks, replay_buffer, args)
        self.transformations = self.assign_transformations(transformations)
        self.transformation_type = self.get_transformation_type(self.transformations)

    def assign_transformations(self, transformations):
        for t_id, transformation in transformations.items():
            transformation.set_transformation_registry(transformations)  # assign pointers
        return transformations

    def get_transformation_type(self, transformations):
        for i, transformation in enumerate(transformations.values()):
            if i > 0:
                assert transformation.__class__.__name__ == transformation_type
            else:
                transformation_type = transformation.__class__.__name__
        return transformation_type

    def forward(self, state, deterministic):
        with torch.no_grad():
            if isinstance(state, np.ndarray):
                state = from_np(state, 'cpu')
            action, dist_params = self.policy.select_action(state, deterministic)
            if self.policy.discrete:
                action = self.transformations[action]
            else:
                # must be a leaf policy
                action = LiteralActionTransformation(action)
            return CentralizedOutput(action=action, dist_params=dist_params)

    def update(self, rl_alg):
        BaseRLAgent.update(self, rl_alg)
        for t_id, transformation in self.transformations.items():
            if transformation.can_be_updated():
                transformation.update(rl_alg)

    def clear_buffer(self):
        BaseRLAgent.clear_buffer(self)
        for t_id, transformation in self.transformations.items():
            transformation.clear_buffer()

    def visualize_parameters(self, pfunc):
        BaseRLAgent.visualize_parameters(self, pfunc)
        for t_id, transformation in self.transformations.items():
            if transformation.trainable:
                pfunc('Parameters for {}-{}'.format(transformation.__class__.__name__, t_id))
                transformation.visualize_parameters(pfunc)