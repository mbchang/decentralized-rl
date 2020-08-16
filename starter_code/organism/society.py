from collections import OrderedDict
from operator import itemgetter
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn

from starter_code.infrastructure.utils import from_np
from starter_code.interfaces.interfaces import DecentralizedOutput
from starter_code.rl_update.replay_buffer import StoredTransition
from starter_code.organism.agent import ActorCriticRLAgent
from starter_code.organism.domain_specific import preprocess_state_before_store
from starter_code.organism.organism import Organism
from starter_code.organism.utilities import vickrey_utilities, credit_conserving_vickrey_utilities, bucket_brigade_utilities, environment_reward_utilities


def agent_update(agent, rl_alg):
    agent.update(rl_alg)

class Society(nn.Module, Organism):
    def __init__(self, agents, unique_agents, device, args):
        super(Society, self).__init__()
        self.agents = nn.ModuleList(agents)
        self.unique_agents = unique_agents  # just a set of pointers
        self.agents_by_id = {agent.id: agent for agent in self.agents}  # this is the registry

        self.device = device
        self.args = args
        self.bootstrap = True
        self.discrete = True
        self.ado = args.ado

        self.transformations = self.assign_transformations()
        self.transformation_type = self.get_transformation_type(self.transformations)

        self.set_trainable(True)

    def assign_transformations(self):
        transformations = OrderedDict()
        for a_id in self.agents_by_id:
            transformations[a_id] = self.agents_by_id[a_id].transformation
        for agent in self.agents:
            agent.transformation.set_transformation_registry(transformations)
        return transformations

    def get_transformation_type(self, transformations):
        for i, transformation in enumerate(transformations.values()):
            if i > 0:
                assert transformation.__class__.__name__ == transformation_type
            else:
                transformation_type = transformation.__class__.__name__
        return transformation_type

    def set_trainable(self, trainable):
        self.trainable = trainable

    def can_be_updated(self):
        return self.trainable

    def agent_dropout(self):
        total_agents = len(self.agents)
        num_inactive_agents = np.random.randint(low=0, high=total_agents-1)
        inactive_agent_ids = np.random.choice(a=range(total_agents), size=num_inactive_agents, replace=False)
        self._set_inactive_agents(inactive_agent_ids)

    def get_state_dict(self):
        return [a.get_state_dict() for a in self.agents]

    def load_state_dict(self, society_state_dict):
        for agent, agent_state_dict in zip(self.agents, society_state_dict):
            agent.load_state_dict(agent_state_dict)

    def _set_inactive_agents(self, agent_ids):
        for agent in self.agents:
            agent.active = True

        for agent_id in agent_ids:
            assert agent_id == self.agents[agent_id].id
            self.agents[agent_id].active = False

    def get_active_agents(self):
        active_agents = []
        for agent in self.agents:
            if agent.active:
                active_agents.append(agent)
        return active_agents

    def _run_auction(self, state, deterministic):
        if self.args.clone:
            with torch.no_grad():
                dists = OrderedDict((a.id, a.policy.get_action_dist(state)) for a in self.unique_agents)
            bids = OrderedDict()
            for i in range(self.args.redundancy):
                for a in self.unique_agents:
                    bid = dists[a.id].sample().item()
                    bids[a.id + i*len(self.unique_agents)] = bid
        else:
            bids = OrderedDict([(a.id, a(state, deterministic=deterministic).item()) for a in self.get_active_agents()])
        return bids

    def compute_utilities(self, utility_args):
        utility_function = dict(
            v=vickrey_utilities,
            bb=bucket_brigade_utilities,
            ccv=credit_conserving_vickrey_utilities,
            env=environment_reward_utilities)[self.args.auctiontype]
        utilities = utility_function(utility_args, self.args)
        return utilities

    def _choose_winner(self, bids):
        winner = max(bids.items(), key=itemgetter(1))[0]
        return winner

    def _select_action(self, winner):
        return self.agents_by_id[winner].transformation

    def _get_learnable_active_agents(self):
        learnable_active_agents = [a for a in self.unique_agents if a.learnable and len(a.replay_buffer) > 0]
        return learnable_active_agents

    def step_optimizer_schedulers(self, pfunc):
        for agent in self.agents:
            agent.step_optimizer_schedulers(pfunc)

    def flail(self, env):
        bids = OrderedDict([(a.id, a.flail()) for a in self.agents])
        return self._produce_output(bids)

    def forward(self, state, deterministic):
        with torch.no_grad():
            if isinstance(state, np.ndarray):
                state = from_np(state, 'cpu')
            bids = self._run_auction(state, deterministic=deterministic)
        winner = self._choose_winner(bids)
        action = self._select_action(winner)
        return DecentralizedOutput(action=action, winner=winner, bids=bids)

    def update(self, rl_alg):
        learnable_active_agents = self._get_learnable_active_agents()

        if self.args.parallel_update:
            processes = []
            for agent in learnable_active_agents:
                p = mp.Process(target=agent_update, args=(agent, rl_alg))
                p.start()
                processes.append(p)
            for p in processes:
                p.join()
        else:
            for agent in learnable_active_agents:
                agent.update(rl_alg)

    def store_path(self, path):
        processed_path = {a_id: [] for a_id in path[0].bids}  # path[0] is hacky
        for step in path:
            for a_id in step.bids:
                if self.args.memoryless:
                    mask = 0
                else:
                    mask = step.mask

                step = preprocess_state_before_store(step)
                processed_path[a_id].append(
                    StoredTransition(
                        state=step.state,
                        action=np.array([step.bids[a_id]]),
                        next_state=step.next_state,
                        mask=mask,
                        reward=step.payoffs[a_id],
                        start_time=step.start_time,
                        end_time=step.end_time,
                        current_transformation_id=step.current_transformation_id,
                        next_transformation_id=step.next_transformation_id,
                        )
                    )
        for a_id in processed_path:
            self.agents_by_id[a_id].replay_buffer.add_path(processed_path[a_id])

    def clear_buffer(self):
        for a_id, agent in enumerate(self.agents):
            agent.clear_buffer()

    def visualize_parameters(self, pfunc):
        for agent in self.agents:
            pfunc('Primitive: {}'.format(agent.id))
            agent.visualize_parameters(pfunc)


class BiddingPrimitive(ActorCriticRLAgent):
    def __init__(self, id_num, transformation, networks, replay_buffer, args):
        ActorCriticRLAgent.__init__(self, networks, replay_buffer, args)
        self.id = id_num
        self.transformation = transformation
        self.learnable = True
        self._active = True
        self.is_subpolicy = False

    @property
    def active(self):
        return self._active

    @active.setter
    def active(self, value):
        self._active = value

    def forward(self, state, deterministic):
        with torch.no_grad():
            bid, dist = self.policy.select_action(state, deterministic)
        return bid

    def flail(self):
        bid = np.random.uniform()  # note that the range is [0, 1]
        return bid

    def update(self, rl_alg):
        rl_alg.improve(self)
        if self.transformation.can_be_updated():
            self.transformation.update(rl_alg)

    def clear_buffer(self):
        self.replay_buffer.clear_buffer()
        self.transformation.clear_buffer()
