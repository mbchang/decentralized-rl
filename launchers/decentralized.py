
import argparse
import numpy as np
import torch

from launchers.monolithic import BaseLauncher
from launchers.parsers import build_parser
from starter_code.organism.society import Society, BiddingPrimitive
from starter_code.experiment.experiments import TabularExperiment, DecentralizedExperiment
from starter_code.infrastructure.log import MultiBaseLogger
from starter_code.learner.learners import TabularDecentralizedLearner, DecentralizedLearner
from starter_code.modules.policies import SimpleBetaMeanPolicy, BetaMeanCNNPolicy
from starter_code.rl_update.replay_buffer import PathMemory
from starter_code.rl_update.rl_algs import rlalg_switch


def parse_args():
    parser = build_parser(auction=True)
    args = parser.parse_args()
    args.hrl = False
    return args


class DecentralizedLauncher(BaseLauncher):

    @classmethod
    def policy_switch(cls, state_dim, args):
        envtype = cls.env_registry.get_env_type(args.env_name[0])
        if envtype in ['mg', 'vcomp']:
            policy_name = 'cbeta'
        else:
            policy_name = 'beta'

        policy = dict(
            beta = lambda: SimpleBetaMeanPolicy(state_dim, args.hdim, 1),
            cbeta = lambda: BetaMeanCNNPolicy(state_dim, 1)
        )
        return policy[policy_name]

    @classmethod
    def experiment_switch(cls, env_name):
        envtype = cls.env_registry.get_env_type(env_name)

        if envtype == 'tab':
            learner = TabularDecentralizedLearner
            experiment = TabularExperiment
        elif envtype in ['gym', 'mg', 'vcomp']:
            learner = DecentralizedLearner
            experiment = DecentralizedExperiment
        else:
            assert False

        experiment_builder = lambda society, task_progression, rl_alg, logger, device, args: experiment(
            learner=learner(
                organism=society,
                rl_alg=rl_alg,
                logger=logger,
                device=device,
                args=args,
                ),
            task_progression=task_progression,
            logger=logger,
            args=args,
            )
        return experiment_builder

    @classmethod
    def create_organism(cls, device, task_progression, args):
        if not hasattr(args, 'num_primitives'):
            args.num_primitives = task_progression.action_dim

        if args.alg_name in ['ppo']:
            policy = cls.policy_switch(task_progression.state_dim, args)
            valuefn = cls.value_switch(task_progression.state_dim, args)
            networks = lambda: dict(
                        policy=policy(),
                        valuefn=valuefn()
                        )
            agent_builder = BiddingPrimitive
        else:
            assert False

        agents, unique_agents = cls.create_agents(
            agent_builder=agent_builder,
            networks=networks,
            task_progression=task_progression,
            redundancy=args.redundancy,
            args=args,
            device=device)

        organism = Society(
            agents=agents,
            unique_agents=unique_agents,
            device=device,
            args=args)

        return organism

    @classmethod
    def create_cloned_agents(cls, agent_builder, networks, transformation_builder, num_primitives, redundancy, args, device):
        agents = []
        unique_agents = []

        replay_buffer = lambda: PathMemory(
            max_replay_buffer_size=args.max_buffer_size*redundancy)
        for k in range(len(args.parents)):
            for i in range(num_primitives):
                # same networks
                agent_networks = networks()
                agent_replay_buffer = replay_buffer()
                transformation = transformation_builder(i)

                for j in range(redundancy):
                    # because winner % action_dim!
                    id_num = i + j*num_primitives + k*(num_primitives*redundancy)
                    agent = agent_builder(
                        id_num=id_num,
                        transformation=transformation,
                        networks=agent_networks,  # these are cloned!
                        replay_buffer = agent_replay_buffer,
                        args=args).to(device)
                    agents.append(agent)
                    if j == 0:
                        unique_agents.append(agent)
        return agents, unique_agents

    @classmethod
    def create_uncloned_agents(cls, agent_builder, networks, transformation_builder, num_primitives, redundancy, args, device):
        agents = []
        replay_buffer = lambda: PathMemory(max_replay_buffer_size=args.max_buffer_size)
        for k in range(len(args.parents)):
            for i in range(num_primitives):
                transformation = transformation_builder(i)  # the bidders are uncloned, but the transformations are
                for j in range(args.redundancy):
                    id_num = i + j*num_primitives + k*(num_primitives*redundancy)
                    agent = agent_builder(
                        id_num=id_num,
                        transformation=transformation,
                        networks=networks(),
                        replay_buffer = replay_buffer(),
                        args=args).to(device)
                    agents.append(agent)
        return agents, agents

    @classmethod
    def create_agents_generic(cls, agent_builder, networks, task_progression, num_primitives, redundancy, args, device):
        transformation_builder = cls.create_transformation_builder(
            state_dim=task_progression.state_dim,
            action_dim=task_progression.action_dim,
            args=args)

        if args.clone:
            agents, unique_agents = cls.create_cloned_agents(
                agent_builder=agent_builder,
                networks=networks,
                transformation_builder=transformation_builder,
                num_primitives=num_primitives,
                redundancy=redundancy,
                args=args,
                device=device)
        else:
            agents, unique_agents = cls.create_uncloned_agents(
                agent_builder=agent_builder,
                networks=networks,
                transformation_builder=transformation_builder,
                num_primitives=num_primitives,
                redundancy=redundancy,
                args=args,
                device=device)

        agents = sorted(agents, key=lambda x: x.id)
        unique_agents = sorted(unique_agents, key=lambda x: x.id)
        return agents, unique_agents

    @classmethod
    def create_agents(cls, agent_builder, networks, task_progression, redundancy, args, device):
        agents = cls.create_agents_generic(
            agent_builder=agent_builder,
            networks=networks,
            task_progression=task_progression,
            num_primitives=args.num_primitives,
            redundancy=redundancy,
            args=args,
            device=device)
        return agents

    @classmethod
    def main(cls, parse_args):
        args, device = cls.initialize(parse_args())
        logger = MultiBaseLogger(args=args)
        task_progression = cls.create_task_progression(logger, args)
        organism = cls.create_organism(device, task_progression, args)
        rl_alg = rlalg_switch(args.alg_name)(device=device, args=args)
        experiment_builder = cls.experiment_switch(args.env_name[0])
        experiment = experiment_builder(
            society=organism,
            task_progression=task_progression,
            rl_alg=rl_alg,
            logger=logger,
            device=device,
            args=args)
        experiment.main_loop(max_epochs=args.max_epochs)


if __name__ == '__main__':
    launcher = DecentralizedLauncher()
    launcher.main(parse_args)
