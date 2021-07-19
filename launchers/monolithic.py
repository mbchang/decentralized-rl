import argparse
from collections import OrderedDict
import numpy as np
import os
import pandas as pd
import torch
import ujson

from launchers.parsers import build_parser
from starter_code.infrastructure.configs import process_config
from starter_code.infrastructure.log import MultiBaseLogger, env_manager_switch
from starter_code.infrastructure.multitask import construct_task_progression, task_prog_spec_multi
from starter_code.environment.env_config import EnvRegistry as ER
from starter_code.experiment.experiment import Experiment
from starter_code.experiment.experiments import TabularExperimentCentralized
from starter_code.learner.learners import CentralizedLearner
from starter_code.modules.policies import DiscretePolicy, IsotropicGaussianPolicy, DiscreteCNNPolicy
from starter_code.modules.value_function import SimpleValueFn, CNNValueFn
from starter_code.organism.agent import ActorCriticRLAgent
from starter_code.organism.transformations import LiteralActionTransformation
from starter_code.organism.learnable_transformations import SubpolicyTransformation
from starter_code.rl_update.replay_buffer import PathMemory
from starter_code.rl_update.rl_algs import rlalg_switch


def parse_args():
    parser = build_parser()
    args = parser.parse_args()
    args.hrl = False
    return args

def get_seed_dirs(seed_parent_folder):
    seed_dict = {}
    for seed_dir in os.listdir(seed_parent_folder):
        seed = int(seed_dir[len('seed'):seed_dir.find('__')])
        if seed in seed_dict:
            assert False, 'should not have duplicate seeds'
        else:
            seed_dict[seed] = seed_dir
    return seed_dict

class BaseLauncher:
    env_registry = ER()

    @classmethod
    def initialize(cls, args):
        args = process_config(args)
        if torch.cuda.is_available() and not args.cpu:
            device = torch.device('cuda', index=args.gpu_index) 
        else:
            device = torch.device('cpu')
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        return args, device

    @classmethod
    def create_task_progression(cls, logger, args):
        task_progression = construct_task_progression(
                task_prog_spec_multi(args.env_name),
                env_manager_switch(args.env_name[0], cls.env_registry),
                logger,
                cls.env_registry,
                args)
        return task_progression

    @classmethod
    def centralized_policy_switch(cls, state_dim, action_dim, args):
        envtype = cls.env_registry.get_env_type(args.env_name[0])
        if envtype in ['mg', 'vcomp']:
            policy_name = 'cnn'
        else:
            policy_name = 'mlp'
        policy = dict(
            mlp = lambda: DiscretePolicy(
                state_dim=state_dim, 
                hdim=args.hdim, 
                action_dim=action_dim),
            cnn = lambda: DiscreteCNNPolicy(
                state_dim=state_dim,
                action_dim=action_dim))
        return policy[policy_name]

    @classmethod
    def continuous_centralized_policy_switch(cls, state_dim, action_dim, args):
        envtype = cls.env_registry.get_env_type(args.env_name[0])  # first environment of the envs
        if envtype == 'gym':
            policy = lambda: IsotropicGaussianPolicy(
                state_dim=state_dim, 
                hdim=args.hdim, 
                action_dim=action_dim)
        else:
            assert False
        return policy

    @classmethod
    def value_switch(cls, state_dim, args):
        envtype = cls.env_registry.get_env_type(args.env_name[0])
        if envtype in ['mg', 'vcomp']:
            value_name = 'cnn'
        else:
            value_name = 'mlp'
        valuefn = dict(
            mlp = lambda: SimpleValueFn(state_dim, args.hdim),
            cnn = lambda: CNNValueFn(state_dim),
        )
        return valuefn[value_name]

    @classmethod
    def create_transformation_builder(cls, state_dim, action_dim, args):
        if args.hrl:
            transform_policy = cls.centralized_policy_switch(
                state_dim=state_dim,
                action_dim=action_dim,
                args=args)
            transform_valuefn = cls.value_switch(
                state_dim=state_dim,
                args=args)

            if args.shared_vfn:
                shared_transform_valuefn = transform_valuefn()
                transformation_builder = lambda id_num: SubpolicyTransformation(
                    id_num=id_num,
                    networks=dict(
                        policy=transform_policy(),
                        valuefn=shared_transform_valuefn,
                        ),
                    transformations = OrderedDict([(i, LiteralActionTransformation(id_num=i)) for i in range(action_dim)]),
                    replay_buffer=PathMemory(max_replay_buffer_size=args.max_buffer_size),
                    args=args
                    )
            else:
                transformation_builder = lambda id_num: SubpolicyTransformation(
                    id_num=id_num,
                    networks=dict(
                        policy=transform_policy(),
                        valuefn=transform_valuefn(),
                        ),
                    transformations = OrderedDict([(i, LiteralActionTransformation(id_num=i)) for i in range(action_dim)]),
                    replay_buffer=PathMemory(max_replay_buffer_size=args.max_buffer_size),
                    args=args
                    )
        else:
            transformation_builder = lambda id_num: LiteralActionTransformation(id_num)
        return transformation_builder

    @classmethod
    def create_organism(cls, device, task_progression, args):
        replay_buffer = PathMemory(max_replay_buffer_size=args.max_buffer_size)
        if args.alg_name == 'ppo':
            if task_progression.is_disc_action:
                policy = cls.centralized_policy_switch(
                    state_dim=task_progression.state_dim, 
                    action_dim=task_progression.action_dim, 
                    args=args)          
            else:
                policy = cls.continuous_centralized_policy_switch(
                    state_dim=task_progression.state_dim, 
                    action_dim=task_progression.action_dim, 
                    args=args)
            critic = cls.value_switch(task_progression.state_dim, args)

            agent = ActorCriticRLAgent(
                networks=dict(
                    policy=policy(),
                    valuefn=critic()),
                replay_buffer=replay_buffer, 
                args=args).to(device)
        else:
            assert False
        return agent

    @classmethod
    def load_checkpoints(cls, ckpt_infos, args, ckpt_type):
        ckpts = []
        for ckpt_info in ckpt_infos:
            ckpt_file_prefix = os.path.join('runs', args.subroot, ckpt_info[ckpt_type])
            seed_dict = get_seed_dirs(ckpt_file_prefix)
            seed = int(ckpt_info['seed']) if 'seed' in ckpt_info else args.seed
            # now get the parent env
            seed_dir = os.path.join(
                ckpt_file_prefix,
                seed_dict[seed])
            ckpt_envs = ujson.load(open(os.path.join(seed_dir, 'code', 'params.json'), 'r'))['env_name']
            assert len(ckpt_envs) == 1
            ckpt_env = ckpt_envs[0]
            ckpt_file_dir = os.path.join(
                ckpt_file_prefix,
                seed_dict[seed],
                'group_0',
                '{}_0_test'.format(ckpt_env),
                'checkpoints')
            ckpt_summary = pd.read_csv(os.path.join(ckpt_file_dir, 'summary.csv'))
            ckpt_mode = ckpt_info['mode'] if 'mode' in ckpt_info else 'best'
            ckpt_file = os.path.join(ckpt_file_dir, ckpt_summary[ckpt_mode].iloc[-1])
            print('Loading in checkpoint: {}'.format(ckpt_file))
            ckpts.append(torch.load(ckpt_file, map_location=torch.device('cpu')))
        return ckpts

    @classmethod
    def main(cls, parse_args):
        args, device = cls.initialize(parse_args())
        logger = MultiBaseLogger(args=args)
        task_progression = cls.create_task_progression(logger, args)
        organism = cls.create_organism(device, task_progression, args)
        rl_alg = rlalg_switch(args.alg_name)(device=device, args=args)
        envtype = cls.env_registry.get_env_type(args.env_name[0])
        args.envtype=envtype
        if envtype == 'tab':
            experiment = TabularExperimentCentralized(
            learner=CentralizedLearner(
                organism=organism,
                rl_alg=rl_alg,
                logger=logger,
                device=device,
                args=args,
                ),
            task_progression=task_progression,
            logger=logger,
            args=args)
        else:
            experiment = Experiment(
                learner=CentralizedLearner(
                    organism=organism,
                    rl_alg=rl_alg,
                    logger=logger,
                    device=device,
                    args=args,
                    ),
                task_progression=task_progression,
                logger=logger,
                args=args)
        experiment.main_loop(max_epochs=args.max_epochs)
        

if __name__ == '__main__':
    launcher = BaseLauncher()
    launcher.main(parse_args)
