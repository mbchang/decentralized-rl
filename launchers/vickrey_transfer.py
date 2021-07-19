import argparse
import itertools
import numpy as np
import os
import torch
import yaml
import ujson
import pandas as pd


# from agents import ActorCritic_BidAgent, SAC_BidAgent
# from auction import Vickrey_Auction, Bucket_Brigade, Credit_Conserving_Vickrey_Auction, Env_Reward
# from experiment import GridWorldExperiment, DecentralizedExperiment
# from decentralized_sampler import DecentralizedSampler

# from starter_code.configs import env_manager_switch, process_config, get_expid
# from starter_code.env_config import EnvRegistry
# from starter_code.log import MultiBaseLogger
# from starter_code.multitask import construct_task_progression, default_task_prog_spec
# from starter_code.policies import SimpleBetaSoftPlusPolicy, BetaCNNPolicy
# from starter_code.rb import StaticMemory, PathMemory
# from starter_code.rl_algs import rlalg_switch
# from starter_code.run import BaseLauncher
# from starter_code.transitions import AuctionStepInfo
# import starter_code.utils as u
# from starter_code.value_function import SimpleValueFn, CNNValueFn, CNNQFn

from launchers.monolithic import BaseLauncher
from launchers.decentralized import DecentralizedLauncher 
from starter_code.infrastructure.configs import process_config, get_expid
from launchers.parsers import build_parser
from starter_code.organism.society import Society, BiddingPrimitive
from starter_code.experiment.experiments import TabularExperiment, DecentralizedExperiment
from starter_code.infrastructure.log import MultiBaseLogger
from starter_code.learner.learners import TabularDecentralizedLearner, DecentralizedLearner
from starter_code.modules.policies import SimpleBetaMeanPolicy, BetaMeanCNNPolicy
from starter_code.rl_update.replay_buffer import PathMemory
from starter_code.rl_update.rl_algs import rlalg_switch
# from launchers.hierarchical_decentralized_pretrained_primitives import HierarchicalPretrainedDecentralizedLauncher
from launchers.parsers import build_parser
from starter_code.organism.transformations import LiteralActionTransformation
# from starter_code.organism.learnable_transformations import SubpolicyTransformation
from starter_code.infrastructure.configs import get_expid
from starter_code.infrastructure.log import MultiBaseLogger
from starter_code.infrastructure.utils import all_same
from starter_code.rl_update.replay_buffer import PathMemory
from starter_code.rl_update.rl_algs import rlalg_switch


# from starter_code.agent import LiteralAction_Agent, Subpolicy_Agent
# from starter_code.configs import get_expid
# from starter_code.log import MultiBaseLogger
# from starter_code.rb import PathMemory
# from starter_code.rl_algs import rlalg_switch
# import starter_code.utils as u



def parse_args():
    parser = build_parser(auction=True, transfer=True)
    args = parser.parse_args()
    args.hrl = False
    return args

# def parse_args():
#     parser = argparse.ArgumentParser(description='Decentralized Bucket Transfer')
#     parser.add_argument('--ckpts', nargs='+', type=yaml.load, required=True)
#     parser.add_argument('--subroot', type=str, required=True)
#     parser.add_argument('--env-name', nargs='+', type=str, default='GW2')
#     parser.add_argument('--seed', type=int, default=0)
#     parser.add_argument('--alg-name', type=str, default='ppo')
#     parser.add_argument('--printf', action='store_true')
#     parser.add_argument('--debug', action='store_true')
#     parser.add_argument('--cpu', action='store_true')
#     parser.add_argument('--expid', type=str, default='9999999')
#     parser.add_argument('--autorm', action='store_true')
#     parser.add_argument('--parallel_collect', action ='store_true')



#     # debugging
#     parser.add_argument('--gamma', type=float, default=0.99)
#     parser.add_argument('--max_buffer_size', type=int, default=4096)
#     parser.add_argument('--optim_batch_size', type=int, default=256)
#     parser.add_argument('--visualize_every', type=int, default=500)
#     parser.add_argument('--eval_every', type=int, default=50)

#     parser.add_argument('--eplencoeff', type=int, default=4)
#     parser.add_argument('--step_reward', type=float, default=0)
#     parser.add_argument('--memoryless', action='store_true')

#     parser.add_argument('--redundancy', type=int, default=2)
#     parser.add_argument('--clone', action='store_true',
#                         help='redundant agents are cloned')
#     parser.add_argument('--auctiontype', type=str, default='bb')
#     parser.add_argument('--policy', type=str, default='cbeta',
#                         help='beta | cbeta')
#     parser.add_argument('--ado', action='store_true',
#                         help='agent dropout')
#     parser.add_argument('--hdim', nargs='+', type=int, default=[16])
#     parser.add_argument('--plr', type=float, default=4e-5)
#     parser.add_argument('--vlr', type=float, default=5e-3)
#     parser.add_argument('--entropy_coeff', type=float, default=0)
#     parser.add_argument('--parallel-update', action='store_true')

#     ##########################################
#     parser.add_argument('--doublesampling', action='store_true',
#                         help='sample redundant agents twice')
#     ##########################################

#     # TODO: let's see how many of these things can be moved to default

#     args = parser.parse_args()
#     args.hrl = False
#     return args

def get_seed_dirs(seed_parent_folder):
    seed_dict = {}
    for seed_dir in os.listdir(seed_parent_folder):
        seed = int(seed_dir[len('seed'):seed_dir.find('__')])
        if seed in seed_dict:
            assert False, 'should not have duplicate seeds'
        else:
            seed_dict[seed] = seed_dir
    return seed_dict

class DecentralizedTransferLauncher(DecentralizedLauncher):

    @classmethod
    def load_organism_weights(cls, organism, ckpts, pfunc):
        """
        [
            [0_primitive0, 0_primitive1, 0_primitive0-clone, 0_primitive1-clone]
            [1_primitive0, 1_primitive1, 1_primitive0-clone, 1_primitive1-clone]
        ]
        to
        [0_primitive0, 0_primitive1, 0_primitive0-clone, 0_primitive1-clone, 1_primitive0, 1_primitive1, 1_primitive0-clone, 1_primitive1-clone]
        """
        pfunc('Before loading pre-trained weights')
        # u.visualize_params({'Primitive: {}'.format(a.id): a for a in organism.agents}, pfunc)
        organism.visualize_parameters(pfunc)
        society_state_dict = list(itertools.chain.from_iterable([c['organism'] for c in ckpts]))
        organism.load_state_dict(society_state_dict)
        pfunc('After loading pre-trained weights')
        # u.visualize_params({'Primitive: {}'.format(a.id): a for a in organism.agents}, pfunc)
        organism.visualize_parameters(pfunc)
        return organism

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
            # assert len(ckpt_envs) == 1
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
    def update_dirs(cls, args, ckpt_expnames):
        args.expname = '{}__from__{}'.format(args.expname, '__and__'.join(
            [str(get_expid(e)) for e in ckpt_expnames]))
        return args

    @classmethod
    def transfer_args(cls, args, ckpts):
        # assert all_same([c['args'].policy for c in ckpts])  # in general this need not be the same
        assert all_same([c['args'].auctiontype for c in ckpts])
        assert all_same([c['args'].ado for c in ckpts])
        # args.policy = ckpts[0]['args'].policy
        args.auctiontype = ckpts[0]['args'].auctiontype  # this should be the same I think
        args.ado = ckpts[0]['args'].ado  # this should be the same too
        args.parents = args.ckpts
        return args

    @classmethod
    def main(cls, parse_args):
        args, device = cls.initialize(parse_args())  # TODO: need to check which args are affected
        ##########################################
        ckpts = cls.load_checkpoints(args.ckpts, args, ckpt_type='parent')
        args = cls.update_dirs(args, [c['args'].expname for c in ckpts])  # GOOD

        # here assign the task parent here; perhaps have a task-tree object
        args = cls.transfer_args(args, ckpts)  # GOOD, but need to check what args we transfer
        ##########################################
        logger = MultiBaseLogger(args=args)
        task_progression = cls.create_task_progression(logger, args)
        organism = cls.create_organism(device, task_progression, args)
        ##########################################
        organism = cls.load_organism_weights(organism, ckpts, logger.printf)
        ##########################################
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
    launcher = DecentralizedTransferLauncher()
    launcher.main(parse_args)
