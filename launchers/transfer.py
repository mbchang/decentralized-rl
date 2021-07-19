import argparse
import os
import torch
import yaml


import ujson
import pandas as pd
import pprint


from launchers.monolithic import BaseLauncher
from launchers.parsers import build_parser
from starter_code.infrastructure.configs import process_config, get_expid
from starter_code.infrastructure.log import MultiBaseLogger, env_manager_switch
from starter_code.infrastructure.multitask import construct_task_progression, task_prog_spec_multi
from starter_code.environment.env_config import EnvRegistry as ER
from starter_code.experiment.experiment import Experiment
from starter_code.experiment.experiments import TabularExperiment, TabularExperimentCentralized
from starter_code.learner.learners import CentralizedLearner
from starter_code.modules.policies import DiscretePolicy, IsotropicGaussianPolicy, DiscreteCNNPolicy
from starter_code.modules.value_function import SimpleValueFn, CNNValueFn
from starter_code.organism.agent import ActorCriticRLAgent
from starter_code.organism.transformations import LiteralActionTransformation
# from starter_code.organism.learnable_transformations import SubpolicyTransformation
from starter_code.rl_update.replay_buffer import PathMemory
from starter_code.rl_update.rl_algs import rlalg_switch




def parse_args():
    parser = build_parser(transfer=True)
    args = parser.parse_args()
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

class TransferLauncher(BaseLauncher):

    @classmethod
    def load_organism_weights(cls, organism, ckpt, pfunc):
        pfunc('Before loading pre-trained weights')
        organism.visualize_parameters(pfunc)
        organism.load_state_dict(ckpt['organism'])
        pfunc('After loading pre-trained weights')
        organism.visualize_parameters(pfunc)

        return organism

    @classmethod
    def load_checkpoints(cls, ckpt_infos, args, ckpt_type):
        ckpts = []
        for ckpt_info in ckpt_infos:
            ckpt_file_prefix = os.path.join('runs', args.subroot, ckpt_info[0][ckpt_type])
            seed_dict = get_seed_dirs(ckpt_file_prefix)
            seed = int(ckpt_info['seed']) if 'seed' in ckpt_info else args.seed
            # now getting the parent env
            seed_dir = os.path.join(
                ckpt_file_prefix,
                seed_dict[seed])
            ckpt_envs = ujson.load(open(os.path.join(seed_dir, 'code', 'params.json'), 'r'))['env_name']
            ckpt_env = ckpt_envs[0]
            ckpt_file_dir = os.path.join(
                ckpt_file_prefix,
                seed_dict[seed],
                'group_0',
                '{}_0_test'.format(ckpt_env),
                'checkpoints')
            ckpt_summary = pd.read_csv(os.path.join(ckpt_file_dir, 'summary.csv'))
            ckpt_mode = ckpt_info['mode'] if 'mode' in ckpt_info else 'best'
            ckpt_file = os.path.join(ckpt_file_dir, ckpt_summary[ckpt_mode].iloc[-1])#[-1])
            print('Loading in checkpoint: {}'.format(ckpt_file))
            ckpts.append(torch.load(ckpt_file, map_location=torch.device('cpu')))
        return ckpts


    @classmethod
    def update_dirs(cls, args, ckpt_subroot, ckpt_expname):
        args.expname = '{}__from__{}'.format(args.expname, get_expid(ckpt_expname))
        return args

    @classmethod
    def transfer_args(cls, args):
        args.parents = [args.ckpts]
        return args

    @classmethod
    def main(cls, parse_args):
        args, device = cls.initialize(parse_args())
        pprint.pprint(args.__dict__)
        ##########################################
        ckpt = cls.load_checkpoints([args.ckpts], args, ckpt_type='parent')[0]
        args = cls.update_dirs(args, ckpt['args'].subroot, ckpt['args'].expname)
        args = cls.transfer_args(args)
        ##########################################
        logger = MultiBaseLogger(args=args)
        task_progression = cls.create_task_progression(logger, args)
        organism = cls.create_organism(device, task_progression, args)
        ##########################################
        organism = cls.load_organism_weights(organism, ckpt, logger.printf)
        ##########################################
        rl_alg = rlalg_switch(args.alg_name)(device=device, args=args)
        envtype = cls.env_registry.get_env_type(args.env_name[0])
        args.envtype=envtype

        if envtype == 'tab':
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
    launcher = TransferLauncher()
    launcher.main(parse_args)
