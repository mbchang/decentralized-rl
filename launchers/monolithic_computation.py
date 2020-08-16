import argparse

from launchers.hierarchical_monolithic import HierarchicalMonolithicLauncher
from launchers.parsers import build_parser
from mnist.mnist_env import get_affine_transforms
from starter_code.experiment.experiment import Experiment
from starter_code.infrastructure.log import MultiBaseLogger
from starter_code.learner.learners import CentralizedLearner
from starter_code.organism.learnable_transformations import FunctionTransformation
from starter_code.rl_update.replay_buffer import PathMemory
from starter_code.rl_update.rl_algs import rlalg_switch

def parse_args():
    parser = build_parser(transformation='function')
    args = parser.parse_args()
    args.hrl = False
    return args


class MonolithicComputationLauncher(HierarchicalMonolithicLauncher):

    @classmethod
    def create_transformation_builder(cls, state_dim, action_dim, args):
        modules = get_affine_transforms()
        assert args.num_primitives == len(modules)
        transformation_builder = lambda id_num: FunctionTransformation(
            id_num=id_num, networks=dict(function=modules[id_num]), args=args)
        return transformation_builder


if __name__ == '__main__':
    launcher = MonolithicComputationLauncher()
    launcher.main(parse_args)