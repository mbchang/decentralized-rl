from collections import OrderedDict

from launchers.hierarchical_monolithic_pretrained_primitives import HierarchicalPretrainedMonolithicLauncher
from launchers.parsers import build_parser
from starter_code.organism.transformations import LiteralActionTransformation
from starter_code.organism.learnable_transformations import SubpolicyTransformation
from starter_code.infrastructure.configs import get_expid
from starter_code.experiment.experiment import Experiment
from starter_code.infrastructure.log import MultiBaseLogger
from starter_code.learner.learners import CentralizedLearner
from starter_code.rl_update.replay_buffer import PathMemory
from starter_code.rl_update.rl_algs import rlalg_switch


def parse_args():
    parser = build_parser(transformation='subpolicy', pretrain_primitives=True, transfer=True)
    args = parser.parse_args()
    args.hrl = True
    return args


class HierarchicalPretrainedMonolithicTransferLauncher(HierarchicalPretrainedMonolithicLauncher):

    @classmethod
    def load_organism_weights(cls, organism, ckpts, pfunc):
        assert len(ckpts) == 1
        ckpt = ckpts[0]
        pfunc('\nBefore loading organism weights\n')
        organism.visualize_parameters(pfunc)
        organism.load_state_dict(ckpt['organism'])
        pfunc('\nAfter loading organism weights\n')
        organism.visualize_parameters(pfunc)
        return organism

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
            # no shared value function because we have pre-trained primitives.
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
    def update_dirs_parent(cls, args, ckpt_expnames):
        args.expname = '{}__from__{}'.format(args.expname, '__and__'.join(
            [str(get_expid(e)) for e in ckpt_expnames]))
        return args

    @classmethod
    def transfer_args(cls, args, ckpts):
        args.parents = args.ckpts
        return args

    @classmethod
    def main(cls, parse_args):
        args, device = cls.initialize(parse_args())

        ##########################################
        # update args for pre-trained parent
        # ****************************************
        parent_ckpts = cls.load_checkpoints(args.ckpts, args, ckpt_type='parent')  
        args = cls.update_dirs_parent(args, [c['args'].expname for c in parent_ckpts])
        args = cls.transfer_args(args, parent_ckpts)
        # ****************************************ss
        # update args for pre-trained primitives
        # ****************************************
        primitive_ckpts = cls.load_checkpoints(args.primitives, args, ckpt_type='primitive')
        args = cls.update_dirs_primitive(args, [c['args'].expname for c in primitive_ckpts])
        args.num_primitives = len(primitive_ckpts)
        ##########################################
        logger = MultiBaseLogger(args=args)
        task_progression = cls.create_task_progression(logger, args)
        organism = cls.create_organism(device, task_progression, args)
        ##########################################
        # load weights for pre-trained primitives
        # ****************************************
        organism = cls.load_primitive_weights(organism, primitive_ckpts, logger.printf)
        # ****************************************
        # load weights for parents
        # ****************************************
        organism = cls.load_organism_weights(organism, parent_ckpts, logger.printf)
        ##########################################

        rl_alg = rlalg_switch(args.alg_name)(device=device, args=args)
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
    launcher = HierarchicalPretrainedMonolithicTransferLauncher()
    launcher.main(parse_args)
