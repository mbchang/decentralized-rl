from collections import OrderedDict

from launchers.parsers import build_parser
from launchers.decentralized import DecentralizedLauncher
from starter_code.organism.transformations import LiteralActionTransformation
from starter_code.organism.learnable_transformations import SubpolicyTransformation
from starter_code.infrastructure.configs import get_expid
from starter_code.infrastructure.log import MultiBaseLogger
from starter_code.rl_update.replay_buffer import PathMemory
from starter_code.rl_update.rl_algs import rlalg_switch


def parse_args():
    parser = build_parser(auction=True, transformation='subpolicy', pretrain_primitives=True)
    args = parser.parse_args()
    args.hrl = True
    return args

class HierarchicalPretrainedDecentralizedLauncher(DecentralizedLauncher):

    @classmethod
    def load_primitive_weights(cls, organism, ckpts, pfunc):
        pfunc('Before loading pre-trained weights')
        organism.visualize_parameters(pfunc)
        for i, (agent, ckpt) in enumerate(zip(organism.unique_agents, ckpts)):
            agent.transformation.load_state_dict(ckpt['organism'])
            agent.transformation.set_trainable(not organism.args.freeze_primitives)
        pfunc('After loading pre-trained weights')
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
    def update_dirs_primitive(cls, args, ckpt_expnames):
        args.expname = '{}__using__{}'.format(args.expname, '__and__'.join(
            [str(get_expid(e)) for e in ckpt_expnames]))
        return args

    @classmethod
    def main(cls, parse_args):
        args, device = cls.initialize(parse_args())
        ##########################################
        # update args for pre-trained primitives
        # ****************************************
        ckpts = cls.load_checkpoints(args.primitives, args, ckpt_type='primitive')
        args = cls.update_dirs_primitive(args, [c['args'].expname for c in ckpts])
        args.num_primitives = len(ckpts)
        ##########################################
        logger = MultiBaseLogger(args=args)
        task_progression = cls.create_task_progression(logger, args)
        organism = cls.create_organism(device, task_progression, args)
        ##########################################
        # load weights for pre-trained primitives
        # ****************************************
        organism = cls.load_primitive_weights(organism, ckpts, logger.printf)
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
    launcher = HierarchicalPretrainedDecentralizedLauncher()
    launcher.main(parse_args)
