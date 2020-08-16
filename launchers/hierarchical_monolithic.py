from collections import OrderedDict

from launchers.parsers import build_parser
from launchers.monolithic import BaseLauncher
from starter_code.organism.agent import ActorCriticHRLAgent
from starter_code.rl_update.replay_buffer import PathMemory

def parse_args():
    parser = build_parser(transformation='subpolicy')
    args = parser.parse_args()
    args.hrl = True
    return args

class HierarchicalMonolithicLauncher(BaseLauncher):
    @classmethod
    def create_organism(cls, device, task_progression, args):
        replay_buffer = PathMemory(max_replay_buffer_size=args.max_buffer_size)
        policy = cls.centralized_policy_switch(task_progression.state_dim, args.num_primitives, args)
        critic = cls.value_switch(task_progression.state_dim, args)
        transformation_builder = cls.create_transformation_builder(
            task_progression.state_dim, task_progression.action_dim, args)

        transformations = OrderedDict()  # note that these have their own optimizers
        for i in range(args.num_primitives):
            transformations[i] = transformation_builder(id_num=i)

        agent = ActorCriticHRLAgent(
            networks=dict(
                policy=policy(),
                valuefn=critic()),
            transformations = transformations,
            replay_buffer=replay_buffer, 
            args=args).to(device)
        return agent

if __name__ == '__main__':
    launcher = HierarchicalMonolithicLauncher()
    launcher.main(parse_args)