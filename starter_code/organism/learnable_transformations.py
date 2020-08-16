from collections import OrderedDict
import numpy as np
import torch.nn as nn

from starter_code.organism.base_agent import BaseAgent
from starter_code.organism.transformations import Transform_Agent
from starter_code.organism.agent import ActorCriticHRLAgent
from starter_code.sampler.sampler import step_agent
from starter_code.sampler.hierarchy_utils import visualize_episode_data
from starter_code.infrastructure.log import log_string
from starter_code.interfaces.tree import OptionTransformNode, FunctionTransformNode
from starter_code.interfaces.transitions import AgentStepInfo
from starter_code.interfaces.interfaces import TransformOutput, PolicyTransformParams


class SubpolicyTransformation(ActorCriticHRLAgent, Transform_Agent):
    def __init__(self, id_num, networks, transformations, replay_buffer, args):
        ActorCriticHRLAgent.__init__(self, networks, transformations, replay_buffer, args)
        Transform_Agent.__init__(self, id_num)
        self.is_subpolicy = True
        # by default it is trainable
        self.set_trainable(True)

    def can_be_updated(self):
        return Transform_Agent.can_be_updated(self) and len(self.replay_buffer) > 0

    def clear_buffer(self):
        self.replay_buffer.clear_buffer()

    def transform(self, state, env, transform_params):
        path_data = []
        t = 0
        while t < transform_params.max_steps_this_option:
            state, step_output = step_agent(
                env=env,
                organism=self,
                state=state,
                step_info_builder=AgentStepInfo,
                transform_params=transform_params)
            path_data.append(step_output.step_info)
            t += step_output.option_length

            if -1 in self.args.oplen:
                assert len(self.args.oplen) == 1
                option_done = step_output.step_info.action == env.env.actions.done
            else:
                option_done = t >= np.random.choice(self.args.oplen)

            if option_done: break
            if step_output.done: break

        if self.args.hrl_verbose:
            print('Option data for option {} with length {}'.format(self.id_num, t))
            visualize_episode_data(path_data)

        transform_node = OptionTransformNode(
            id_num=self.id_num,
            organism=self,
            path_data=path_data)

        return TransformOutput(next_state=state, done=step_output.done, transform_node=transform_node)

    def __repr__(self):
        return Transform_Agent.__repr__(self)


class FunctionTransformation(BaseAgent, Transform_Agent):
    def __init__(self, id_num, networks, args):
        BaseAgent.__init__(self, networks, args)
        Transform_Agent.__init__(self, id_num)
        self.is_subpolicy = False
        self.set_trainable(True)

    def can_be_updated(self):
        # from the high level policy's perspective
        return False

    def transform(self, state, env, transform_params=None):
        next_state, reward, done, info = env.step(self.networks['function'])
        transform_node = FunctionTransformNode(
            id_num=self.id_num, 
            path_data=dict(reward=reward))
        return TransformOutput(next_state=next_state, done=done, transform_node=transform_node)

    def __repr__(self):
        return Transform_Agent.__repr__(self)

    def clear_buffer(self):
        pass




