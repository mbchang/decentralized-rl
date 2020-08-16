import numpy as np

from starter_code.interfaces.tree import LiteralTransformNode
from starter_code.interfaces.interfaces import TransformOutput


class Transform_Agent():
    def __init__(self, id_num):
        self.id_num = id_num

    def get_id(self):
        return self.id_num

    def set_trainable(self, trainable):
        self.trainable = trainable

    def can_be_updated(self):
        return self.trainable

    def set_transformation_registry(self, transformations_by_id):
        self.transformations_by_id = transformations_by_id

    def get_transformation_by_id(self, id_num):
        return self.transformations_by_id[id_num]

    def transform(self):
        raise NotImplementedError

    def __repr__(self):
        return '{} {}'.format(self.__class__.__name__, str(self.id_num))


class LiteralActionTransformation(Transform_Agent):
    def __init__(self, id_num):
        Transform_Agent.__init__(self, id_num)
        self.is_subpolicy = False
        self.set_trainable(False)

    def transform(self, state, env, transform_params=None):
        next_state, reward, done, info = env.step(self.id_num)
        assert info == {}
        transform_node = LiteralTransformNode(
            id_num=self.id_num,
            path_data=dict(reward=reward))
        return TransformOutput(next_state=next_state, done=done, transform_node=transform_node)

    def get_transformation(self):
        return self.id_num

    def clear_buffer(self):
        pass


