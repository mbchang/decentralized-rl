
from collections import namedtuple
import gym
from gym.wrappers.time_limit import TimeLimit
import gym_minigrid
from gym_minigrid.wrappers import ImgObsWrapper
import babyai
import pprint
import torch
import torch.nn.functional as F

from starter_code.environment.envs import *


class EnvInfo():
    def __init__(self, env_name, env_type, reward_shift=0, reward_scale=1, **kwargs):
        self.env_name = env_name
        self.env_type = env_type

        # set default
        self.reward_shift = reward_shift
        self.reward_scale = reward_scale

        for key, value in kwargs.items():
            self.__setattr__(key, value)

    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __str__(self):
        return str(self.__dict__)

    def to_dict(self):
        return self.__dict__

    def __repr__(self):
        return str(self.__dict__)


def build_env_infos(d):
    count = 0
    tranposed_dict = dict()
    for key, subdict in d.items():
        for subkey in subdict:
            tranposed_dict[subkey] = EnvInfo(env_name=subkey, env_type=key, **subdict[subkey])
            count += 1
    assert len(tranposed_dict.keys()) == count
    return tranposed_dict


def simplify_name(names):
    return '_'.join(''.join(x for  x in name if not x.islower()) for name in names)


class RewardNormalize(gym.RewardWrapper):
    def __init__(self, env, scale=1, shift=0):
        super().__init__(env)
        self.scale = scale
        self.shift = shift

    def reward(self, r):
        return (r - self.shift) * self.scale


class GymRewardNormalize(RewardNormalize):
    def __init__(self, env, scale=1, shift=0):
        if isinstance(env, TimeLimit):
            self._max_episode_steps = env._max_episode_steps
            # unwrap the TimeLimit
            RewardNormalize.__init__(self, env.env, scale, shift)
        else:
            assert False


class MiniGridRewardNormalize(RewardNormalize):
    def __init__(self, env, scale=1, shift=0):
        RewardNormalize.__init__(self, env, scale, shift)
        self.max_steps = env.env.max_steps


class EnvRegistry():
    def __init__(self):
        self.envs_type_name = {
            'gym': {
                'Hopper-v2': dict(),
            },
            'mg': {
                'BabyAI-GreenTwoRoomTest-v0': dict(),
                'BabyAI-BlueTwoRoomTest-v0': dict(),

                'BabyAI-RedGoalTwoRoomTest-v0': dict(),
                'BabyAI-GreenGoalTwoRoomTest-v0': dict(),
                'BabyAI-BlueGoalTwoRoomTest-v0': dict(),
            },
            'vcomp': {
                'MentalRotation': dict(constructor=lambda: VisualComputationEnv(
                    dataset=MentalRotation(),
                    loss_fn=mnist_loss_01,
                    max_steps=2)),
            },
            'tab': {
                'Bandit': dict(constructor=lambda: OneStateOneStepKActionEnv(4)),
                'BanditTransfer': dict(constructor=lambda: OneStateOneStepKActionEnvTransfer(4)),
                'LinBandit': dict(constructor=lambda: MultiTaskDebugTaskBanditA),
                'LinBanditTransfer': dict(constructor=lambda: MultiTaskDebugTaskBanditD),
                'Chain': dict(constructor=lambda: OneHotChainK(6)),
                'Duality': dict(constructor=lambda: Duality(absorbing_reward=-1, asymmetric_coeff=1, big_reward=0.5, small_reward=0.3)), 
                'InvarV1': dict(constructor=MultiTaskDebugTaskTwoStepInvarianceV1),
                'InvarV2': dict(constructor=MultiTaskDebugTaskTwoStepInvarianceV2),
                'ParallelV1': dict(constructor=MultiTaskDebugTaskTwoStepParallelismV1),
                'ParallelV2': dict(constructor=MultiTaskDebugTaskTwoStepParallelismV2),
                'ComAncAB': dict(constructor=MultiTaskDebugCommonAncestorAB),
                'ComAncAC': dict(constructor=MultiTaskDebugCommonAncestorAC),
                'ComAncAE': dict(constructor=MultiTaskDebugCommonAncestorAE),
                'ComAncAF': dict(constructor=MultiTaskDebugCommonAncestorAF),
                'ComAncDB': dict(constructor=MultiTaskDebugCommonAncestorDB),
                'ComAncDC': dict(constructor=MultiTaskDebugCommonAncestorDC),
                'ComDescAC': dict(constructor=MultiTaskDebugCommonDecendantAC),
                'ComDescBC': dict(constructor=MultiTaskDebugCommonDecendantBC),
                'ComDescDC': dict(constructor=MultiTaskDebugCommonDecendantDC),
                'ComDescEC': dict(constructor=MultiTaskDebugCommonDecendantEC),
                'ComDescAF': dict(constructor=MultiTaskDebugCommonDecendantAF),
                'ComDescBF': dict(constructor=MultiTaskDebugCommonDecendantBF),
                'LinABC': dict(constructor=MultiTaskDebugTaskABCSIX),
                'LinABD': dict(constructor=MultiTaskDebugTaskABDSIX),
                'LinAEC': dict(constructor=MultiTaskDebugTaskAECSIX),
                'LinFBC': dict(constructor=MultiTaskDebugTaskFBCSIX),
                'LinABCD': dict(constructor=MultiTaskDebugTaskABCDLenFOUR),
                'LinABCH': dict(constructor=MultiTaskDebugTaskABCHLenFOUR),
                'LinABCDE': dict(constructor=MultiTaskDebugTaskABCDELenFIVE),
                'LinABCDJ': dict(constructor=MultiTaskDebugTaskABCDJLenFIVE),
                'ABCEDPretrain' : dict(constructor= ABCEDPretrain),
                'ABEDTransfer' : dict(constructor= ABEDTransfer),
                'CondensedLinearABC': dict(constructor = CondensedLinearABC),
                'CondensedLinearABCV2': dict(constructor = CondensedLinearABCV2),
                'CondensedLinearABD': dict(constructor = CondensedLinABD),
                'CondensedLinearAEC': dict(constructor = CondensedLinAEC),
                'CondensedLinearFBC': dict(constructor = CondensedLinFBC),
                'LinearOneHotABC': dict(constructor = LinearOneHotABC),
                'LinearOneHotABD': dict(constructor = LinearOneHotABD),
                'LinearOneHotAEC': dict(constructor = LinearOneHotAEC),
                'LinearOneHotFBC': dict(constructor = LinearOneHotFBC),
            },
        }

        self.typecheck(self.envs_type_name)
        self.env_infos = build_env_infos(self.envs_type_name)

    def typecheck(self, d):
        assert type(d) == dict
        for key, value in d.items():
            assert type(value) == dict or type(value) == set

    def get_env_constructor(self, env_name):
        env_type = self.get_env_type(env_name)
        if env_type == 'mg':
            constructor = lambda: MiniGridRewardNormalize(
                ImgObsWrapper(gym.make(env_name)),
                scale=self.env_infos[env_name].reward_scale,
                shift=self.env_infos[env_name].reward_shift)
        elif env_type == 'gym':
            constructor = lambda: GymRewardNormalize(
                gym.make(env_name),
                scale=self.env_infos[env_name].reward_scale,
                shift=self.env_infos[env_name].reward_shift)
        elif env_type in ['tab', 'vcomp']:
            constructor = self.env_infos[env_name].constructor
        else:
            assert False

        return constructor

    def get_env_type(self, env_name):
        return self.env_infos[env_name].env_type

    def get_reward_normalization_info(self, env_name):
        env_info = self.env_infos[env_name]
        return dict(reward_shift=env_info.reward_shift, reward_scale=env_info.reward_scale)
