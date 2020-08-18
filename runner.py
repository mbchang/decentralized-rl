import argparse
from collections import OrderedDict
import itertools
import numpy as np
import os
import ujson
import time

parser = argparse.ArgumentParser()
parser.add_argument('--bandit', action='store_true')
parser.add_argument('--chain', action='store_true')
parser.add_argument('--duality', action='store_true')
parser.add_argument('--mental-rotation', action='store_true')
parser.add_argument('--tworooms-subpolicies', action='store_true')
parser.add_argument('--tworooms-pretrain-task', action='store_true')
parser.add_argument('--tworooms-transfer-task', action='store_true')
parser.add_argument('--for-real', action='store_true')
args = parser.parse_args()

def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))

class Runner():
    def __init__(self, command='', gpus=[]):
        self.gpus = gpus
        self.command = command
        self.flags = {}

    def add_flag(self, flag_name, flag_values=''):
        self.flags[flag_name] = flag_values

    def append_flags_to_command(self, command, flag_dict):
        for flag_name, flag_value in flag_dict.items():
            if type(flag_value) == bool:
                if flag_value == True:
                    command += ' --{}'.format(flag_name)
            else:
                command += ' --{} {}'.format(flag_name, flag_value)
        return command

    def command_prefix(self, i):
        prefix = 'CUDA_VISIBLE_DEVICES={} DISPLAY=:0 '.format(self.gpus[i]) if len(self.gpus) > 0 else 'DISPLAY=:0 '
        command = prefix+self.command
        return command

    def command_suffix(self, command):
        if len(self.gpus) == 0:
            command += ' --cpu'
        command += ' --printf'
        command += ' &'
        return command

    def generate_commands(self, execute=False):
        i = 0
        j = 0
        for flag_dict in product_dict(**self.flags):
            command = self.command_prefix(i)
            command = self.append_flags_to_command(command, flag_dict)
            command = self.command_suffix(command)

            print(command)
            if execute:
                os.system(command)
            if len(self.gpus) > 0:
                i = (i + 1) % len(self.gpus)
            j += 1
        if execute:
            print('Launched {} jobs'.format(j))
        else:
            print('Dry-run')

class RunnerWithIDs(Runner):
    def __init__(self, command, gpus):
        Runner.__init__(self, command, gpus)

    def product_dict(self, **kwargs):
        ordered_kwargs_dict = OrderedDict()
        for k, v in kwargs.items():
            if not k == 'seed':
                ordered_kwargs_dict[k] = v

        keys = ordered_kwargs_dict.keys()
        vals = ordered_kwargs_dict.values()

        for instance in itertools.product(*vals):
            yield dict(zip(keys, instance))

    def generate_commands(self, execute=False):
        if 'seed' not in self.flags:
            Runner.generate_commands(self, execute)
        else:
            i = 0
            j = 0

            for flag_dict in self.product_dict(**self.flags):
                command = self.command_prefix()
                command = self.append_flags_to_command(command, flag_dict)

                # add exp_id: one exp_id for each flag_dict.
                exp_id = ''.join(str(s) for s in np.random.randint(10, size=7))
                command += ' --expid {}'.format(exp_id)

                # command doesn't get modified from here on
                for seed in self.flags['seed']:
                    seeded_command = command
                    seeded_command += ' --seed {}'.format(seed)

                    seeded_command = self.command_suffix(seeded_command)

                    print(seeded_command)
                    if execute:
                        os.system(seeded_command)
                    if len(self.gpus) > 0:
                        i = (i + 1) % len(self.gpus)
                    j += 1
            if execute:
                print('Launched {} jobs'.format(j))
            else:
                print('Dry-run')


def run_bandit():
    r = RunnerWithIDs(command='python launchers/decentralized.py', gpus=[])
    r.add_flag('env-name', ['Bandit'])
    r.add_flag('ado', [True])
    r.add_flag('subroot', ['bandit'])
    r.generate_commands(execute=args.for_real)

def run_chain():
    r = RunnerWithIDs(command='python launchers/decentralized.py', gpus=[])
    r.add_flag('env-name', ['Chain'])
    r.add_flag('parallel_collect', [True])
    r.add_flag('subroot', ['chain'])
    r.generate_commands(execute=args.for_real)

def run_duality():
    r = RunnerWithIDs(command='python launchers/decentralized.py', gpus=[])
    r.add_flag('env-name', ['Duality'])
    r.add_flag('parallel_collect', [True])
    r.add_flag('subroot', ['duality'])
    r.generate_commands(execute=args.for_real)

def run_mental_rotation():
    r = RunnerWithIDs(command='python -W ignore launchers/decentralized_computation.py', gpus=[0])
    r.add_flag('num_primitives', [6])
    r.add_flag('parallel_collect', [True])
    r.add_flag('env-name', ['MentalRotation'])
    r.add_flag('subroot', ['mental_rotation'])
    r.generate_commands(execute=args.for_real)

def run_tworooms_subpolicies():
    r = RunnerWithIDs(command='python -W ignore launchers/monolithic.py', gpus=[0])
    r.add_flag('env-name', ['BabyAI-RedGoalTwoRoomTest-v0', 'BabyAI-GreenGoalTwoRoomTest-v0', 'BabyAI-BlueGoalTwoRoomTest-v0'])
    r.add_flag('parallel_collect', [True])
    r.add_flag('subroot', ['two_rooms'])
    r.generate_commands(execute=args.for_real)

def run_tworooms_pretrain_task():
    primitive_expids = [
        'i9138938i_BAI-RGTRT-0_ppo',
        'i2713858i_BAI-GGTRT-0_ppo',
        'i0342222i_BAI-BGTRT-0_ppo',
    ]
    raise NotImplementedError('Replace primitive_expids with the corresponding folder names generated by pretrain_tworooms_subpolicies')
    primitive_string = ' '.join(ujson.dumps(dict(primitive=expfolder)) for expfolder in primitive_expids).replace('"', '').replace('{', '\"{').replace('}', '}\"').replace(':', ': ')
    """
    An example command would be:
    `python launchers/hierarchical_monolithic_pretrained_primitives.py --env-name BabyAI-GreenGoalTwoRoomTest-v0 --freeze_primitives --primitives "{primitive: i9138938i_BAI-RGTRT-0_ppo}" "{primitive: i2713858i_BAI-GGTRT-0_ppo}" "{primitive: i0342222i_BAI-BGTRT-0_ppo}" --subroot tworooms`
    """
    r = RunnerWithIDs(command='python launchers/hierarchical_monolithic_pretrained_primitives.py', gpus=[0])
    r.add_flag('env-name', ['BabyAI-GreenTwoRoomTest-v0'])
    r.add_flag('freeze_primitives', [True])
    r.add_flag('parallel_collect', [True])
    r.add_flag('primitives', [primitive_string])
    r.add_flag('subroot', ['two_rooms'])
    r.generate_commands(execute=args.for_real)

    """
    An example command would be:
    `python launchers/hierarchical_decentralized_pretrained_primitives.py --env-name BabyAI-GreenGoalTwoRoomTest-v0 --freeze_primitives --primitives "{primitive: i9138938i_BAI-RGTRT-0_ppo}" "{primitive: i2713858i_BAI-GGTRT-0_ppo}" "{primitive: i0342222i_BAI-BGTRT-0_ppo}" --subroot tworooms`
    """
    r = RunnerWithIDs(command='python launchers/hierarchical_decentralized_pretrained_primitives.py', gpus=[0])
    r.add_flag('env-name', ['BabyAI-GreenTwoRoomTest-v0'])
    r.add_flag('freeze_primitives', [True])
    r.add_flag('parallel_collect', [True])
    r.add_flag('primitives', [primitive_string])
    r.add_flag('subroot', ['two_rooms'])
    r.generate_commands(execute=args.for_real)


def run_tworooms_transfer_task():
    primitive_expids = [
        'i9138938i_BAI-RGTRT-0_ppo',
        'i2713858i_BAI-GGTRT-0_ppo',
        'i0342222i_BAI-BGTRT-0_ppo',
    ]
    raise NotImplementedError('Replace primitive_expids with the corresponding folder names generated by pretrain_tworooms_subpolicies')
    primitive_string = ' '.join(ujson.dumps(dict(primitive=expfolder)) for expfolder in primitive_expids).replace('"', '').replace('{', '\"{').replace('}', '}\"').replace(':', ': ')

    monolithic_parent_string = '\"{parent: i0733954i_BAI-GTRT-0_ppo__using__9138938__and__2713858__and__342222}\"'
    decentralized_parent_string = '\"{parent: i8905761i_BAI-GTRT-0_ccv_cln__using__9138938__and__2713858__and__342222}\"'

    """
    An example command would be:
    `python launchers/hierarchical_monolithic_pretrained_primitives_transfer.py --env-name BabyAI-BlueTwoRoomTest-v0 --freeze_primitives --parallel_collect --primitives "{primitive: i9138938i_BAI-RGTRT-0_ppo}" "{primitive: i2713858i_BAI-GGTRT-0_ppo}" "{primitive: i0342222i_BAI-BGTRT-0_ppo}" --ckpts "{parent: i0733954i_BAI-GTRT-0_ppo__using__9138938__and__2713858__and__342222}" --subroot two_rooms`
    """
    r = RunnerWithIDs(command='python launchers/hierarchical_monolithic_pretrained_primitives_transfer.py', gpus=[0])
    r.add_flag('env-name', ['BabyAI-BlueTwoRoomTest-v0'])
    r.add_flag('freeze_primitives', [True])
    r.add_flag('parallel_collect', [True])
    r.add_flag('primitives', [primitive_string])
    r.add_flag('ckpts', [monolithic_parent_string])
    r.add_flag('subroot', ['two_rooms'])
    r.generate_commands(execute=args.for_real)


    """
    An example command would be:
    `python launchers/hierarchical_decentralized_pretrained_primitives_transfer.py --env-name BabyAI-BlueTwoRoomTest-v0 --freeze_primitives --parallel_collect --primitives "{primitive: i9138938i_BAI-RGTRT-0_ppo}" "{primitive: i2713858i_BAI-GGTRT-0_ppo}" "{primitive: i0342222i_BAI-BGTRT-0_ppo}" --ckpts "{parent: i8905761i_BAI-GTRT-0_ccv_cln__using__9138938__and__2713858__and__342222}" --subroot two_rooms`
    """
    r = RunnerWithIDs(command='python launchers/hierarchical_decentralized_pretrained_primitives_transfer.py', gpus=[0])
    r.add_flag('env-name', ['BabyAI-BlueTwoRoomTest-v0'])
    r.add_flag('freeze_primitives', [True])
    r.add_flag('parallel_collect', [True])
    r.add_flag('primitives', [primitive_string])
    r.add_flag('ckpts', [decentralized_parent_string])
    r.add_flag('subroot', ['two_rooms'])
    r.generate_commands(execute=args.for_real)



if __name__ == '__main__':
    if args.bandit:
        run_bandit()
    if args.chain:
        run_chain()
    if args.duality:
        run_duality()
    if args.mental_rotation:
        run_mental_rotation()
    if args.tworooms_subpolicies:
        run_tworooms_subpolicies()
    if args.tworooms_pretrain_task:
        run_tworooms_pretrain_task()
    if args.tworooms_transfer_task:
        run_tworooms_transfer_task()
