import argparse
from collections import OrderedDict
import itertools
import numpy as np
import os
import ujson
import time

parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true')
parser.add_argument('--bandit', action='store_true')
parser.add_argument('--bandittransfer', action='store_true')
parser.add_argument('--bandit_dec', action='store_true')
parser.add_argument('--bandittransfer_dec', action='store_true')
parser.add_argument('--invarV1_decent', action='store_true')
parser.add_argument('--invarV2_decent', action='store_true')
parser.add_argument('--invarV1Prime_decent', action='store_true')
parser.add_argument('--invarV1_mon', action='store_true')
parser.add_argument('--invarV2_mon', action='store_true')
parser.add_argument('--invarV1Prime_mon', action='store_true')
parser.add_argument('--commonancpretrain_mon', action='store_true')
parser.add_argument('--commondecpretrain_mon', action='store_true')
parser.add_argument('--commonancpretrain_dec', action='store_true')
parser.add_argument('--commondecpretrain_dec', action='store_true')
parser.add_argument('--commonancrest_mon', action='store_true')
parser.add_argument('--commondecrest_mon', action='store_true')
parser.add_argument('--commonancrest_decent', action='store_true')
parser.add_argument('--commondecrest_decent', action='store_true')
parser.add_argument('--linearpretrain_mon', action='store_true')
parser.add_argument('--linearpretrain_dec', action='store_true')
parser.add_argument('--linearrest_mon', action='store_true')
parser.add_argument('--linearrest_decent', action='store_true')
parser.add_argument('--shared_ccv', action='store_true')
parser.add_argument('--offpolicy', action='store_true')
parser.add_argument('--vlr', action='store_true')
parser.add_argument('--factorized', action='store_true')
parser.add_argument('--for-real', action='store_true')
parser.add_argument('--parent', type=str, default='', help='parent of exp for transfer')



# parser.add_argument('--parent', default=[], action="append")


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
                command = self.command_prefix(i)
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


def run_debug():
    r = RunnerWithIDs(command='python launchers/decentralized.py', gpus=[])
    r.add_flag('env-name', ['Bandit'])
    r.add_flag('ado', [True])
    r.add_flag('subroot', ['bandit'])
    r.generate_commands(execute=args.for_real)

def run_bandit():
    r = RunnerWithIDs(command='python launchers/monolithic.py', gpus=[])
    r.add_flag('env-name', ['Bandit'])
    r.add_flag('parallel_collect', [True])
    r.add_flag('subroot', ['tab_runs'])
    r.add_flag('seed', [i for i in range(10)])
    r.add_flag('pretrain', [250]) #,1250, 6250, 31250])
    if args.factorized:
        r.add_flag('factorized', [True])
    if args.parent != '':
        parent_string = '\"{parent: ' + args.parent + '}\"'
        r.add_flag('ckpts', [parent_string])
    r.generate_commands(execute=args.for_real)

def run_bandit_transfer():
    r = RunnerWithIDs(command='python launchers/transfer.py', gpus=[])
    r.add_flag('env-name', ['BanditTransfer'])
    r.add_flag('parallel_collect', [True])
    r.add_flag('subroot', ['tab_runs'])
    r.add_flag('seed', [i for i in range(10)])
    if args.factorized:
        r.add_flag('factorized', [True])
    if args.parent != '':
        # parent_string = '\"{parent: ' + args.parent + '}\"'
        # r.add_flag('ckpts', [parent_string])
        parentlst = list(args.parent.split(" ")) 
        parentlstformatted = ['\"{parent: ' + par + '}\"' for par in parentlst]
        r.add_flag('ckpts', [parstr for parstr in parentlstformatted])
    r.generate_commands(execute=args.for_real)

def run_bandit_dec():
    r = RunnerWithIDs(command='python launchers/decentralized.py', gpus=[])
    r.add_flag('env-name', ['Bandit'])
    r.add_flag('parallel_collect', [True])
    r.add_flag('subroot', ['tab_runs'])
    r.add_flag('seed', [i for i in range(10)])
    r.add_flag('pretrain', [250])#, 1250, 6250, 31250])
    # if args.factorized:
    #     r.add_flag('factorized', [True])
    if args.parent != '':
        parent_string = '\"{parent: ' + args.parent + '}\"'
        r.add_flag('ckpts', [parent_string])
    r.generate_commands(execute=args.for_real)

def run_bandit_transfer_dec():
    r = RunnerWithIDs(command='python launchers/vickrey_transfer.py', gpus=[])
    r.add_flag('env-name', ['BanditTransfer'])
    r.add_flag('parallel_collect', [True])
    r.add_flag('subroot', ['tab_runs'])
    r.add_flag('seed', [i for i in range(10)])
    # if args.factorized:
    #     r.add_flag('factorized', [True])
    if args.parent != '':
        # parent_string = '\"{parent: ' + args.parent + '}\"'
        # r.add_flag('ckpts', [parent_string])
        parentlst = list(args.parent.split(" ")) 
        parentlstformatted = ['\"{parent: ' + par + '}\"' for par in parentlst]
        r.add_flag('ckpts', [parstr for parstr in parentlstformatted])
    r.generate_commands(execute=args.for_real)

def run_chain():
    r = RunnerWithIDs(command='python launchers/decentralized.py', gpus=[])
    r.add_flag('env-name', ['Chain'])
    r.add_flag('parallel_collect', [True])
    r.add_flag('subroot', ['chain'])
    r.generate_commands(execute=args.for_real)



def run_invarV1_decent():
    r = RunnerWithIDs(command='python launchers/decentralized.py', gpus=[])
    r.add_flag('env-name', ['InvarV1'])
    r.add_flag('parallel_collect', [True])
    r.add_flag('subroot', ['tab_runs'])
    r.add_flag('seed', [i for i in range(10)])
    r.add_flag('pretrain', [1250])
    if args.parent != '':
        parent_string = '\"{parent: ' + args.parent + '}\"'
        r.add_flag('ckpts', [parent_string])
    if args.shared_ccv:
        r.add_flag('share-encoder', [True])
    if args.offpolicy:
        r.add_flag('offpolicy', [True])
        r.add_flag('alg-name', ['sac'])
    if args.vlr: 
        # r.add_flag('vlr', [1e-3, 7e-4, 5e-4, 3e-4])
        r.add_flag('vlr', [1e-3])
    r.generate_commands(execute=args.for_real)

def run_invarV1Prime_decent():
    parent_string = '\"{parent: ' + args.parent + '}\"'
    r = RunnerWithIDs(command='python launchers/vickrey_transfer.py', gpus=[])
    r.add_flag('env-name', ['InvarV1'])
    r.add_flag('parallel_collect', [True])
    r.add_flag('subroot', ['tab_runs'])
    r.add_flag('seed', [i for i in range(10)])
    r.add_flag('ckpts', [parent_string])
    # r.add_flag('pretrain', [1250])
    if args.parent != '':
        # parent_string = '\"{parent: ' + args.parent + '}\"'
        # r.add_flag('ckpts', [parent_string])
        parentlst = list(args.parent.split(" ")) 
        parentlstformatted = ['\"{parent: ' + par + '}\"' for par in parentlst]
        r.add_flag('ckpts', [parstr for parstr in parentlstformatted])
    if args.shared_ccv:
        r.add_flag('share-encoder', [True])
    if args.offpolicy:
        r.add_flag('offpolicy', [True])
        r.add_flag('alg-name', ['sac'])
    if args.vlr: 
        # r.add_flag('vlr', [1e-3, 7e-4, 5e-4, 3e-4])
        r.add_flag('vlr', [1e-3])
    r.generate_commands(execute=args.for_real)

def run_invarV2_decent():
    parent_string = '\"{parent: ' + args.parent + '}\"'
    r = RunnerWithIDs(command='python launchers/vickrey_transfer.py', gpus=[])
    r.add_flag('env-name', ['InvarV2'])
    r.add_flag('parallel_collect', [True])
    r.add_flag('subroot', ['tab_runs'])
    r.add_flag('seed', [i for i in range(10)])
    r.add_flag('ckpts', [parent_string])
    r.add_flag('pretrain', [1250, 6250, 31250])
    if args.shared_ccv:
        r.add_flag('share-encoder', [True])
    if args.offpolicy:
        r.add_flag('offpolicy', [True])
        r.add_flag('alg-name', ['sac'])
    if args.vlr: 
        # r.add_flag('vlr', [1e-3, 7e-4, 5e-4, 3e-4])
        r.add_flag('vlr', [1e-3])

    r.generate_commands(execute=args.for_real)

def run_invarV1_mon():
    r = RunnerWithIDs(command='python launchers/monolithic.py', gpus=[])
    r.add_flag('env-name', ['InvarV1'])
    r.add_flag('parallel_collect', [True])
    r.add_flag('subroot', ['tab_runs'])
    r.add_flag('seed', [i for i in range(10)])
    r.add_flag('pretrain', [1250])    
    if args.factorized:
        r.add_flag('factorized', [True])
    if args.parent != '':
        parent_string = '\"{parent: ' + args.parent + '}\"'
        r.add_flag('ckpts', [parent_string])
    r.generate_commands(execute=args.for_real)

def run_invarV2_mon():
    parent_string = '\"{parent: ' + args.parent + '}\"'
    r = RunnerWithIDs(command='python launchers/transfer.py', gpus=[])
    r.add_flag('env-name', ['InvarV2'])
    r.add_flag('parallel_collect', [True])
    r.add_flag('subroot', ['tab_runs'])
    r.add_flag('seed', [i for i in range(10)])
    r.add_flag('pretrain', [1250, 6250, 31250])
    if args.factorized:
        r.add_flag('factorized', [True])
    r.add_flag('ckpts', [parent_string])
    r.generate_commands(execute=args.for_real)

def run_invarV1Prime_mon():
    # import pdb; pdb.set_trace()
    # for parent in args.parent: 
    #     parent_string = '\"{parent: ' + parent + '}\"'
    #     parentstrlst.append(parent_string)
    r = RunnerWithIDs(command='python launchers/transfer.py', gpus=[])
    r.add_flag('env-name', ['InvarV1'])
    r.add_flag('parallel_collect', [True])
    r.add_flag('subroot', ['tab_runs'])
    r.add_flag('seed', [i for i in range(10)])
    if args.factorized:
        r.add_flag('factorized', [True])
    parentlst = list(args.parent.split(" ")) 
    parentlstformatted = ['\"{parent: ' + par + '}\"' for par in parentlst]
    r.add_flag('ckpts', [parstr for parstr in parentlstformatted])
    r.generate_commands(execute=args.for_real)

def run_commonanc_pretrain_mon():
    r = RunnerWithIDs(command='python launchers/monolithic.py', gpus=[])
    r.add_flag('env-name', ['ComAncAB ComAncAC'])
    r.add_flag('parallel_collect', [True])
    r.add_flag('subroot', ['tab_runs'])
    r.add_flag('seed', [i for i in range(10)])
    r.add_flag('pretrain', [2450])    
    if args.factorized:
        r.add_flag('factorized', [True])
    if args.parent != '':
        parent_string = '\"{parent: ' + args.parent + '}\"'
        r.add_flag('ckpts', [parent_string])
    r.generate_commands(execute=args.for_real)

def run_commonanc_pretrain_dec():
    r = RunnerWithIDs(command='python launchers/decentralized.py', gpus=[])
    r.add_flag('env-name', ['ComAncAB ComAncAC'])
    r.add_flag('parallel_collect', [True])
    r.add_flag('subroot', ['tab_runs'])
    r.add_flag('seed', [i for i in range(10)])
    r.add_flag('pretrain', [2450])    
    if args.parent != '':
        parent_string = '\"{parent: ' + args.parent + '}\"'
        r.add_flag('ckpts', [parent_string])
    if args.shared_ccv:
        print('gogogogo')
        r.add_flag('share-encoder', [True])
    if args.offpolicy:
        r.add_flag('offpolicy', [True])
        r.add_flag('alg-name', ['sac'])
    if args.vlr: 
        # r.add_flag('vlr', [1e-3, 7e-4, 5e-4, 3e-4])
        r.add_flag('vlr', [1e-3])
    r.generate_commands(execute=args.for_real)

def run_commonanc_rest_mon():

    r = RunnerWithIDs(command='python launchers/transfer.py', gpus=[])
    r.add_flag('env-name', ['ComAncDB ComAncDC'])
    r.add_flag('parallel_collect', [True])
    r.add_flag('subroot', ['tab_runs'])
    r.add_flag('seed', [i for i in range(10)])
    if args.factorized:
        r.add_flag('factorized', [True])
    if args.parent != '':
        parent_string = '\"{parent: ' + args.parent + '}\"'
        r.add_flag('ckpts', [parent_string])
    r.generate_commands(execute=args.for_real)

    r = RunnerWithIDs(command='python launchers/transfer.py', gpus=[])
    r.add_flag('env-name', ['ComAncAB ComAncAE'])
    r.add_flag('parallel_collect', [True])
    r.add_flag('subroot', ['tab_runs'])
    r.add_flag('seed', [i for i in range(10)])
    if args.factorized:
        r.add_flag('factorized', [True])
    if args.parent != '':
        parent_string = '\"{parent: ' + args.parent + '}\"'
        r.add_flag('ckpts', [parent_string])
    r.generate_commands(execute=args.for_real)

    r = RunnerWithIDs(command='python launchers/transfer.py', gpus=[])
    r.add_flag('env-name', ['ComAncAE ComAncAF'])
    r.add_flag('parallel_collect', [True])
    r.add_flag('subroot', ['tab_runs'])
    r.add_flag('seed', [i for i in range(10)])
    if args.factorized:
        r.add_flag('factorized', [True])
    if args.parent != '':
        parent_string = '\"{parent: ' + args.parent + '}\"'
        r.add_flag('ckpts', [parent_string])
    r.generate_commands(execute=args.for_real)

def run_commonanc_rest_decent():

    r = RunnerWithIDs(command='python launchers/vickrey_transfer.py', gpus=[])
    r.add_flag('env-name', ['ComAncDB ComAncDC'])
    r.add_flag('parallel_collect', [True])
    r.add_flag('subroot', ['tab_runs'])
    r.add_flag('cpu', [True])
    r.add_flag('parallel_collect', [True])
    r.add_flag('seed', [i for i in range(10)])
    if args.parent != '':
        parent_string = '\"{parent: ' + args.parent + '}\"'
        r.add_flag('ckpts', [parent_string])
    if args.shared_ccv:
        r.add_flag('share-encoder', [True])
    if args.offpolicy:
        r.add_flag('offpolicy', [True])
        r.add_flag('alg-name', ['sac'])
    if args.vlr: 
        # r.add_flag('vlr', [1e-3, 7e-4, 5e-4, 3e-4])
        r.add_flag('vlr', [1e-3])
    r.generate_commands(execute=args.for_real)

    r = RunnerWithIDs(command='python launchers/vickrey_transfer.py', gpus=[])
    r.add_flag('env-name', ['ComAncAB ComAncAE'])
    r.add_flag('parallel_collect', [True])
    r.add_flag('subroot', ['tab_runs'])
    r.add_flag('cpu', [True])
    r.add_flag('parallel_collect', [True])
    r.add_flag('seed', [i for i in range(10)])
    if args.parent != '':
        parent_string = '\"{parent: ' + args.parent + '}\"'
        r.add_flag('ckpts', [parent_string])
    if args.shared_ccv:
        r.add_flag('share-encoder', [True])
    if args.offpolicy:
        r.add_flag('offpolicy', [True])
        r.add_flag('alg-name', ['sac'])
    if args.vlr: 
        # r.add_flag('vlr', [1e-3, 7e-4, 5e-4, 3e-4])
        r.add_flag('vlr', [1e-3])
    r.generate_commands(execute=args.for_real)

    r = RunnerWithIDs(command='python launchers/vickrey_transfer.py', gpus=[])
    r.add_flag('env-name', ['ComAncAE ComAncAF'])
    r.add_flag('parallel_collect', [True])
    r.add_flag('subroot', ['tab_runs'])
    r.add_flag('cpu', [True])
    r.add_flag('parallel_collect', [True])
    r.add_flag('seed', [i for i in range(10)])
    if args.parent != '':
        parent_string = '\"{parent: ' + args.parent + '}\"'
        r.add_flag('ckpts', [parent_string])
    if args.shared_ccv:
        r.add_flag('share-encoder', [True])
    if args.offpolicy:
        r.add_flag('offpolicy', [True])
        r.add_flag('alg-name', ['sac'])
    if args.vlr: 
        # r.add_flag('vlr', [1e-3, 7e-4, 5e-4, 3e-4])
        r.add_flag('vlr', [1e-3])
    r.generate_commands(execute=args.for_real)

def run_commondec_pretrain_mon():
    r = RunnerWithIDs(command='python launchers/monolithic.py', gpus=[])
    r.add_flag('env-name', ['ComDescAC ComDescBC'])
    r.add_flag('parallel_collect', [True])
    r.add_flag('subroot', ['tab_runs'])
    r.add_flag('seed', [i for i in range(10)])
    r.add_flag('pretrain', [2450])    
    if args.factorized:
        r.add_flag('factorized', [True])
    if args.parent != '':
        parent_string = '\"{parent: ' + args.parent + '}\"'
        r.add_flag('ckpts', [parent_string])
    r.generate_commands(execute=args.for_real)
    
def run_commondec_pretrain_dec():
    r = RunnerWithIDs(command='python launchers/decentralized.py', gpus=[])
    r.add_flag('env-name', ['ComDescAC ComDescBC'])
    r.add_flag('parallel_collect', [True])
    r.add_flag('subroot', ['tab_runs'])
    r.add_flag('seed', [i for i in range(10)])
    r.add_flag('pretrain', [2450])    
    if args.parent != '':
        parent_string = '\"{parent: ' + args.parent + '}\"'
        r.add_flag('ckpts', [parent_string])
    if args.shared_ccv:
        r.add_flag('share-encoder', [True])
    if args.offpolicy:
        r.add_flag('offpolicy', [True])
        r.add_flag('alg-name', ['sac'])
    if args.vlr: 
        # r.add_flag('vlr', [1e-3, 7e-4, 5e-4, 3e-4])
        r.add_flag('vlr', [1e-3])
    r.generate_commands(execute=args.for_real)

def run_commondec_rest_mon():
    r = RunnerWithIDs(command='python launchers/transfer.py', gpus=[])
    r.add_flag('env-name', ['ComDescAF ComDescBF'])
    r.add_flag('parallel_collect', [True])
    r.add_flag('subroot', ['tab_runs'])
    r.add_flag('seed', [i for i in range(10)])
    if args.factorized:
        r.add_flag('factorized', [True])
    if args.parent != '':
        parent_string = '\"{parent: ' + args.parent + '}\"'
        r.add_flag('ckpts', [parent_string])
    r.generate_commands(execute=args.for_real)

    r = RunnerWithIDs(command='python launchers/transfer.py', gpus=[])
    r.add_flag('env-name', ['ComDescDC ComDescEC'])
    r.add_flag('parallel_collect', [True])
    r.add_flag('subroot', ['tab_runs'])
    r.add_flag('seed', [i for i in range(10)])
    if args.factorized:
        r.add_flag('factorized', [True])
    if args.parent != '':
        parent_string = '\"{parent: ' + args.parent + '}\"'
        r.add_flag('ckpts', [parent_string])
    r.generate_commands(execute=args.for_real)

    r = RunnerWithIDs(command='python launchers/transfer.py', gpus=[])
    r.add_flag('env-name', ['ComDescAC ComDescDC'])
    r.add_flag('parallel_collect', [True])
    r.add_flag('subroot', ['tab_runs'])
    r.add_flag('seed', [i for i in range(10)])
    if args.factorized:
        r.add_flag('factorized', [True])
    if args.parent != '':
        parent_string = '\"{parent: ' + args.parent + '}\"'
        r.add_flag('ckpts', [parent_string])
    r.generate_commands(execute=args.for_real)

def run_commondec_rest_decent():

    r = RunnerWithIDs(command='python launchers/vickrey_transfer.py', gpus=[])
    r.add_flag('env-name', ['ComDescAF ComDescBF'])
    r.add_flag('parallel_collect', [True])
    r.add_flag('subroot', ['tab_runs'])
    r.add_flag('cpu', [True])
    r.add_flag('parallel_collect', [True])
    r.add_flag('seed', [i for i in range(10)])
    if args.parent != '':
        parent_string = '\"{parent: ' + args.parent + '}\"'
        r.add_flag('ckpts', [parent_string])
    if args.shared_ccv:
        r.add_flag('share-encoder', [True])
    if args.offpolicy:
        r.add_flag('offpolicy', [True])
        r.add_flag('alg-name', ['sac'])
    if args.vlr: 
        # r.add_flag('vlr', [1e-3, 7e-4, 5e-4, 3e-4])
        r.add_flag('vlr', [1e-3])
    r.generate_commands(execute=args.for_real)

    r = RunnerWithIDs(command='python launchers/vickrey_transfer.py', gpus=[])
    r.add_flag('env-name', ['ComDescDC ComDescEC'])
    r.add_flag('parallel_collect', [True])
    r.add_flag('subroot', ['tab_runs'])
    r.add_flag('cpu', [True])
    r.add_flag('parallel_collect', [True])
    r.add_flag('seed', [i for i in range(10)])
    if args.parent != '':
        parent_string = '\"{parent: ' + args.parent + '}\"'
        r.add_flag('ckpts', [parent_string])
    if args.shared_ccv:
        r.add_flag('share-encoder', [True])
    if args.offpolicy:
        r.add_flag('offpolicy', [True])
        r.add_flag('alg-name', ['sac'])
    if args.vlr: 
        # r.add_flag('vlr', [1e-3, 7e-4, 5e-4, 3e-4])
        r.add_flag('vlr', [1e-3])
    r.generate_commands(execute=args.for_real)  

    r = RunnerWithIDs(command='python launchers/vickrey_transfer.py', gpus=[])
    r.add_flag('env-name', ['ComDescAC ComDescDC'])
    r.add_flag('parallel_collect', [True])
    r.add_flag('subroot', ['tab_runs'])
    r.add_flag('cpu', [True])
    r.add_flag('parallel_collect', [True])
    r.add_flag('seed', [i for i in range(10)])
    if args.parent != '':
        parent_string = '\"{parent: ' + args.parent + '}\"'
        r.add_flag('ckpts', [parent_string])
    if args.shared_ccv:
        r.add_flag('share-encoder', [True])
    if args.offpolicy:
        r.add_flag('offpolicy', [True])
        r.add_flag('alg-name', ['sac'])
    if args.vlr: 
        # r.add_flag('vlr', [1e-3, 7e-4, 5e-4, 3e-4])
        r.add_flag('vlr', [1e-3])
    r.generate_commands(execute=args.for_real)


def run_linear_pretrain_mon():
    r = RunnerWithIDs(command='python launchers/monolithic.py', gpus=[])
    r.add_flag('env-name', ['LinABC'])
    r.add_flag('parallel_collect', [True])
    r.add_flag('subroot', ['tab_runs'])
    r.add_flag('seed', [i for i in range(10)])
    r.add_flag('pretrain', [2450])#, 6250, 31250])
    if args.factorized:
        r.add_flag('factorized', [True])
    if args.parent != '':
        parent_string = '\"{parent: ' + args.parent + '}\"'
        r.add_flag('ckpts', [parent_string])
    r.generate_commands(execute=args.for_real)

def run_linear_pretrain_dec():
    r = RunnerWithIDs(command='python launchers/decentralized.py', gpus=[])
    r.add_flag('env-name', ['LinABC'])
    r.add_flag('parallel_collect', [True])
    r.add_flag('subroot', ['tab_runs'])
    r.add_flag('seed', [0,1,2,3,4,5,6,7,8,9])
    r.add_flag('pretrain', [2450])#[1250, 6250, 31250])
    if args.parent != '':
        parent_string = '\"{parent: ' + args.parent + '}\"'
        r.add_flag('ckpts', [parent_string])
    if args.shared_ccv:
        r.add_flag('share-encoder', [True])
    if args.offpolicy:
        r.add_flag('offpolicy', [True])
        r.add_flag('alg-name', ['sac'])
    if args.vlr: 
        # r.add_flag('vlr', [1e-3, 7e-4, 5e-4, 3e-4])
        r.add_flag('vlr', [1e-3])
    r.generate_commands(execute=args.for_real)

def run_linear_rest_decent():

    r = RunnerWithIDs(command='python launchers/vickrey_transfer.py', gpus=[])
    r.add_flag('env-name', ['LinABD'])
    r.add_flag('parallel_collect', [True])
    r.add_flag('subroot', ['tab_runs'])
    r.add_flag('seed', [0,1,2,3,4,5,6,7,8,9])
    
    # if args.parent != '':
    #     parent_string = '\"{parent: ' + args.parent + '}\"'
    #     r.add_flag('ckpts', [parent_string])
    if args.parent != '':
        # parent_string = '\"{parent: ' + args.parent + '}\"'
        # r.add_flag('ckpts', [parent_string])
        parentlst = list(args.parent.split(" ")) 
        parentlstformatted = ['\"{parent: ' + par + '}\"' for par in parentlst]
        r.add_flag('ckpts', [parstr for parstr in parentlstformatted])
    if args.shared_ccv:
        r.add_flag('share-encoder', [True])
    if args.offpolicy:
        r.add_flag('offpolicy', [True])
        r.add_flag('alg-name', ['sac'])
    if args.vlr: 
        # r.add_flag('vlr', [1e-3, 7e-4, 5e-4, 3e-4])
        r.add_flag('vlr', [1e-3])
    r.generate_commands(execute=args.for_real)

    r = RunnerWithIDs(command='python launchers/vickrey_transfer.py', gpus=[])
    r.add_flag('env-name', ['LinAEC'])
    r.add_flag('parallel_collect', [True])
    r.add_flag('subroot', ['tab_runs'])
    r.add_flag('seed', [i for i in range(10)])
    # if args.parent != '':
    #     parent_string = '\"{parent: ' + args.parent + '}\"'
    #     r.add_flag('ckpts', [parent_string])
    if args.parent != '':
        # parent_string = '\"{parent: ' + args.parent + '}\"'
        # r.add_flag('ckpts', [parent_string])
        parentlst = list(args.parent.split(" ")) 
        parentlstformatted = ['\"{parent: ' + par + '}\"' for par in parentlst]
        r.add_flag('ckpts', [parstr for parstr in parentlstformatted])
    if args.shared_ccv:
        r.add_flag('share-encoder', [True])
    if args.offpolicy:
        r.add_flag('offpolicy', [True])
        r.add_flag('alg-name', ['sac'])
    if args.vlr: 
        # r.add_flag('vlr', [1e-3, 7e-4, 5e-4, 3e-4])
        r.add_flag('vlr', [1e-3])
    r.generate_commands(execute=args.for_real)  

    r = RunnerWithIDs(command='python launchers/vickrey_transfer.py', gpus=[])
    r.add_flag('env-name', ['LinFBC'])
    r.add_flag('parallel_collect', [True])
    r.add_flag('subroot', ['tab_runs'])
    r.add_flag('seed', [i for i in range(10)])
    # if args.parent != '':
    #     parent_string = '\"{parent: ' + args.parent + '}\"'
    #     r.add_flag('ckpts', [parent_string])
    if args.parent != '':
        # parent_string = '\"{parent: ' + args.parent + '}\"'
        # r.add_flag('ckpts', [parent_string])
        parentlst = list(args.parent.split(" ")) 
        parentlstformatted = ['\"{parent: ' + par + '}\"' for par in parentlst]
        r.add_flag('ckpts', [parstr for parstr in parentlstformatted])
    if args.shared_ccv:
        r.add_flag('share-encoder', [True])
    if args.offpolicy:
        r.add_flag('offpolicy', [True])
        r.add_flag('alg-name', ['sac'])
    if args.vlr: 
        # r.add_flag('vlr', [1e-3, 7e-4, 5e-4, 3e-4])
        r.add_flag('vlr', [1e-3])
    r.generate_commands(execute=args.for_real)

def run_linear_rest_mon():
    r = RunnerWithIDs(command='python launchers/transfer.py', gpus=[])
    r.add_flag('env-name', ['LinABD'])
    r.add_flag('parallel_collect', [True])
    r.add_flag('subroot', ['tab_runs'])
    # r.add_flag('seed', [0,1,2,3,4,5,6,7,8])
    r.add_flag('seed', [i for i in range(10)])
    if args.factorized:
        r.add_flag('factorized', [True])
    if args.parent != '':
        # parent_string = '\"{parent: ' + args.parent + '}\"'
        # r.add_flag('ckpts', [parent_string])
        parentlst = list(args.parent.split(" ")) 
        parentlstformatted = ['\"{parent: ' + par + '}\"' for par in parentlst]
        r.add_flag('ckpts', [parstr for parstr in parentlstformatted])
    r.generate_commands(execute=args.for_real)

    r = RunnerWithIDs(command='python launchers/transfer.py', gpus=[])
    r.add_flag('env-name', ['LinAEC'])
    r.add_flag('parallel_collect', [True])
    r.add_flag('subroot', ['tab_runs'])
    r.add_flag('seed', [i for i in range(10)])
    if args.factorized:
        r.add_flag('factorized', [True])
    if args.parent != '':
        # parent_string = '\"{parent: ' + args.parent + '}\"'
        # r.add_flag('ckpts', [parent_string])
        parentlst = list(args.parent.split(" ")) 
        parentlstformatted = ['\"{parent: ' + par + '}\"' for par in parentlst]
        r.add_flag('ckpts', [parstr for parstr in parentlstformatted])
    r.generate_commands(execute=args.for_real)

    r = RunnerWithIDs(command='python launchers/transfer.py', gpus=[])
    r.add_flag('env-name', ['LinFBC'])
    r.add_flag('parallel_collect', [True])
    r.add_flag('subroot', ['tab_runs'])
    r.add_flag('seed', [i for i in range(10)])
    if args.factorized:
        r.add_flag('factorized', [True])
    if args.parent != '':
        # parent_string = '\"{parent: ' + args.parent + '}\"'
        # r.add_flag('ckpts', [parent_string])
        parentlst = list(args.parent.split(" ")) 
        parentlstformatted = ['\"{parent: ' + par + '}\"' for par in parentlst]
        r.add_flag('ckpts', [parstr for parstr in parentlstformatted])
    r.generate_commands(execute=args.for_real)




if __name__ == '__main__':
    if args.debug:
        run_debug()
    if args.bandit:
        run_bandit()
    if args.bandittransfer:
        run_bandit_transfer()
    if args.bandit_dec:
        run_bandit_dec()
    if args.bandittransfer_dec:
        run_bandit_transfer_dec()
    if args.invarV1_decent:
        run_invarV1_decent()
    if args.invarV1_mon:
        run_invarV1_mon()
    if args.invarV2_decent:
        run_invarV2_decent()
    if args.invarV2_mon:
        run_invarV2_mon()
    if args.invarV1Prime_decent:
        run_invarV1Prime_decent()
    if args.invarV1Prime_mon:
        run_invarV1Prime_mon()
    if args.commonancpretrain_mon:
        run_commonanc_pretrain_mon()
    if args.commonancpretrain_dec:
        run_commonanc_pretrain_dec()
    if args.commondecpretrain_mon:
        run_commondec_pretrain_mon()
    if args.commondecpretrain_dec:
        run_commondec_pretrain_dec()
    if args.commonancrest_mon:
        run_commonanc_rest_mon()
    if args.commondecrest_mon:
        run_commondec_rest_mon()    
    if args.commonancrest_decent:
        run_commonanc_rest_decent()
    if args.commondecrest_decent:
        run_commondec_rest_decent()   
    if args.linearpretrain_mon:
        run_linear_pretrain_mon()
    if args.linearpretrain_dec:
        run_linear_pretrain_dec()
    if args.linearrest_mon:
        run_linear_rest_mon()
    if args.linearrest_decent:
        run_linear_rest_decent()