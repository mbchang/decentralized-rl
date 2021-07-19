import numpy as np

from starter_code.environment.env_config import simplify_name

def rlalg_config(args):
    args.alg_name = 'ppo'
    rlalg_configs = dict(
        ppo=ppo_config,
    )
    args = rlalg_configs[args.alg_name](args)
    if not hasattr(args, 'parallel_update'):
        args.parallel_update = False
    return args

def ppo_config(args):
    args.gamma = 0.99
    args.plr = 4e-5
    args.vlr = 5e-3
    args.entropy_coeff = 0.1
    if args.debug:
        args.max_buffer_size = 100
        args.optim_batch_size = 10
    else:
        args.max_buffer_size = 4096
        args.optim_batch_size = 256
    return args

def experiment_config(args):
    args.gpu_index = 0

    args.max_epochs = args.pretrain#int(1e7)
    args.num_test = 10
    args.log_every = 10
    args.save_every = 50
    if 'MentalRotation' in args.env_name[0]:
        args.visualize_every = 5
        args.eval_every = 5
    else:
        args.visualize_every = 50
        args.eval_every = 50

    if args.debug:
        args.max_epochs = 12
        args.eval_every = 3
        args.save_every = 3
        args.visualize_every = 3
        args.log_every = 3
        args.num_test = 4
    return args

def lifelong_config(args):
    args.parents = ['root']
    return args

def hierarchical_config(args):
    if not hasattr(args, 'primitives'):
        args.primitives = []
    if not hasattr(args, 'hrl_verbose'):
        args.hrl_verbose = False
    return args

def network_config(args):
    args.hdim = [20, 20]
    return args

def training_config(args):
    args.anneal_policy_lr = True
    args.anneal_policy_lr_step = 100
    args.anneal_policy_lr_gamma = 0.99
    args.anneal_policy_lr_after = 500
    if args.debug:
        args.anneal_policy_lr_step = 1
        args.anneal_policy_lr_after = 2
    return args

def verbose_config(args):
    args.param_verbose = False
    return args

def society_config(args):
    args.redundancy = 2
    args.clone = True
    args.memoryless = True
    return args

def build_expname(args):
    args.expname = simplify_name(args.env_name)
    args.expname = 'i{}i_'.format(create_experiment_id(args))+ args.expname
    if hasattr(args, 'auctiontype'):
        args.expname += '_{}'.format(args.auctiontype)
        if args.clone:
            args.expname += '_cln'
    else:
        args.expname += '_{}'.format(args.alg_name)
    return args

def get_expid(exp_string):
    assert exp_string[0] == 'i'
    assert exp_string[8] == 'i'  # because experiment id has 7 integers
    exp_id = int(exp_string[1:8])
    return exp_id

def create_experiment_id(args):
    if args.expid == '9999999':  # default
         # 10^7 unique ids
        exp_id = ''.join(str(s) for s in np.random.randint(10, size=7)) 
    else:
        exp_id = args.expid
    return exp_id

def process_config(args):
    args = experiment_config(args)
    args = training_config(args)
    args = rlalg_config(args)
    args = network_config(args)
    if hasattr(args, 'auctiontype'):
        args = society_config(args)
    args = lifelong_config(args)
    args = hierarchical_config(args)
    args = verbose_config(args)
    args = build_expname(args)
    return args