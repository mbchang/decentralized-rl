import argparse
import yaml

def build_parser(auction=False, transformation='literal', pretrain_primitives=False, transfer=False):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--subroot', type=str, default='debug', help='folder for storing output')
    parser.add_argument('--cpu', action='store_true', help='force device to be cpu')
    parser.add_argument('--autorm', action='store_true', help='auto remove log folder')
    parser.add_argument('--env-name', nargs='+', type=str, default=['Hopper-v2'], help='list of environments')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--printf', action='store_true', help='print to text file')
    parser.add_argument('--debug', action='store_true', help='turns on debug mode')
    parser.add_argument('--parallel_collect', action='store_true', help='collect data with multiple workers')
    parser.add_argument('--expid', type=str, default='9999999', help='specify id of experiment')
    parser.add_argument('--pretrain', type=int, default=int(1e7), help='limit intial training')

    if auction:
        parser.add_argument('--auctiontype', type=str, default='ccv', help='type of auction mechanism: ccv | v | bb')
        parser.add_argument('--ado', action='store_true', help='agent dropout')
    if transformation == 'subpolicy':
        parser.add_argument('--oplen', nargs='+', type=int, default=[-1], help='max number of time-steps to invoke option. -1 means there is no limit')
        parser.add_argument('--shared_vfn', action='store_true', help='share value function among subpolicies')
        parser.add_argument('--hrl_verbose', action='store_true', help='print statements for hrl debugging')
        if pretrain_primitives:
            parser.add_argument('--primitives', nargs='+', type=yaml.safe_load, required=True, help='checkpoints for subpolicies')
            parser.add_argument('--freeze_primitives', action='store_true', help='freezee the weights of subpolicies')
        else:
            parser.add_argument('--num_primitives', type=int, required=True, help='number of subpolicies to initialize')
    if transfer:
        parser.add_argument('--ckpts', nargs='+', type=yaml.safe_load, required=True, help='checkpoint of society to transfer from')
    
    elif transformation == 'function':
        parser.add_argument('--num_primitives', type=int, default=1, help='number of functions to initialize')
    return parser