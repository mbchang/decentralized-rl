from collections import OrderedDict
import numpy as np

def get_second_highest_bid(bids, winner):
    if len(bids) == 1:
        second_highest_bid = 0
    else:
        second_highest_bid = -np.inf
        for index, b in bids.items():
            if b > second_highest_bid and index != winner:
                second_highest_bid = b
                second_highest_bid_index = index
        assert bids[second_highest_bid_index] == second_highest_bid <= bids[winner]
    return second_highest_bid

def vickrey_utilities(self, utility_args, args):
    adjusted_gamma = args.gamma**(utility_args.end_time-utility_args.start_time)
    second_highest_bid = get_second_highest_bid(
        utility_args.bids, utility_args.winner)
    utilities = OrderedDict()
    for a_id in utility_args.bids:
        if a_id == utility_args.swinner:                
            lookahead = utility_args.next_winner_bid
            revenue = utility_args.reward + adjusted_gamma*lookahead
            utilities[a_id] = revenue - second_highest_bid
        else:
            utilities[a_id] = 0
    return utilities

def credit_conserving_vickrey_utilities(utility_args, args):
    adjusted_gamma = args.gamma**(utility_args.end_time-utility_args.start_time)
    second_highest_bid = get_second_highest_bid(
        utility_args.bids, utility_args.winner)
    utilities = OrderedDict()
    for a_id in utility_args.bids:
        if a_id == utility_args.winner:                
            lookahead = utility_args.next_second_highest_bid
            revenue = utility_args.reward + adjusted_gamma*lookahead
            utilities[a_id] = revenue - second_highest_bid
        else:
            utilities[a_id] = 0
    return utilities


def bucket_brigade_utilities(utility_args, args):
    adjusted_gamma = args.gamma**(utility_args.end_time-utility_args.start_time)
    utilities = OrderedDict()
    for a_id in utility_args.bids:
        if a_id == utility_args.winner:
            revenue = utility_args.reward + adjusted_gamma*utility_args.next_winner_bid
            utilities[a_id] = revenue - utility_args.bids[winner]          
        else:
            utilities[a_id] = 0
    return utilities

def environment_reward_utilities(utility_args, args):
    utilities = OrderedDict()
    for a_id in utility_args.bids:
        if a_id == utility_args.winner:
            utilities[a_id] = utility_args.reward
        else:
            utilities[a_id] = 0
    return utilities