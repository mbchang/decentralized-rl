import numpy as np

from starter_code.sampler.sampler import Sampler
from starter_code.interfaces.interfaces import UtilityArgs
from starter_code.organism.utilities import get_second_highest_bid


class DecentralizedSampler(Sampler):
    def begin_episode(self, env):
        if not self.deterministic and self.organism.args.ado:
            self.organism.agent_dropout()
        state = super(DecentralizedSampler, self).begin_episode(env)
        return state

    def finish_episode(self, state, episode_data, env):
        episode_data = Sampler.finish_episode(self, state, episode_data, env)
        episode_data = self.assign_utilities(episode_data)
        return episode_data

    def assign_utilities(self, society_episode_data):
        for t in range(len(society_episode_data)):
            winner = society_episode_data[t].winner
            bids = society_episode_data[t].bids
            reward = society_episode_data[t].reward  # not t+1!
            start_time = society_episode_data[t].start_time
            end_time = society_episode_data[t].end_time

            if t < len(society_episode_data)-1:
                next_winner = society_episode_data[t+1].winner
                next_bids = society_episode_data[t+1].bids
                next_winner_bid = next_bids[next_winner]
                next_second_highest_bid = get_second_highest_bid(next_bids, next_winner)
            else:
                next_winner_bid = 0
                next_second_highest_bid = 0

            utilities = self.organism.compute_utilities(
                UtilityArgs(
                            bids=bids,
                            winner=winner,
                            next_winner_bid=next_winner_bid,
                            next_second_highest_bid=next_second_highest_bid,
                            reward=reward,
                            start_time=start_time,
                            end_time=end_time))

            society_episode_data[t].set_payoffs(utilities)
        return society_episode_data

    def get_bids_for_episode(self, society_episode_data):
        a_ids = society_episode_data[0].bids.keys()
        episode_bids = {a_id: [] for a_id in a_ids}
        for step_info in society_episode_data:
            for a_id in a_ids:
                episode_bids[a_id].append(step_info.bids[a_id])
        return episode_bids