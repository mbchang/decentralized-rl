import numpy as np

from starter_code.infrastructure.utils import from_onehot
from starter_code.sampler.temporary_buffer import DecentralizedStatsCollector

class DecentralizedHRLStatsCollector(DecentralizedStatsCollector):

    def summarize(self):
        DecentralizedStatsCollector.summarize(self)
        self.data['room_stats'] = dict()
        for episode_data in self.data['episode_datas']:
            for i, step in enumerate(episode_data):
                self.data['room_stats'][str(step.state.room)] = dict()
                for a_id in step.bids:
                    self.data['room_stats'][str(step.state.room)][str(a_id)]= dict(bid=[], payoff=[])

    def _bundle_room_payoff_bid(self):
        for episode_data in self.data['episode_datas']:
            for i, stp in enumerate(episode_data):
                for a_id in stp.bids:
                    self.data['room_stats'][str(stp.state.room)][str(a_id)]['bid'].append(stp.bids[a_id])
                    self.data['room_stats'][str(stp.state.room)][str(a_id)]['payoff'].append(stp.payoffs[a_id])
        for room in self.data['room_stats']:
            for a_id in self.data['room_stats'][room].keys():
                self.data['room_stats'][room][a_id]['payoff'] = np.mean(np.array(self.data['room_stats'][room][a_id]['payoff']))
                self.data['room_stats'][room][a_id]['bid'] = np.mean(np.array(self.data['room_stats'][room][a_id]['bid']))
        return self.data['room_stats']

    def bundle_batch_stats(self):
        stats = super(DecentralizedStatsCollector, self).bundle_batch_stats()
        room_stats = self._bundle_room_payoff_bid()
        stats = dict({**stats, **dict(room_stats=room_stats)})
        return stats


class DecentralizedTabularStatsCollector(DecentralizedStatsCollector):

    def summarize(self):
        def get_state_dim(episode_datas):
            state_dim = episode_datas[0][0].state.shape
            assert len(state_dim) == 1
            state_dim = state_dim[0]
            return state_dim

        DecentralizedStatsCollector.summarize(self)
        self.data['state_stats'] = {}
        state_dim = get_state_dim(self.data['episode_datas'])
        for episode_data in self.data['episode_datas']:
            for i, step in enumerate(episode_data):
                assert len(step.state) == state_dim
                state = from_onehot(step.state, state_dim)
                if state not in self.data['state_stats']:
                    self.data['state_stats'][state] = dict(payoffs={}, bids={})
                for a_id in step.bids:
                    self._summarize_agent_payoff_bid(
                        a_id=a_id, step=step, summary_dict=self.data['state_stats'][state])

    def bundle_batch_stats(self):
        stats = DecentralizedStatsCollector.bundle_batch_stats(self)
        state_stats = {}
        for state in self.data['state_stats']:
            state_stats[state] = self._bundle_agent_payoff_bid(summary_dict=self.data['state_stats'][state])
        stats['state_stats'] = state_stats
        return stats

