
import numpy as np

class StatsCollector:
    def __init__(self):
        self.reset()

    def reset(self):
        self.data = dict(steps=0, episode_datas=[])

    def append(self, episode_data):
        self.data['steps'] += len(episode_data)
        self.data['episode_datas'].append(episode_data)

    def extend(self, other_data):
        self.data['steps'] += other_data['steps']
        self.data['episode_datas'].extend(other_data['episode_datas'])

    def get_total_steps(self):
        return self.data['steps']

    def summarize(self):
        self.data['returns'] = [sum([e.reward for e in episode_data])
            for episode_data in self.data['episode_datas']]

    def bundle_batch_stats(self):
        self.summarize()
        stats = dict(num_episodes=len(self.data['episode_datas']))
        stats = dict({**stats, **self.log_metrics(np.array(self.data['returns']), 'return')})
        stats = dict({**stats, **self.log_metrics(np.array(self.data['steps']), 'steps')})
        return stats

    def __len__(self):
        return len(self.data['episode_datas'])

    def log_metrics(self, data, label):
        labeler = lambda cmp: '{}_{}'.format(cmp, label)
        stats = {}
        stats[label] = data
        stats[labeler('mean')] = np.mean(data)
        stats[labeler('std')] = np.std(data)
        stats[labeler('min')] = np.min(data)
        stats[labeler('max')] = np.max(data)
        stats[labeler('total')] = np.sum(data)
        return stats


class DecentralizedStatsCollector(StatsCollector):

    def _summarize_agent_payoff_bid(self, a_id, step, summary_dict):
        p = a_id not in summary_dict['payoffs']
        q = a_id not in summary_dict['bids']
        assert (p and q) or (not p and not q)

        if p:
            summary_dict['payoffs'][a_id] = []
            summary_dict['bids'][a_id] = []

        summary_dict['payoffs'][a_id].append(step.payoffs[a_id])
        summary_dict['bids'][a_id].append(step.bids[a_id])

    def summarize(self):
        StatsCollector.summarize(self)
        self.data['payoffs'] = {}  
        self.data['bids'] = {}
        for episode_data in self.data['episode_datas']:
            for i, step in enumerate(episode_data):
                for a_id in step.bids:
                    self._summarize_agent_payoff_bid(
                        a_id=a_id, step=step, summary_dict=self.data)

    def _bundle_agent_payoff_bid(self, summary_dict):
        agent_stats = {}
        for a_id in summary_dict['bids']:
            a_key = 'agent_{}'.format(a_id)
            agent_stats[a_key] = {
                **self.log_metrics(np.array(summary_dict['payoffs'][a_id]), 'payoff'),
                **self.log_metrics(np.array(summary_dict['bids'][a_id]), 'bid')}
        return agent_stats

    def bundle_batch_stats(self):
        stats = super(DecentralizedStatsCollector, self).bundle_batch_stats()
        agent_stats = self._bundle_agent_payoff_bid(summary_dict=self.data)
        stats = dict({**stats, **agent_stats})
        return stats
