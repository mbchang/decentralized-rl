from collections import OrderedDict

from starter_code.experiment.experiment import Experiment
from starter_code.infrastructure.log import log_string

class DecentralizedExperiment(Experiment):
    def __init__(self, learner, task_progression, logger, args):
        super(DecentralizedExperiment, self).__init__(learner, task_progression, logger, args)
        if args.ado and self.learner.parallel_collect:
            raise Exception('Cannot do agent dropout with parallel data collection')

    def _log_agent_payoff_bid(self, s, stats_dict, prefix=''):
        a_keys = {}
        for a_key in stats_dict:
            if isinstance(a_key, str) and 'agent' in a_key:
                a_keys[a_key] = stats_dict[a_key]
        for a_key in sorted(a_keys, key=lambda x: int(x[x.rfind('_')+1:])):
            agent_string = log_string(OrderedDict({
                '{}agent'.format(prefix): a_key[a_key.rfind('_')+1:],
                'avg payoff': stats_dict[a_key]['mean_payoff'],
                'std payoff': stats_dict[a_key]['std_payoff'],
                'min payoff': stats_dict[a_key]['min_payoff'],
                'max payoff': stats_dict[a_key]['max_payoff'],
                'avg bid': stats_dict[a_key]['mean_bid'],
                'std bid': stats_dict[a_key]['std_bid'],
                'min bid': stats_dict[a_key]['min_bid'],
                'max bid': stats_dict[a_key]['max_bid'],
                }))
            s.append(agent_string)
        return s

    def log(self, epoch, epoch_stats, mode):
        s = Experiment.log(self,epoch, epoch_stats, mode)
        s = self._log_agent_payoff_bid(s=s, stats_dict=epoch_stats)
        return s

    def visualize(self, env_manager, epoch, stats, name, eval_mode=False):
        Experiment.visualize(self, env_manager, epoch, stats, name, eval_mode)
        if eval_mode==False and 'room_stats' in stats:
            env_manager.visualize_bids_payoffs(stats['room_stats'], epoch)


class TabularExperiment(DecentralizedExperiment):
    def __init__(self, learner, task_progression, logger, args):
        super(TabularExperiment, self).__init__(learner, task_progression, logger, args)
        self.states_encountered = set()

    def log(self, epoch, epoch_stats, mode):
        s = DecentralizedExperiment.log(self, epoch, epoch_stats, mode)
        for state in sorted(epoch_stats['state_stats']):
            s = self._log_agent_payoff_bid(
                s=s, stats_dict=epoch_stats['state_stats'][state], prefix='state {} '.format(state))
        return s

    def update_state_metrics(self, env_manager, epoch, stats, state_metrics):
        for metric in state_metrics:
            for state in stats['state_stats']:
                env_manager.record_state_variable(state, self.learner.steps, stats['state_stats'][state], metric)

    def plot_state_metrics(self, env_manager, stats, name, state_metrics):
        for metric in state_metrics:
            for state in stats['state_stats']:
                env_manager.visualize_data(state=state, title='{} state {}'.format('_'.join(self.args.env_name), state), metric=metric)
        env_manager.save_json('agent_state_stats.json')

    def visualize(self, env_manager, epoch, stats, name, eval_mode=False):
        DecentralizedExperiment.visualize(self, env_manager, epoch, stats, name, eval_mode)
        state_metrics = ['mean_bid', 'mean_payoff']
        self.update_state_metrics(env_manager, epoch, stats, state_metrics)
        self.plot_state_metrics(env_manager, stats, env_manager.env_name, state_metrics)
