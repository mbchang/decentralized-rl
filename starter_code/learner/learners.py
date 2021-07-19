from collections import OrderedDict

from starter_code.learner.learner import Learner
from starter_code.infrastructure.log import log_string, MinigridEnvManager
from starter_code.interfaces.transitions import AgentStepInfo, OptionStepInfo, AuctionStepInfo, OptionAuctionStepInfo
from starter_code.sampler.decentralized_sampler import DecentralizedSampler
from starter_code.sampler.domain_specific_temporary_buffers import DecentralizedHRLStatsCollector, DecentralizedTabularStatsCollector
from starter_code.sampler.hierarchy_utils import is_hierarchical
from starter_code.sampler.sampler import Sampler
from starter_code.sampler.temporary_buffer import DecentralizedStatsCollector, StatsCollector, CentralizedStatsCollector


def computation_sampler_builder(base_class):
    class ComputationSampler(base_class):
        def finish_episode(self, state, episode_data, env):
            output = state
            reward = env.apply_loss(output).item()
            # assign reward to last step
            episode_data[-1].hierarchy_info.set_reward(reward)
            episode_data[-1].set_reward(reward)
            episode_data = base_class.finish_episode(self, state, episode_data, env)
            return episode_data
    return ComputationSampler

class CentralizedLearner(Learner):
    def __init__(self, organism, rl_alg, logger, device, args):
        Learner.__init__(self, organism, rl_alg, logger, device, args)

        if organism.transformation_type == 'FunctionTransformation':
            sampler_builder = computation_sampler_builder(Sampler)
        else:
            sampler_builder = Sampler

        if organism.args.envtype=='tab':
            self.stats_collector_builder = CentralizedStatsCollector
        # elif organism.args.envtype=='BabyAI' or organism.args.envtype=='mg': 
        #     # import pdb; pdb.set_trace()
        #     self.stats_collector_builder = CentralizedHRLStatsCollector
        else:
            self.stats_collector_builder = StatsCollector

        if organism.transformation_type == 'SubpolicyTransformation':
            step_info = OptionStepInfo
        else:
            step_info = AgentStepInfo

        # self.stats_collector_builder = StatsCollector
        self.sampler = sampler_builder(
            organism=organism,
            step_info=step_info,
            deterministic=False,
            )

class DecentralizedLearner(Learner):
    def __init__(self, organism, rl_alg, logger, device, args):
        Learner.__init__(self, organism, rl_alg, logger, device, args)

        if organism.transformation_type == 'FunctionTransformation':
            sampler_builder = computation_sampler_builder(DecentralizedSampler)
        else:
            sampler_builder = DecentralizedSampler

        if organism.transformation_type == 'SubpolicyTransformation':
            step_info = OptionAuctionStepInfo
        else:
            step_info = AuctionStepInfo

        # domain specific
        if 'BabyAI' in args.env_name[0]:
            self.stats_collector_builder = DecentralizedHRLStatsCollector
        else:
            self.stats_collector_builder = DecentralizedStatsCollector
        self.sampler = sampler_builder(
            organism=organism,
            step_info=step_info,
            deterministic=False,
            )


    def get_qualitative_output(self, env_manager, sampler, episode_data, epoch, i):
        Learner.get_qualitative_output(self, env_manager, sampler, episode_data, epoch, i)

        if env_manager.visual and isinstance(env_manager, MinigridEnvManager):
            """
            time t | state | high-level action | high-level winner | high-level bids | high-level payoffs | high-level reward | low-level action | next_state | low-level reward |
            time t+1 | state | high-level action | high-level winner | high-level bids | high-level payoffs | high-level reward | low-level action | next_state | low-level reward |
            """
            print_keys = ['Epoch', epoch]
            print_values = []

            def low_level_print(organism, episode_data, high_level_info_dict=None):
                time=0
                for step in episode_data:
                    if not step.hierarchy_info.leaf:
                        sub_organism = organism.transformations[step.hierarchy_info.organism]
                        high_level_info_dict = dict(
                            action=step.action, 
                            winner=step.winner,
                            bids=['{}: {:.5f}'.format(a_id, bid) for a_id, bid in step.bids.items()],
                            payoffs=['{}: {:.5f}'.format(a_id, payoff) for a_id, payoff in step.payoffs.items()],
                            reward=str(step.reward),
                            t=time)
                        low_level_time = low_level_print(sub_organism, step.hierarchy_info.path_data, high_level_info_dict)
                        time+=low_level_time
                    else:
                        step_dict = OrderedDict(
                            t=str(time+high_level_info_dict['t']),
                            high_level_action =  high_level_info_dict['action'] ,
                            high_level_winner = high_level_info_dict['winner'],
                            high_level_bids = high_level_info_dict['bids'],
                            high_level_payoffs = high_level_info_dict['payoffs'],
                            high_level_reward = high_level_info_dict['reward'] ,
                            low_level_action=step.action,
                            low_level_reward=str(step.reward),
                            room_num = step.state.room
                            )
                        keys, values = [], []
                        for key, value in step_dict.items():
                            keys.append(key)
                            values.append(value)
                        print_values.append(values)
                        self.logger.printf(log_string(step_dict))
                        time+=1
                return time
            low_level_print(self.organism, episode_data)
            self.logger.save_print_csv(print_keys, print_values)


class TabularDecentralizedLearner(Learner):
    def __init__(self, organism, rl_alg, logger, device, args):
        Learner.__init__(self, organism, rl_alg, logger, device, args)

        if organism.transformation_type == 'SubpolicyTransformation':
            step_info = OptionAuctionStepInfo
        else:
            step_info = AuctionStepInfo

        self.stats_collector_builder = DecentralizedTabularStatsCollector
        self.sampler = DecentralizedSampler(
            organism=organism,
            step_info=step_info,
            deterministic=False,
            )

    def get_qualitative_output(self, env_manager, sampler, episode_data, epoch, i):
        """
        | time t | state | action | winner | bids | payoffs |
        | time t+1 | state | action | winner | bids | payoffs |
        """
        for t, step_data in enumerate(episode_data):
            step_dict = OrderedDict(
                t='{}\t'.format(t),
                state=env_manager.env.from_onehot(step_data.state),
                action=step_data.action,
                next_state=env_manager.env.from_onehot(step_data.next_state),
                reward='{}\t'.format(step_data.reward),
                mask=step_data.mask)
            if hasattr(step_data, 'bids'):
                assert hasattr(step_data, 'payoffs') and hasattr(step_data, 'winner')
                step_dict['winner'] = step_data.winner
                step_dict['bids'] = ', '.join(['{}: {:.5f}'.format(a_id, bid) for a_id, bid in step_data.bids.items()])
                step_dict['payoffs'] = ', '.join(['{}: {:.5f}'.format(a_id, payoff) for a_id, payoff in step_data.payoffs.items()])
            self.logger.printf(log_string(step_dict))