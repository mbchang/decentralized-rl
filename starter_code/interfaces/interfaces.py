
from collections import namedtuple

CentralizedOutput = namedtuple('CentralizedOutput', ('action', 'dist_params'))
DecentralizedOutput = namedtuple('DecentralizedOutput', ('action', 'winner', 'bids'))

SphericalMultivariateNormalParams = namedtuple('SphericalMultivariateNormalParams', ('mu', 'logstd'))
BetaParams = namedtuple('BetaParams', ('alpha', 'beta'))

TransformOutput = namedtuple('TransformOutput', ('next_state', 'done', 'transform_node'))
StepOutput = namedtuple('StepOutput', ('done', 'step_info', 'option_length'))
SamplingArgs = namedtuple('SamplingArgs', ('max_steps_this_option', 'deterministic', 'render', 'random_collection'))
PolicyTransformParams = namedtuple('SubpolicyTransformParams', 
    ('max_steps_this_option', 'deterministic', 'render', 
        # 'random_collection'
        ))

UtilityArgs = namedtuple('UtilityArgs', 
    ('bids', 'winner', 'next_winner_bid', 'next_second_highest_bid', 'reward', 'start_time', 'end_time'))