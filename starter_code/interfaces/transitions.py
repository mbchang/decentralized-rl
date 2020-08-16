class AbstractStepInfo():
    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return str(self.__dict__)

    def to_dict(self):
        return self.__dict__


class BasicStepInfo(AbstractStepInfo):
    def __init__(self, state, organism_output, next_state, info):
        self.state = state
        self.organism_output = organism_output
        self.next_state = next_state
        self.hierarchy_info = info

    def get_hierarchy_info(self):
        return self.hierarchy_info

    def set_reward(self, reward):
        self.reward = reward
    
    @property
    def start_time(self):
        return self.hierarchy_info.before

    @property
    def end_time(self):
        return self.hierarchy_info.after

    @property
    def current_transformation_id(self):
        return self.hierarchy_info.current_transformation_id

    @property
    def next_transformation_id(self):
        return self.hierarchy_info.next_transformation_id


class AgentStepInfo(BasicStepInfo):
    def __init__(self, state, organism_output, next_state, info):
        super(AgentStepInfo, self).__init__(
            state, organism_output, next_state, info)
        self.action = self.organism_output.action.get_id()
        self.action_dist = self.organism_output.dist_params

    def set_reward(self, reward):
        # assume that this is a leaf
        assert reward == self.get_hierarchy_info().get_reward()
        BasicStepInfo.set_reward(self, reward)


class AuctionStepInfo(BasicStepInfo):
    def __init__(self, state, organism_output, next_state, info):
        super(AuctionStepInfo, self).__init__(
            state, organism_output, next_state, info)
        self.bids = self.organism_output.bids
        self.action = self.organism_output.action
        self.winner = self.organism_output.winner

    def set_reward(self, reward):
        # assume that this is a leaf
        assert reward == self.get_hierarchy_info().get_reward()
        BasicStepInfo.set_reward(self, reward)

    def set_payoffs(self, payoffs):
        self.payoffs = payoffs

    def set_bid_diff(self, bid_diff):
        self.bid_difference = bid_diff

    def set_Q_diff(self, Q_diff):
        self.Q_difference = Q_diff


class OptionAuctionStepInfo(BasicStepInfo):
    def __init__(self, state, organism_output, next_state, info):
        BasicStepInfo.__init__(self, state, organism_output, next_state, info)
        self.bids = self.organism_output.bids
        self.action = self.organism_output.action
        self.winner = self.organism_output.winner
        self.option_length = self.hierarchy_info.get_length()
        self.reward = None

    def set_payoffs(self, payoffs):
        assert self.reward is not None
        self.payoffs = payoffs

    def set_bid_diff(self, bid_diff):
        self.bid_difference = bid_diff

    def set_Q_diff(self, Q_diff):
        self.Q_difference = Q_diff


class OptionStepInfo(BasicStepInfo):
    def __init__(self, state, organism_output, next_state, info):
        BasicStepInfo.__init__(self, state, organism_output, next_state, info)
        self.action = self.organism_output.action.get_id()
        self.action_dist = self.organism_output.dist_params
        self.option_length = self.hierarchy_info.get_length()
        self.reward = None