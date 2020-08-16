from starter_code.organism.base_agent import BaseRLAgent, BaseHRLAgent

class ActorCriticRLAgent(BaseRLAgent):
    def __init__(self, networks, replay_buffer, args):
        BaseRLAgent.__init__(self, networks, replay_buffer, args)
        self.initialize_optimizer(lrs=dict(policy=self.args.plr, valuefn=self.args.vlr))
        self.initialize_optimizer_schedulers(args)

    def flail(self, env):
        action = env.action_space.sample()
        return CentralizedOutput(action=LiteralActionTransformation(action), dist_params=[1])

    def bundle_networks(self, networks):
        BaseHRLAgent.bundle_networks(self, networks)
        self.valuefn = networks['valuefn']


class ActorCriticHRLAgent(BaseHRLAgent):
    def __init__(self, networks, transformations, replay_buffer, args):
        BaseHRLAgent.__init__(self, networks, transformations, replay_buffer, args)
        self.initialize_optimizer(lrs=dict(policy=self.args.plr, valuefn=self.args.vlr))
        self.initialize_optimizer_schedulers(args)

    def flail(self, env):
        action = np.random.randint(len(self.transformations))
        return CentralizedOutput(action=LiteralActionTransformation(action), dist_params=[1])

    def bundle_networks(self, networks):
        BaseHRLAgent.bundle_networks(self, networks)
        self.valuefn = networks['valuefn']



