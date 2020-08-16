import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.beta import Beta

from starter_code.infrastructure.utils import to_np
from starter_code.interfaces.interfaces import SphericalMultivariateNormalParams, BetaParams

def dist_bsize(torch_dist):
    batch_shape = torch_dist.batch_shape
    assert len(batch_shape) == 1 or (len(batch_shape) == 2 and batch_shape[-1] == 1)
    return batch_shape[0]

def extract_dist_params(dist):
    if isinstance(dist, Categorical):
        if dist.batch_shape == torch.Size([]):
            return [p.item() for p in dist.probs]
        elif dist.batch_shape == torch.Size([1]):
            return [p.item() for p in dist.probs[0]]
        else:
            assert False
    elif isinstance(dist, Beta):
        if dist.batch_shape == torch.Size([1]):
            return BetaParams(alpha=dist.concentration1.item(), beta=dist.concentration0.item())
        elif dist.batch_shape == torch.Size([1,1]):
            return BetaParams(alpha=dist.concentration1[0].item(), beta=dist.concentration0[0].item())
        else:
            assert False
    elif isinstance(dist, SphericalMultivariateNormal):
        return SphericalMultivariateNormalParams(mu=dist.mu, logstd=dist.logstd)
    else:
        assert False

class SphericalMultivariateNormal(MultivariateNormal):
    def __init__(self, mu, logstd):
        self.mu = mu
        self.logstd = logstd
        MultivariateNormal.__init__(self, loc=mu, scale_tril=torch.diag_embed(torch.exp(logstd)))

class BasePolicy(nn.Module):
    def __init__(self):
        super(BasePolicy, self).__init__()

    def get_log_prob(self, action_dist, action):
        bsize = dist_bsize(action_dist)
        log_prob = action_dist.log_prob(action).view(bsize, 1)
        assert log_prob.size() == (bsize, 1)
        return log_prob

    def get_entropy(self, action_dist):
        bsize = dist_bsize(action_dist)
        entropy = action_dist.entropy().view(bsize, 1)
        assert entropy.size() == (bsize, 1)
        return entropy

class BaseDiscretePolicy(BasePolicy):
    def __init__(self):
        super(BaseDiscretePolicy, self).__init__()
        self.discrete = True

    def get_action_dist(self, obs):
        action_probs = self.forward(obs)
        action_dist = Categorical(action_probs)
        return action_dist

    def get_log_prob(self, action_dist, action):
        bsize = dist_bsize(action_dist)
        action = action.view(bsize)
        return super(BaseDiscretePolicy, self).get_log_prob(action_dist, action)

    def select_action(self, state, deterministic):
        dist = self.get_action_dist(state)
        if deterministic:
            action = torch.argmax(dist.probs, dim=-1)
        else:
            action = dist.sample()
        action = self._adjust_action_dim(action)
        return int(to_np(action)), extract_dist_params(dist)


class BaseContinuousPolicy(BasePolicy):
    def __init__(self):
        super(BaseContinuousPolicy, self).__init__()
        self.discrete = False

    def select_action(self, state, deterministic, reparameterize=False):
        dist = self.get_action_dist(state)
        if reparameterize:
            action = dist.rsample()
        else:
            action = dist.sample()
        return to_np(action), extract_dist_params(dist)

class BaseBetaPolicy(BaseContinuousPolicy):
    def __init__(self):
        super(BaseBetaPolicy, self).__init__()

    def forward(self, x):
        x = self.encoder(x)
        alpha, beta = self.decoder(x)
        return alpha, beta

    def get_action_dist(self, obs):
        alpha, beta = self.forward(obs)
        dist = Beta(concentration1=alpha, concentration0=beta)
        return dist

class BaseIsotropicGaussianPolicy(BaseContinuousPolicy):
    def __init__(self):
        super(BaseIsotropicGaussianPolicy, self).__init__()

    def forward(self, x):
        x = self.encoder(x)
        mu, logstd = self.decoder(x)
        return mu, logstd

    def get_action_dist(self, obs):
        mu, logstd = self.forward(obs)
        dist = SphericalMultivariateNormal(mu=mu, logstd=logstd)
        return dist