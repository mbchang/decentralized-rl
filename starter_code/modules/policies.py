import torch
import torch.nn.functional as F

from mnist.embedded_mnist import MNIST_CNN
from starter_code.modules.base_policy import BaseDiscretePolicy, BaseBetaPolicy, BaseIsotropicGaussianPolicy
from starter_code.modules.networks import MLP, GaussianParams, BetaSoftPlusParams, MinigridCNN, BetaMeanParams


class DiscretePolicy(BaseDiscretePolicy):
    def __init__(self, state_dim, hdim, action_dim):
        super(DiscretePolicy, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.network = MLP(dims=[state_dim, *hdim, action_dim])

    def forward(self, x):
        action_scores = self.network(x)
        action_probs = F.softmax(action_scores, dim=-1)
        return action_probs

    def _adjust_action_dim(self, action):
        assert action.size() == torch.Size([])
        return action

class DiscreteCNNPolicy(BaseDiscretePolicy):
    def __init__(self, state_dim, action_dim):
        super(DiscreteCNNPolicy, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        if self.state_dim == (1, 64, 64):
            self.encoder = MNIST_CNN(self.action_dim)
            self.decoder = lambda x: x
        elif self.state_dim == (7, 7, 3):
            self.encoder = MinigridCNN(*state_dim[:-1])
            self.decoder = MLP(dims=[self.encoder.image_embedding_size, action_dim])
        else:
            assert False

    def forward(self, x):
        action_scores = self.decoder(self.encoder(x))
        action_probs = F.softmax(action_scores, dim=-1)
        return action_probs

    def _adjust_action_dim(self, action):
        if action.shape[0] == 1:
            action = action.squeeze(0)
        assert action.size() == torch.Size([])
        return action

class BetaCNNPolicy(BaseBetaPolicy):
    def __init__(self, state_dim, action_dim):
        super(BetaCNNPolicy, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        if self.state_dim == (1, 64, 64):
            image_embedding_size = 32
            self.encoder = MNIST_CNN(image_embedding_size)
            self.encoder.image_embedding_size = image_embedding_size
        elif self.state_dim == (7, 7, 3):
            self.encoder = MinigridCNN(*state_dim[:-1])
        self.decoder = BetaSoftPlusParams(self.encoder.image_embedding_size, action_dim)

class BetaMeanCNNPolicy(BaseBetaPolicy):
    def __init__(self, state_dim, action_dim):
        super(BetaMeanCNNPolicy, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        if self.state_dim == (1, 64, 64):
            image_embedding_size = 32
            self.encoder = MNIST_CNN(image_embedding_size)
            self.encoder.image_embedding_size = image_embedding_size
        elif self.state_dim == (7, 7, 3):
            self.encoder = MinigridCNN(*state_dim[:-1])
        self.decoder = BetaMeanParams(self.encoder.image_embedding_size, action_dim)

class SimpleBetaSoftPlusPolicy(BaseBetaPolicy):
    def __init__(self, state_dim, hdim, action_dim):
        super(SimpleBetaSoftPlusPolicy, self).__init__()
        self.encoder = MLP(dims=[state_dim, *hdim])
        self.decoder = BetaSoftPlusParams(hdim[-1], action_dim)

class SimpleBetaMeanPolicy(BaseBetaPolicy):
    def __init__(self, state_dim, hdim, action_dim):
        super(SimpleBetaMeanPolicy, self).__init__()
        self.encoder = MLP(dims=[state_dim, *hdim])
        self.decoder = BetaMeanParams(hdim[-1], action_dim)

class IsotropicGaussianPolicy(BaseIsotropicGaussianPolicy):
    def __init__(self, state_dim, hdim, action_dim):
        super(IsotropicGaussianPolicy, self).__init__()
        self.encoder = MLP(dims=[state_dim, *hdim])
        self.decoder = GaussianParams(hdim[-1], action_dim)



