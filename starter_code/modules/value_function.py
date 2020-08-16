import torch
import torch.nn as nn
import torch.nn.functional as F

from starter_code.modules.networks import MLP, MinigridCNN
from mnist.embedded_mnist import MNIST_CNN

class SimpleValueFn(nn.Module):
    def __init__(self, state_dim, hdim):
        super(SimpleValueFn, self).__init__()
        self.value_net = MLP(dims=[state_dim, *hdim, 1])

    def forward(self, state):
        state_values = self.value_net(state)
        return state_values

class CNNValueFn(nn.Module):
    def __init__(self, state_dim):
        super(CNNValueFn, self).__init__()
        self.state_dim = state_dim
        if self.state_dim == (1, 64, 64):
            self.encoder = MNIST_CNN(1)
            self.decoder = lambda x: x
        elif self.state_dim == (7, 7, 3):
            self.encoder = MinigridCNN(*state_dim[:-1])
            self.decoder = nn.Linear(self.encoder.image_embedding_size, 1)
        else:
            assert False

    def forward(self, state):
        state_values = self.decoder(self.encoder(state))
        return state_values