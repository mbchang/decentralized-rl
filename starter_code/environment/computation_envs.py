from collections import namedtuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

class ComputationEnv():
    def __init__(self, dataset, loss_fn, max_steps):
        self.dataset = dataset
        self.max_steps = max_steps  # number of actions
        self.loss_fn = loss_fn

    def seed(self, seed):
        pass

    def render(self, mode):
        pass

    @property
    def input_dim(self):
        return self.dataset.in_dim
    
    @property
    def output_dim(self):
        return self.dataset.out_dim
    
    def reset(self):
        self.counter = 0
        inp, out = self.dataset[np.random.choice(len(self.dataset))]
        self.state = inp
        self.target = out
        return self.state

    def step(self, function):
        next_state = function(self.state)
        self.counter += 1
        done = self.counter == self.max_steps
        reward = 0
        self.state = next_state
        return next_state, reward, done, dict()

    def apply_loss(self, state):
        return self.loss_fn(state, self.target)

class VisualComputationEnv(ComputationEnv):

    def render(self, mode):
        if mode == 'rgb_array':
            assert self.state.shape[0] == 1
            rendered_state = self.state.cpu().numpy()
            assert rendered_state.shape[:2] == (1,1)
            rendered_state = np.tile(rendered_state.squeeze(0).transpose(1, 2, 0), (1,1, 3))
            return rendered_state
        else:
            assert False


