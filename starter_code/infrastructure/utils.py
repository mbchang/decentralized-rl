import copy
import math
import numpy as np
import torch
from operator import itemgetter

def to_device(device, *args):
    return [x.to(device) for x in args]

def visualize_parameters(model, pfunc):
    for n, p in model.named_parameters():
        if p.grad is None:
            pfunc('{}\t{}\t{}'.format(n, p.data.norm(), None))
        else:
            pfunc('{}\t{}\t{}'.format(n, p.data.norm(), p.grad.data.norm()))

def from_onehot(state, state_dim):
    state_id = np.argmax(state)
    if state_id == state_dim-1:
        state_id = -1
    return state_id

class AttrDict(dict):
  __getattr__ = dict.__getitem__
  __setattr__ = dict.__setitem__    

def from_np(np_array, device):
    return torch.tensor(np_array).float().to(device)

def to_np(tensor):
    return tensor.detach().cpu().numpy()

def is_float(n):
    return isinstance(n, float) or isinstance(n, np.float64) or isinstance(n, np.float32)

def all_same(items):
    return all(x == items[0] for x in items)