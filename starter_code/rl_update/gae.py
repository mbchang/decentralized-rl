from collections import namedtuple
import torch

from starter_code.infrastructure.utils import to_device

SupervisionSignal = namedtuple('SupervisionSignal', 
    ('rewards', 'masks', 'values', 'last_value', 'start_times', 'end_times'))

def adjust_gamma(start_times, end_times, gamma):
    """
        Standard RL: end_time - start_time = 1
        Hierarchical RL: end_time - start_time = number of steps the subpolicy took
    """
    T = start_times.size(0)
    adjusted_gammas = []
    for i in range(T):
        adjusted_gamma = gamma ** (end_times[i]-start_times[i])
        adjusted_gammas.append(adjusted_gamma)
    return adjusted_gammas

def compute_deltas(rewards, values, masks, last_value, adjusted_gammas):
    T = rewards.size(0)
    deltas = torch.empty_like(values)
    values = torch.cat((values, last_value))
    assert values.shape == torch.Size([T+1, 1])
    for i in range(T):
        deltas[i] = rewards[i] + adjusted_gammas[i] * values[i+1] * masks[i] - values[i]
    return deltas

def smooth_advantages(deltas, masks, adjusted_gammas, tau):
    advantages = torch.empty_like(deltas)
    prev_advantage = 0
    for i in reversed(range(deltas.size(0))):
        advantages[i] = deltas[i] + adjusted_gammas[i] * tau * prev_advantage * masks[i]
        prev_advantage = advantages[i, 0]
    return advantages

def estimate_advantages(supervision_signal, gamma, tau, device):
    rewards, masks, values, last_value, start_times, end_times = to_device(
        torch.device('cpu'), *supervision_signal)
    adjusted_gammas = adjust_gamma(start_times, end_times, gamma)
    deltas = compute_deltas(rewards, values, masks, last_value, adjusted_gammas)
    advantages = smooth_advantages(deltas, masks, adjusted_gammas, tau)
    returns = values + advantages
    advantages, returns = to_device(device, advantages, returns)
    return advantages, returns

def estimate_advantages_by_episode(paths, gamma, tau, device):
    advantages = []
    returns = []
    for episode_data in paths:
        a, r = estimate_advantages(
            SupervisionSignal(
                rewards=episode_data.reward, 
                masks=episode_data.mask, 
                values=episode_data.value, 
                last_value=episode_data.last_value,
                start_times=episode_data.start_time, 
                end_times=episode_data.end_time,
                ),
            gamma=gamma, 
            tau=tau, 
            device=device)
        advantages.append(a)
        returns.append(r)
    advantages = torch.cat(advantages)
    returns = torch.cat(returns)
    return advantages, returns


