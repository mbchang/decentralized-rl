import torch

def preprocess_state_before_store(step):
    # Minigrid
    if isinstance(step.state, tuple):
        step.state = step.state.obs
        step.next_state = step.next_state.obs

    # MNIST
    if isinstance(step.state, torch.Tensor):
        assert isinstance(step.next_state, torch.Tensor)
        if step.state.dim() == 4:
            assert step.state.shape == step.next_state.shape == (1, 1, 64, 64)
            step.state = step.state[0]
            step.next_state = step.next_state[0]
        step.state = step.state.detach().numpy()
        step.next_state = step.next_state.detach().numpy()
    return step

def preprocess_state_before_forward(state):
    if isinstance(state,tuple):
        return state.obs
    else:
        return state