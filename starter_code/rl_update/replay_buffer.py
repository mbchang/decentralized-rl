from collections import namedtuple, deque
import numpy as np
import torch

StoredTransition = namedtuple('StoredTransition',
    ('state', 'action', 'next_state', 'mask', 'reward',
         'start_time', 'end_time',
         'current_transformation_id',
         'next_transformation_id'
         ))

class PathMemory(object):
    def __init__(self, max_replay_buffer_size):
        self._max_replay_buffer_size = max_replay_buffer_size
        self.clear_buffer()

    @property
    def size(self):
        return sum(len(i) for i in self.paths)

    def __len__(self):
        return len(self.paths)

    def clear_buffer(self):
        self.paths = deque()
        self.remaining_length = self._max_replay_buffer_size

    def _evict_paths(self, n):
        excess = n - self.remaining_length
        assert excess > 0

        path_index = 0
        length_of_whole_paths = 0
        while length_of_whole_paths + len(self.paths[path_index]) <= excess:
            length_of_whole_paths += len(self.paths[path_index])
            path_index += 1
            if path_index == len(self.paths):

                assert length_of_whole_paths == n
                break

        assert length_of_whole_paths <= excess

        for i in range(path_index):
            self.paths.popleft()

        self.remaining_length += length_of_whole_paths

    def _evict_transitions(self, n):
        excess = n - self.remaining_length
        assert (excess == 0) or (excess > 0 and excess < len(self.paths[0]))

        for i in range(excess):
            self.paths[0].popleft()

        self.remaining_length += excess
        assert self.remaining_length == n

    def _evict(self, n):
        if n > self.remaining_length:
            self._evict_paths(n)
            self._evict_transitions(n)

    def _safely_add_path(self, path):
        self.paths.append(path)
        self.remaining_length -= len(path)
        assert self.remaining_length >= 0

    def add_path(self, path):
        path = deque(path)
        n = len(path)
        if n > self._max_replay_buffer_size: assert False
        self._evict(n)
        self._safely_add_path(path)

    def sample(self, device):
        """
        self.paths:
            deque(
                deque(      
                    StoredTransition,       # t=0
                    StoredTransition,       # t=1
                    StoredTransition,       # t=2
                ),                      # episode 0
                deque(      
                    StoredTransition,       # t=0
                    StoredTransition,       # t=1
                    StoredTransition,       # t=2
                ),                      # episode 1
            )

        ouput: a StoredTransition tuple with torch tensor entries for each episode:
            state: (eplen_i, *sdim)                 - torch.float32
            action: (eplen_i, adim)                 - torch.float32
            next_state: (eplen_i, *sdim)            - torch.float32
            mask: (eplen_i)                         - torch.float32
            reward: (eplen_i)                       - torch.float32
            start_time: (eplen_i)                   - torch.float32
            end_time: (eplen_i)                     - torch.float32
            current_transformation_id: (eplen_i)    - torch.float32
            next_transformation_id: (eplen_i)       - torch.float32
        """
        path_lengths = [len(path) for path in self.paths]
        everything = StoredTransition(*zip(*[StoredTransition(*zip(*path)) for path in self.paths]))
        torch_concatenate = lambda list_of_lists: torch.from_numpy(np.concatenate([np.stack(lst) for lst in list_of_lists])).to(torch.float32).to(device)
        stacked_paths = StoredTransition(*map(torch_concatenate, everything))
        return stacked_paths, path_lengths
