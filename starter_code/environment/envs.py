import itertools
import numpy as np
import pprint
# from starter_code.environment.env_tester import tabular_tester

class OneHotEnv(object):
    def __init__(self):
        super(OneHotEnv, self).__init__()

    @property
    def state_dim(self):
        return len(self.states)+1

    def to_onehot(self, state_id):
        state = np.zeros(self.state_dim)
        state[state_id] = 1
        return state

    def from_onehot(self, state):
        state_id = np.argmax(state)
        return state_id

class AtMostTwoHotEnv(object):
    def __init__(self):
        super(AtMostTwoHotEnv, self).__init__()

    @property
    def state_dim(self):
        return len(self.states)+len(self.keys)

    def to_twohot(self, state_key_id):
        assert len(state_key_id)==2
        state = np.zeros(self.state_dim)
        state[state_key_id[0]] = 1
        if state_key_id[1]!=-1:
            state[len(self.states)+state_key_id[1]]=1

        return state

    def from_twohot(self, state):
        state_id = np.argmax(state[:len(self.states)])
        key_id= np.argmax(state[len(self.states):])
        if 1 not in state[len(self.states):]:
            key_id=-1
        return (state_id,key_id)
    #For env_tester, same as from_twohot
    def from_onehot(self, state):
        return self.from_twohot(state)

def not_in_array(np_array, value):
    return not np.all(np.equal(np_array, value))


class GridWorldK(object):
    def __init__(self, height, width, goal, step_reward=-0.1):
        self.height = height
        self.width = width
        self.num_states = self.height * self.width
        self.goal_idx = self.get_goal_idx(goal)
        self.goal_state = (self.goal_idx // self.width, self.goal_idx % self.width)

        self.grid_states = self.create_grid_states()
        self.states = self.grid_states.flatten()
        self.actions = self.get_actions()#{0: 'L', 1: 'R', 2: 'D', 3: 'U'}
        self.transitions = self.create_transitions()

        self.step_reward = step_reward#-0.005#-0.1  # NOTE THAT THIS IS SPARSE REWARD!
        self.goal_reward = 0.8  # note that for GW2 I use 0.5 here
        self.rewards = self.create_rewards()

        print('grid_states', self.grid_states)
        print('states', self.states)
        print('action', self.actions)
        print('transitions', self.transitions)
        print('rewards', self.rewards)

        self.gamma = 1
        self.Qs = self.Q_iteration_deterministic()

    def get_actions(self):
        return {0: 'L', 1: 'R', 2: 'D', 3: 'U'}

    def get_goal_idx(self, goal):
        if goal is None:
            goal_idx = np.random.randint(self.num_states)
        elif goal == -1:
            goal_idx = self.num_states -1
        else:
            goal_idx = goal
        return goal_idx

    def create_grid_states(self):
        grid = np.arange(self.num_states, dtype=np.int64).reshape(self.height, self.width)
        grid[self.goal_state[0], self.goal_state[1]] = self.goal_idx  # goal
        return grid

    def create_transitions(self):
        transition_matrix = np.ones((len(self.actions), self.num_states), dtype=np.int64)*-2  # default value
        print(transition_matrix)
        for row_idx in range(self.height):
            for col_idx in range(self.width):
                state_id = self.grid_states[row_idx, col_idx]
                for action_id in self.actions:
                    if action_id == 0:  # L
                        if  col_idx == 0:
                            transition_matrix[action_id, state_id] = state_id  # don't move
                        else:
                            transition_matrix[action_id, state_id] = self.grid_states[row_idx, col_idx-1]  # left by one column
                    elif action_id == 1:  # R
                        if col_idx == self.width-1:
                            transition_matrix[action_id, state_id] = state_id  # don't move
                        else:
                            transition_matrix[action_id, state_id] = self.grid_states[row_idx, col_idx+1]  # right by one column
                    elif action_id == 2:  # D
                        if row_idx == self.height-1:
                            transition_matrix[action_id, state_id] = state_id  # don't move
                        else:
                            transition_matrix[action_id, state_id] = self.grid_states[row_idx+1, col_idx]  # down by one column
                    elif action_id == 3:   # U
                        if row_idx == 0:
                            transition_matrix[action_id, state_id] = state_id  # don't move
                        else:
                            transition_matrix[action_id, state_id] = self.grid_states[row_idx-1, col_idx]  # up by one column
                    else:
                        assert False

        # the goal state should transition to itself
        transition_matrix[:, self.goal_idx] = self.goal_idx
        assert not_in_array(transition_matrix, -2)
        return transition_matrix

    def create_rewards(self):
        reward_matrix = np.zeros((len(self.actions), self.num_states))
        for action_id in self.actions:
            for state_id in self.states:
                if self.transitions[action_id, state_id] == self.goal_idx:
                    reward_matrix[action_id, state_id] = self.goal_reward
                else:
                    reward_matrix[action_id, state_id] = self.step_reward
        reward_matrix[:, self.goal_idx] = 0
        return reward_matrix

    # this should be fine
    def Q_iteration_deterministic(self):
        Q_old = np.zeros((len(self.actions),self.num_states))
        Q_new = np.zeros((len(self.actions),self.num_states))

        i = 0
        while True:
            print('Q at iteration {}: \n{}'.format(i, Q_old))
            for s in self.states:
                for a in self.actions:
                    s_prime = self.transitions[a][s]
                    newQ = self.rewards[a][s] + self.gamma * np.max(Q_old[:, s_prime])
                    Q_new[a, s] = newQ
            if np.array_equal(Q_new, Q_old):
                print('Q converged to \n{} in {} iterations'.format(Q_new, i))
                break
            Q_old = np.copy(Q_new)
            i += 1
        return Q_old


def convert_grid_to_dict(grid):
    d = {}
    for action_idx in range(len(grid)):
        for state_idx in range(len(grid[action_idx])):
            reported_state_idx = state_idx
            d[reported_state_idx, action_idx] = grid[action_idx][state_idx]
    return d

class MultiStepEnv(OneHotEnv):
    def __init__(self):
        super(MultiStepEnv, self).__init__()
        self.states = []
        self.actions = []
        self.transitions = {}
        self.rewards = {}
        self.Qs = {}
        self.gamma = 1

    def getQ(self, state, action):
        state = self.from_onehot(state)
        return self.Qs[(state, action)]

    def reset(self):
        self.counter = 0
        start_state = np.random.choice(self.starting_states)
        self.state = start_state  # scalar
        return self.to_onehot(start_state)

    def step(self, action):
        next_state = self.transitions[self.state, action]
        self.counter += 1
        done = self.counter == self.eplen
        reward = self._reward(self.state, action)
        self.state = next_state
        return self.to_onehot(next_state), reward, done, {}

    def _reward(self, state, action):
        return self.rewards[self.state, action]

    def render(self, mode):
        pass

    def close(self):
        pass

    def seed(self, seed):
        pass


class MultiStepEnvTwoHot(AtMostTwoHotEnv):
    def __init__(self):
        super(MultiStepEnvTwoHot, self).__init__()
        self.states = []
        self.actions = []
        self.transitions = {}
        self.rewards = {}
        self.Qs = {}
        self.gamma = 1

    def getQ(self, state, action):
        state = self.from_onehot(state)
        return self.Qs[(state, action)]

    def reset(self):
        self.counter = 0
        start_state = np.random.randint(0,len(self.starting_states))
        self.state = self.starting_states[start_state]  # scalar
        return self.to_twohot(self.state)

    def step(self, action):
        next_state = self.transitions[self.state, action]
        self.counter += 1
        done = self.counter == self.eplen
        reward = self._reward(self.state, action)
        self.state = next_state
        return self.to_twohot(next_state), reward, done, {}

    def _reward(self, state, action):
        return self.rewards[self.state, action]

    def render(self, mode):
        pass

    def close(self):
        pass

    def seed(self, seed):
        pass

class OneHotGridWorldK(GridWorldK, OneHotEnv):
    def __init__(self, height, width, eplencoeff, step_reward, rand_init=False, goal=-1):
        super(OneHotGridWorldK, self).__init__(
            height=height, width=width, goal=goal, step_reward=step_reward)
        self.actions = list(self.actions.keys())
        self.transitions = convert_grid_to_dict(self.transitions)
        self.rewards = convert_grid_to_dict(self.rewards)
        self.Qs = convert_grid_to_dict(self.Qs)
        self.eplen = eplencoeff * (height-1 + width - 1)

        self.goal_states = [self.goal_idx]
        self.starting_states = self.get_initial_states(rand_init, self.goal_states)

    def get_initial_states(self, rand_init, goal_states):
        if rand_init:
            starting_states = [x for x in self.states if x not in goal_states]
        else:
            starting_states = [0]
        return starting_states
    
    def getQ(self, state, action):
        state = self.from_onehot(state)
        return self.Qs[(state, action)]

    def reset(self):
        self.counter = 0
        start_state = np.random.choice(self.starting_states)
        self.state = start_state
        return self.to_onehot(start_state)

    def step(self, action):
        next_state = self.transitions[self.state, action]
        self.counter += 1
        done = next_state in self.goal_states or self.counter == self.eplen
        reward = self.rewards[self.state, action]
        self.state = next_state
        return self.to_onehot(next_state), reward, done, {}

    def render(self, mode):
        pass

    def close(self):
        pass

    def seed(self, seed):
        pass


class OneHotChainK(OneHotGridWorldK):
    def __init__(self, length):
        super(OneHotChainK, self).__init__(
            height=1,
            width=length, 
            eplencoeff=4, 
            step_reward=0,
            )

    def get_actions(self):
        return {0: 'L', 1: 'R'}


class OneStateOneStepKActionEnv(MultiStepEnv):
    def __init__(self, k):
        super(OneStateOneStepKActionEnv, self).__init__()
        self.states = [0]
        self.actions = list(range(k))
        self.transitions = {(0, kk): -1 for kk in range(k)}
        self.rewards = {(0, kk): (kk+1)*1.0/(k+1) for kk in range(k)}
        self.Qs = {**{(0, kk): (kk+1)*1.0/(k+1) for kk in range(k)},
                   **{(-1, kk): 0 for kk in range(k)}}
        self.eplen = 1
        self.starting_states = [0]

class OneStateOneStepKActionEnvTransfer(MultiStepEnv):
    def __init__(self, k):
        super(OneStateOneStepKActionEnvTransfer, self).__init__()
        self.states = [0]
        self.actions = list(range(k))
        self.transitions = {(0, kk): -1 for kk in range(k)}
        # self.rewards = {(0, kk): (kk+1)*1.0/(k+1) for kk in range(k)}
        self.rewards={(0,0):0.8,
                      (0,1):0.4,
                      (0,2):0.6,
                      (0,3):0.2}
        self.Qs = {**{(0, kk): (kk+1)*1.0/(k+1) for kk in range(k)},
                   **{(-1, kk): 0 for kk in range(k)}}
        self.eplen = 1
        self.starting_states = [0]


class Duality(MultiStepEnv):
    def __init__(self, absorbing_reward=0, asymmetric_coeff=1, big_reward=0.8, small_reward=0.1):
        super(Duality, self).__init__()
        big = big_reward
        small = small_reward

        self.eplen = 10
        normalizing_coeff = 1.0

        big = big/normalizing_coeff  # reward scale
        small = small/normalizing_coeff

        self.states = [0, 1]
        self.actions = [0, 1]
        self.transitions = {
            # 2 is an absorbing state
            (2, 0): 2,
            (2, 1): 2,

            (0, 0): 2,
            (0, 1): 1,
            (1, 0): 0,
            (1, 1): 1,
        }
        self.rewards = {
            (2, 0): absorbing_reward,
            (2, 1): asymmetric_coeff*absorbing_reward,  # for asymmetry
            (0, 0): 0,
            (0, 1): big,
            (1, 0): big,
            (1, 1): small
        }
        self.starting_states = [0]

    def step(self, action):
        next_state, reward, done, _ = MultiStepEnv.step(self, action)
        return next_state, reward, False, _



class OneHotSquareGridWorldK(OneHotGridWorldK):
    def __init__(self, size, rand_init=False, goal=-1, eplencoeff=4, step_reward=-0.1):
        super(OneHotSquareGridWorldK, self).__init__(
            height=size,
            width=size,
            rand_init=rand_init,
            goal=goal,
            eplencoeff=eplencoeff,
            step_reward=step_reward
            )



class StochasticOneStateOneStepKActionEnv(MultiStepEnv):
    def __init__(self, k):
        super(StochasticOneStateOneStepKActionEnv, self).__init__()
        self.k = k
        self.states = [0]
        self.actions = list(range(k))
        self.transitions = {(0, kk): -1 for kk in range(k)}
        self.rewards = {(0, kk): (kk+1)*1.0/(k+1) for kk in range(k)}
        self.Qs = {**{(0, kk): (kk+1)*1.0/(k+1) for kk in range(k)},
                   **{(-1, kk): 0 for kk in range(k)}}
        self.eplen = 1
        self.starting_states = [0]

    def _reward(self, state, action):
        p = self.rewards[self.state, action]
        n = 10  # n controls the concentration (larger n is more concentrated)
        sample = np.random.binomial(n, p)/float(n)
        return sample

class BernoulliStochasticOneStateOneStepKActionEnv(MultiStepEnv):
    def __init__(self, k):
        super(BernoulliStochasticOneStateOneStepKActionEnv, self).__init__()
        self.k = k
        self.states = [0]
        self.actions = list(range(k))
        self.transitions = {(0, kk): -1 for kk in range(k)}
        self.rewards = {(0, kk): (kk+1)*1.0/(k+1) for kk in range(k)}
        self.Qs = {**{(0, kk): (kk+1)*1.0/(k+1) for kk in range(k)},
                   **{(-1, kk): 0 for kk in range(k)}}
        self.eplen = 1
        self.starting_states = [0]

    def _reward(self, state, action):
        p = self.rewards[self.state, action]
        if np.random.uniform() < p:
            sample = 1
        else:
            sample = 0
        return sample


class CounterExample1Env(MultiStepEnv):
    """
        infinite-horizon
        0, 0 --> -1: 0
        0, 1 --> 1: 0.9
        1, 0 --> 0: 0.9
        1, 1 --> 1: 0.3
        Theoretically it shouldn't work.
    """
    def __init__(self, absorbing_reward=0, asymmetric_coeff=1, big_reward=0.8, small_reward=0.1):
        super(CounterExample1Env, self).__init__()
        big = big_reward
        small = small_reward

        self.eplen = 10
        normalizing_coeff = 1.0

        big = big/normalizing_coeff  # reward scale
        small = small/normalizing_coeff

        self.states = [0, 1]
        self.actions = [0, 1]
        self.transitions = {
            # -1 is an absorbing state
            (2, 0): 2,
            (2, 1): 2,

            (0, 0): 2,
            (0, 1): 1,
            (1, 0): 0,
            (1, 1): 1,
        }
        self.rewards = {
            (2, 0): absorbing_reward,
            (2, 1): asymmetric_coeff*absorbing_reward,  # for asymmetry
            (0, 0): 0,
            (0, 1): big,
            (1, 0): big,
            (1, 1): small
        }
        # Qs are actually not correct
        self.Qs = {
            (0, 0): 0,
            (0, 1): 0,
            (1, 0): 0,
            (1, 1): 0,

            (2, 0): 0,
            (2, 1): 0,
        }
        self.starting_states = [0]

    def step(self, action):
        next_state, reward, done, _ = MultiStepEnv.step(self, action)
        return next_state, reward, False, _


def test_CounterExample1Env():
    action_sequences = {
        (0, 0, 0, 0, 0, 0): [0, 2, 2, 2, 2, 2, 2],
        (0, 1, 1, 1, 1, 1): [0, 2, 2, 2, 2, 2, 2],
        (0, 1, 0, 1, 0, 1): [0, 2, 2, 2, 2, 2, 2],
        (1, 0, 0, 0, 0, 0): [0, 1, 0, 2, 2, 2, 2],
        (1, 0, 1, 0, 1, 0): [0, 1, 0, 1, 0, 1, 0],
        (1, 1, 1, 1, 1, 1): [0, 1, 1, 1, 1, 1, 1],
        }

    for action_sequence in action_sequences:
        env = CounterExample1Env()
        states = []
        rewards = []
        dones = []

        state = env.reset()
        states.append(state)

        for action in action_sequence:
            next_state, reward, done, _ = env.step(action)
            states.append(next_state)
            rewards.append(reward)
            dones.append(done)

        print('Action Sequence: {}'.format(action_sequence))
        print('Expected State Sequence: {}'.format(action_sequences[action_sequence]))
        print('Actual State Sequence: {}'.format(action_sequences[action_sequence]))
        print('Rewards: {}'.format(rewards))
        print('Dones: {}'.format(dones))


class MultiTaskDebug(MultiStepEnvTwoHot):
    def __init__(self):
        MultiStepEnvTwoHot.__init__(self)
        self.big_reward = 0.8
        self.eplen = 10

        self.states = [0, 1, 2, 3]  # 8 is an absorbing state
        self.keys=[0,1,2,3,4,5]
        self.actions = [0, 1, 2, 3, 4, 5] # 0->A , 1->B, 2->C, 3->D, 4->E, 5->F
        # Qs are actually not correct



    def step(self, action):
        next_state = self.transitions[self.state, action]
        self.counter += 1
        done = next_state in self.terminal_states
        reward = self.rewards[self.state, action]
        self.state = next_state
        return self.to_twohot(next_state), reward, done, {}


class MultiTaskDebugTaskABC(MultiTaskDebug):
    def __init__(self):
        MultiTaskDebug.__init__(self)
        self.states = [0, 1, 2, 3]  # 8 is an absorbing state
        self.keys=[0,1,2,3,4,5]
        self.actions = [0, 1, 2] # 0->A , 1->B, 2->C, 3->D, 4->E, 5->F
        self.transitions = {
            # absorbing state
            ((4,-1), 0): (4,-1),
            ((4,-1), 1): (4,-1),
            ((4,-1), 2): (4,-1),


            ((0,0), 0): (1,2),
            ((0,0), 1): (0,0),
            ((0,0), 2): (0,0),

            ((1,2), 0): (1,2),
            ((1,2), 1): (2,-1),
            ((1,2), 2): (1,2),

            ((2,-1), 0): (2,-1),
            ((2,-1), 1): (2,-1),
            ((2,-1), 2): (3,4),

            ((3,4), 0): (4,-1),
            ((3,4), 1): (4,-1),
            ((3,4), 2): (4,-1),


        }
        # Qs are actually not correct
        self.Qs = {
            ((4,-1), 0): 0,
            ((4,-1), 1): 0,
            ((4,-1), 2): 0,


            ((0,0), 0): 0,
            ((0,0), 1): 0,
            ((0,0), 2): 0,

            ((1,2), 0): 0,
            ((1,2), 1): 0,
            ((1,2), 2): 0,

            ((2,-1), 0): 0,
            ((2,-1), 1): 0,
            ((2,-1), 2): 0,

            ((3,4), 0): 0,
            ((3,4), 1): 0,
            ((3,4), 2): 0,

        }
        self.rewards = {
    # absorbing state gets no reward
            ((4,-1), 0): 0,
            ((4,-1), 1): 0,
            ((4,-1), 2): 0,


            ((0,0), 0): 0,
            ((0,0), 1): 0,
            ((0,0), 2): 0,

            ((1,2), 0): 0,
            ((1,2), 1): 0,
            ((1,2), 2): 0,

            ((2,-1), 0): 0,
            ((2,-1), 1): 0,
            ((2,-1), 2): self.big_reward,

            ((3,4), 0): 0,
            ((3,4), 1): 0,
            ((3,4), 2): 0,

        }
        self.starting_states = [(0,0)]
        self.terminal_states = [(3,4),(4,-1)]

class MultiTaskDebugCommonDecendant(MultiTaskDebug):
    def __init__(self):
        MultiTaskDebug.__init__(self)
        self.states = [0, 1, 2, 3]  # 8 is an absorbing state
        self.keys=[0,1,2,3,4,5]
        self.actions = [0, 1, 2, 3, 4, 5] # 0->A , 1->B, 2->C, 3->D, 4->E, 5->F
        self.transitions = {
            # absorbing state
            ((4,-1), 0): (4,-1),
            ((4,-1), 1): (4,-1),
            ((4,-1), 2): (4,-1),
            ((4,-1), 3): (4,-1),
            ((4,-1), 4): (4,-1),
            ((4,-1), 5): (4,-1),


            ((0,0), 0): (0,0),
            ((0,0), 1): (0,0),
            ((0,0), 2): (0,0),
            ((0,0), 3): (0,0),
            ((0,0), 4): (0,0),
            ((0,0), 5): (0,0),

            ((0,1), 0): (0,1),
            ((0,1), 1): (0,1),
            ((0,1), 2): (0,1),
            ((0,1), 3): (0,1),
            ((0,1), 4): (0,1),
            ((0,1), 5): (0,1),


            ((1,3), 0): (1,3),
            ((1,3), 1): (1,3),
            ((1,3), 2): (1,3),
            ((1,3), 3): (1,3),
            ((1,3), 4): (1,3),
            ((1,3), 5): (1,3),

            ((1,4), 0): (1,4),
            ((1,4), 1): (1,4),
            ((1,4), 2): (1,4),
            ((1,4), 3): (1,4),
            ((1,4), 4): (1,4),
            ((1,4), 5): (1,4),

            ((2,2), 0): (2,2),
            ((2,2), 1): (2,2),
            ((2,2), 2): (3,2),
            ((2,2), 3): (2,2),
            ((2,2), 4): (2,2),
            ((2,2), 5): (2,2),

            ((2,5), 0): (2,5),
            ((2,5), 1): (2,5),
            ((2,5), 2): (2,5),
            ((2,5), 3): (2,5),
            ((2,5), 4): (2,5),
            ((2,5), 5): (3,5),

            ((3,2), 0): (4,-1),
            ((3,2), 1): (4,-1),
            ((3,2), 2): (4,-1),
            ((3,2), 3): (4,-1),
            ((3,2), 4): (4,-1),
            ((3,2), 5): (4,-1),

            ((3,5), 0): (4,-1),
            ((3,5), 1): (4,-1),
            ((3,5), 2): (4,-1),
            ((3,5), 3): (4,-1),
            ((3,5), 4): (4,-1),
            ((3,5), 5): (4,-1),

        }
        # Qs are actually not correct
        self.Qs = {
            ((4,-1), 0): 0,
            ((4,-1), 1): 0,
            ((4,-1), 2): 0,
            ((4,-1), 3): 0,
            ((4,-1), 4): 0,
            ((4,-1), 5): 0,


            ((0,0), 0): 0,
            ((0,0), 1): 0,
            ((0,0), 2): 0,
            ((0,0), 3): 0,
            ((0,0), 4): 0,
            ((0,0), 5): 0,

            ((0,1), 0): 0,
            ((0,1), 1): 0,
            ((0,1), 2): 0,
            ((0,1), 3): 0,
            ((0,1), 4): 0,
            ((0,1), 5): 0,


            ((1,3), 0): 0,
            ((1,3), 1): 0,
            ((1,3), 2): 0,
            ((1,3), 3): 0,
            ((1,3), 4): 0,
            ((1,3), 5): 0,

            ((1,4), 0): 0,
            ((1,4), 1): 0,
            ((1,4), 2): 0,
            ((1,4), 3): 0,
            ((1,4), 4): 0,
            ((1,4), 5): 0,

            ((2,2), 0): 0,
            ((2,2), 1): 0,
            ((2,2), 2): 0,
            ((2,2), 3): 0,
            ((2,2), 4): 0,
            ((2,2), 5): 0,

            ((2,5), 0): 0,
            ((2,5), 1): 0,
            ((2,5), 2): 0,
            ((2,5), 3): 0,
            ((2,5), 4): 0,
            ((2,5), 5): 0,

            ((3,2), 0): 0,
            ((3,2), 1): 0,
            ((3,2), 2): 0,
            ((3,2), 3): 0,
            ((3,2), 4): 0,
            ((3,2), 5): 0,

            ((3,5), 0): 0,
            ((3,5), 1): 0,
            ((3,5), 2): 0,
            ((3,5), 3): 0,
            ((3,5), 4): 0,
            ((3,5), 5): 0,

        }
        self.rewards = {
    # absorbing state gets no reward
            ((4,-1), 0): 0,
            ((4,-1), 1): 0,
            ((4,-1), 2): 0,
            ((4,-1), 3): 0,
            ((4,-1), 4): 0,
            ((4,-1), 5): 0,


            ((0,0), 0): 0,
            ((0,0), 1): 0,
            ((0,0), 2): 0,
            ((0,0), 3): 0,
            ((0,0), 4): 0,
            ((0,0), 5): 0,

            ((0,1), 0): 0,
            ((0,1), 1): 0,
            ((0,1), 2): 0,
            ((0,1), 3): 0,
            ((0,1), 4): 0,
            ((0,1), 5): 0,

            ((1,3), 0): 0,
            ((1,3), 1): 0,
            ((1,3), 2): 0,
            ((1,3), 3): 0,
            ((1,3), 4): 0,
            ((1,3), 5): 0,

            ((1,4), 0): 0,
            ((1,4), 1): 0,
            ((1,4), 2): 0,
            ((1,4), 3): 0,
            ((1,4), 4): 0,
            ((1,4), 5): 0,

            ((2,2), 0): 0,
            ((2,2), 1): 0,
            ((2,2), 2): 0,
            ((2,2), 3): 0,
            ((2,2), 4): 0,
            ((2,2), 5): 0,

            ((2,5), 0): 0,
            ((2,5), 1): 0,
            ((2,5), 2): 0,
            ((2,5), 3): 0,
            ((2,5), 4): 0,
            ((2,5), 5): 0,

            ((3,2), 0): 0,
            ((3,2), 1): 0,
            ((3,2), 2): 0,
            ((3,2), 3): 0,
            ((3,2), 4): 0,
            ((3,2), 5): 0,

            ((3,5), 0): 0,
            ((3,5), 1): 0,
            ((3,5), 2): 0,
            ((3,5), 3): 0,
            ((3,5), 4): 0,
            ((3,5), 5): 0,

        }
        self.starting_states = [(0,0)]
        self.terminal_states = [(2,2),(4,-1)]

class MultiTaskDebugCommonDecendantAC(MultiTaskDebugCommonDecendant):
    def __init__(self):
        MultiTaskDebugCommonDecendant.__init__(self)
        self.states = [0, 1, 2, 3]  # 8 is an absorbing state
        self.keys=[0,1,2,3,4,5]
        self.actions = [0, 1, 2, 3, 4, 5] # 0->A , 1->B, 2->C, 3->D, 4->E, 5->F


        self.transitions[((0,0), 0)]= (2,2)
        self.transitions[((0,1), 1)]=(2,2)
        self.transitions[((1,3), 3)]= (2,2)
        self.transitions[((1,4), 4)]= (2,2)


        self.rewards[((2,2), 2)]= self.big_reward
        self.starting_states = [(0,0)]
        self.terminal_states = [(3,2),(4,-1)]

class MultiTaskDebugCommonDecendantBC(MultiTaskDebugCommonDecendant):
    def __init__(self):
        MultiTaskDebugCommonDecendant.__init__(self)
        self.states = [0, 1, 2, 3]  # 8 is an absorbing state
        self.keys=[0,1,2,3,4,5]
        self.actions = [0, 1, 2, 3, 4, 5] # 0->A , 1->B, 2->C, 3->D, 4->E, 5->F


        self.transitions[((0,0), 0)]= (2,2)
        self.transitions[((0,1), 1)]=(2,2)
        self.transitions[((1,3), 3)]= (2,2)
        self.transitions[((1,4), 4)]= (2,2)


        self.rewards[((2,2), 2)]= self.big_reward
        self.starting_states = [(0,1)]
        self.terminal_states = [(3,2),(4,-1)]

class MultiTaskDebugCommonDecendantDC(MultiTaskDebugCommonDecendant):
    def __init__(self):
        MultiTaskDebugCommonDecendant.__init__(self)
        self.states = [0, 1, 2, 3]  # 8 is an absorbing state
        self.keys=[0,1,2,3,4,5]
        self.actions = [0, 1, 2, 3, 4, 5] # 0->A , 1->B, 2->C, 3->D, 4->E, 5->F


        self.transitions[((0,0), 0)]= (2,2)
        self.transitions[((0,1), 1)]=(2,2)
        self.transitions[((1,3), 3)]= (2,2)
        self.transitions[((1,4), 4)]= (2,2)


        self.rewards[((2,2), 2)]= self.big_reward
        self.starting_states = [(1,3)]
        self.terminal_states = [(3,2),(4,-1)]

class MultiTaskDebugCommonDecendantEC(MultiTaskDebugCommonDecendant):
    def __init__(self):
        MultiTaskDebugCommonDecendant.__init__(self)
        self.states = [0, 1, 2, 3]  # 8 is an absorbing state
        self.keys=[0,1,2,3,4,5]
        self.actions = [0, 1, 2, 3, 4, 5] # 0->A , 1->B, 2->C, 3->D, 4->E, 5->F


        self.transitions[((0,0), 0)]= (2,2)
        self.transitions[((0,1), 1)]=(2,2)
        self.transitions[((1,3), 3)]= (2,2)
        self.transitions[((1,4), 4)]= (2,2)


        self.rewards[((2,2), 2)]= self.big_reward
        self.starting_states = [(1,4)]
        self.terminal_states = [(3,2),(4,-1)]

class MultiTaskDebugCommonDecendantAF(MultiTaskDebugCommonDecendant):
    def __init__(self):
        MultiTaskDebugCommonDecendant.__init__(self)
        self.states = [0, 1, 2, 3]  # 8 is an absorbing state
        self.keys=[0,1,2,3,4,5]
        self.actions = [0, 1, 2, 3, 4, 5] # 0->A , 1->B, 2->C, 3->D, 4->E, 5->F


        self.transitions[((0,0), 0)]= (2,5)
        self.transitions[((0,1), 1)]=(2,5)
        self.transitions[((1,3), 3)]= (2,5)
        self.transitions[((1,4), 4)]= (2,5)


        self.rewards[((2,5), 5)]= self.big_reward
        self.starting_states = [(0,0)]
        self.terminal_states = [(3,5),(4,-1)]

class MultiTaskDebugCommonDecendantBF(MultiTaskDebugCommonDecendant):
    def __init__(self):
        MultiTaskDebugCommonDecendant.__init__(self)
        self.states = [0, 1, 2, 3]  # 8 is an absorbing state
        self.keys=[0,1,2,3,4,5]
        self.actions = [0, 1, 2, 3, 4, 5] # 0->A , 1->B, 2->C, 3->D, 4->E, 5->F


        self.transitions[((0,0), 0)]= (2,5)
        self.transitions[((0,1), 1)]=(2,5)
        self.transitions[((1,3), 3)]= (2,5)
        self.transitions[((1,4), 4)]= (2,5)


        self.rewards[((2,5), 5)]= self.big_reward
        self.starting_states = [(0,1)]
        self.terminal_states = [(3,5),(4,-1)]




class MultiTaskDebugCommonAncestorAB(MultiTaskDebug):
    def __init__(self):
        MultiTaskDebug.__init__(self)
        self.states = [0, 1, 2, 3]  # 8 is an absorbing state
        self.keys=[0,1,2,3,4,5]
        self.actions = [0, 1, 2, 3, 4, 5] # 0->A , 1->B, 2->C, 3->D, 4->E, 5->F
        self.transitions = {
            # absorbing state
            ((4,-1), 0): (4,-1),
            ((4,-1), 1): (4,-1),
            ((4,-1), 2): (4,-1),
            ((4,-1), 3): (4,-1),
            ((4,-1), 4): (4,-1),
            ((4,-1), 5): (4,-1),


            ((0,0), 0): (1,2),
            ((0,0), 1): (0,0),
            ((0,0), 2): (0,0),
            ((0,0), 3): (0,0),
            ((0,0), 4): (0,0),
            ((0,0), 5): (0,0),

            ((0,1), 0): (0,1),
            ((0,1), 1): (1,2),
            ((0,1), 2): (0,1),
            ((0,1), 3): (0,1),
            ((0,1), 4): (0,1),
            ((0,1), 5): (0,1),

            ((1,2), 0): (1,2),
            ((1,2), 1): (1,2),
            ((1,2), 2): (2,2),
            ((1,2), 3): (1,2),
            ((1,2), 4): (1,2),
            ((1,2), 5): (1,2),

            ((1,3), 0): (1,3),
            ((1,3), 1): (1,3),
            ((1,3), 2): (1,3),
            ((1,3), 3): (1,3),
            ((1,3), 4): (1,3),
            ((1,3), 5): (1,3),

            ((1,4), 0): (1,4),
            ((1,4), 1): (1,4),
            ((1,4), 2): (1,4),
            ((1,4), 3): (1,4),
            ((1,4), 4): (1,4),
            ((1,4), 5): (1,4),

            ((1,5), 0): (1,5),
            ((1,5), 1): (1,5),
            ((1,5), 2): (1,5),
            ((1,5), 3): (1,5),
            ((1,5), 4): (1,5),
            ((1,5), 5): (1,5),

            ((2,2), 0): (4,-1),
            ((2,2), 1): (4,-1),
            ((2,2), 2): (4,-1),
            ((2,2), 3): (4,-1),
            ((2,2), 4): (4,-1),
            ((2,2), 5): (4,-1),

            ((2,3), 0): (4,-1),
            ((2,3), 1): (4,-1),
            ((2,3), 2): (4,-1),
            ((2,3), 3): (4,-1),
            ((2,3), 4): (4,-1),
            ((2,3), 5): (4,-1),

            ((3,4), 0): (4,-1),
            ((3,4), 1): (4,-1),
            ((3,4), 2): (4,-1),
            ((3,4), 3): (4,-1),
            ((3,4), 4): (4,-1),
            ((3,4), 5): (4,-1),

            ((3,5), 0): (4,-1),
            ((3,5), 1): (4,-1),
            ((3,5), 2): (4,-1),
            ((3,5), 3): (4,-1),
            ((3,5), 4): (4,-1),
            ((3,5), 5): (4,-1),

        }
        # Qs are actually not correct
        self.Qs = {
            ((4,-1), 0): 0,
            ((4,-1), 1): 0,
            ((4,-1), 2): 0,
            ((4,-1), 3): 0,
            ((4,-1), 4): 0,
            ((4,-1), 5): 0,


            ((0,0), 0): 0,
            ((0,0), 1): 0,
            ((0,0), 2): 0,
            ((0,0), 3): 0,
            ((0,0), 4): 0,
            ((0,0), 5): 0,

            ((0,1), 0): 0,
            ((0,1), 1): 0,
            ((0,1), 2): 0,
            ((0,1), 3): 0,
            ((0,1), 4): 0,
            ((0,1), 5): 0,

            ((1,2), 0): 0,
            ((1,2), 1): 0,
            ((1,2), 2): 0,
            ((1,2), 3): 0,
            ((1,2), 4): 0,
            ((1,2), 5): 0,

            ((1,3), 0): 0,
            ((1,3), 1): 0,
            ((1,3), 2): 0,
            ((1,3), 3): 0,
            ((1,3), 4): 0,
            ((1,3), 5): 0,

            ((1,4), 0): 0,
            ((1,4), 1): 0,
            ((1,4), 2): 0,
            ((1,4), 3): 0,
            ((1,4), 4): 0,
            ((1,4), 5): 0,

            ((1,5), 0): 0,
            ((1,5), 1): 0,
            ((1,5), 2): 0,
            ((1,5), 3): 0,
            ((1,5), 4): 0,
            ((1,5), 5): 0,

            ((2,2), 0): 0,
            ((2,2), 1): 0,
            ((2,2), 2): 0,
            ((2,2), 3): 0,
            ((2,2), 4): 0,
            ((2,2), 5): 0,

            ((2,3), 0): 0,
            ((2,3), 1): 0,
            ((2,3), 2): 0,
            ((2,3), 3): 0,
            ((2,3), 4): 0,
            ((2,3), 5): 0,

            ((3,4), 0): 0,
            ((3,4), 1): 0,
            ((3,4), 2): 0,
            ((3,4), 3): 0,
            ((3,4), 4): 0,
            ((3,4), 5): 0,

            ((3,5), 0): 0,
            ((3,5), 1): 0,
            ((3,5), 2): 0,
            ((3,5), 3): 0,
            ((3,5), 4): 0,
            ((3,5), 5): 0,

        }
        self.rewards = {
    # absorbing state gets no reward
            ((4,-1), 0): 0,
            ((4,-1), 1): 0,
            ((4,-1), 2): 0,
            ((4,-1), 3): 0,
            ((4,-1), 4): 0,
            ((4,-1), 5): 0,


            ((0,0), 0): 0,
            ((0,0), 1): 0,
            ((0,0), 2): 0,
            ((0,0), 3): 0,
            ((0,0), 4): 0,
            ((0,0), 5): 0,

            ((0,1), 0): 0,
            ((0,1), 1): 0,
            ((0,1), 2): 0,
            ((0,1), 3): 0,
            ((0,1), 4): 0,
            ((0,1), 5): 0,

            ((1,2), 0): 0,
            ((1,2), 1): 0,
            ((1,2), 2): self.big_reward,
            ((1,2), 3): 0,
            ((1,2), 4): 0,
            ((1,2), 5): 0,

            ((1,3), 0): 0,
            ((1,3), 1): 0,
            ((1,3), 2): 0,
            ((1,3), 3): 0,
            ((1,3), 4): 0,
            ((1,3), 5): 0,

            ((1,4), 0): 0,
            ((1,4), 1): 0,
            ((1,4), 2): 0,
            ((1,4), 3): 0,
            ((1,4), 4): 0,
            ((1,4), 5): 0,

            ((1,5), 0): 0,
            ((1,5), 1): 0,
            ((1,5), 2): 0,
            ((1,5), 3): 0,
            ((1,5), 4): 0,
            ((1,5), 5): 0,

            ((2,2), 0): 0,
            ((2,2), 1): 0,
            ((2,2), 2): 0,
            ((2,2), 3): 0,
            ((2,2), 4): 0,
            ((2,2), 5): 0,

            ((2,3), 0): 0,
            ((2,3), 1): 0,
            ((2,3), 2): 0,
            ((2,3), 3): 0,
            ((2,3), 4): 0,
            ((2,3), 5): 0,

            ((3,4), 0): 0,
            ((3,4), 1): 0,
            ((3,4), 2): 0,
            ((3,4), 3): 0,
            ((3,4), 4): 0,
            ((3,4), 5): 0,

            ((3,5), 0): 0,
            ((3,5), 1): 0,
            ((3,5), 2): 0,
            ((3,5), 3): 0,
            ((3,5), 4): 0,
            ((3,5), 5): 0,

        }
        self.starting_states = [(0,0)]
        self.terminal_states = [(2,2),(4,-1)]

class MultiTaskDebugCommonAncestorAE(MultiTaskDebug):
    def __init__(self):
        MultiTaskDebug.__init__(self)
        self.states = [0, 1, 2, 3]  # 8 is an absorbing state
        self.keys=[0,1,2,3,4,5]
        self.actions = [0, 1, 2, 3, 4, 5] # 0->A , 1->B, 2->C, 3->D, 4->E, 5->F
        self.transitions = {
            # absorbing state
            ((4,-1), 0): (4,-1),
            ((4,-1), 1): (4,-1),
            ((4,-1), 2): (4,-1),
            ((4,-1), 3): (4,-1),
            ((4,-1), 4): (4,-1),
            ((4,-1), 5): (4,-1),


            ((0,0), 0): (1,4),
            ((0,0), 1): (0,0),
            ((0,0), 2): (0,0),
            ((0,0), 3): (0,0),
            ((0,0), 4): (0,0),
            ((0,0), 5): (0,0),

            ((0,1), 0): (0,1),
            ((0,1), 1): (1,4),
            ((0,1), 2): (0,1),
            ((0,1), 3): (0,1),
            ((0,1), 4): (0,1),
            ((0,1), 5): (0,1),

            ((1,2), 0): (1,2),
            ((1,2), 1): (1,2),
            ((1,2), 2): (1,2),
            ((1,2), 3): (1,2),
            ((1,2), 4): (1,2),
            ((1,2), 5): (1,2),

            ((1,3), 0): (1,3),
            ((1,3), 1): (1,3),
            ((1,3), 2): (1,3),
            ((1,3), 3): (1,3),
            ((1,3), 4): (1,3),
            ((1,3), 5): (1,3),

            ((1,4), 0): (1,4),
            ((1,4), 1): (1,4),
            ((1,4), 2): (1,4),
            ((1,4), 3): (1,4),
            ((1,4), 4): (3,4),
            ((1,4), 5): (1,4),

            ((1,5), 0): (1,5),
            ((1,5), 1): (1,5),
            ((1,5), 2): (1,5),
            ((1,5), 3): (1,5),
            ((1,5), 4): (1,5),
            ((1,5), 5): (1,5),

            ((2,2), 0): (4,-1),
            ((2,2), 1): (4,-1),
            ((2,2), 2): (4,-1),
            ((2,2), 3): (4,-1),
            ((2,2), 4): (4,-1),
            ((2,2), 5): (4,-1),

            ((2,3), 0): (4,-1),
            ((2,3), 1): (4,-1),
            ((2,3), 2): (4,-1),
            ((2,3), 3): (4,-1),
            ((2,3), 4): (4,-1),
            ((2,3), 5): (4,-1),

            ((3,4), 0): (4,-1),
            ((3,4), 1): (4,-1),
            ((3,4), 2): (4,-1),
            ((3,4), 3): (4,-1),
            ((3,4), 4): (4,-1),
            ((3,4), 5): (4,-1),

            ((3,5), 0): (4,-1),
            ((3,5), 1): (4,-1),
            ((3,5), 2): (4,-1),
            ((3,5), 3): (4,-1),
            ((3,5), 4): (4,-1),
            ((3,5), 5): (4,-1),

        }
        # Qs are actually not correct
        self.Qs = {
            ((4,-1), 0): 0,
            ((4,-1), 1): 0,
            ((4,-1), 2): 0,
            ((4,-1), 3): 0,
            ((4,-1), 4): 0,
            ((4,-1), 5): 0,


            ((0,0), 0): 0,
            ((0,0), 1): 0,
            ((0,0), 2): 0,
            ((0,0), 3): 0,
            ((0,0), 4): 0,
            ((0,0), 5): 0,

            ((0,1), 0): 0,
            ((0,1), 1): 0,
            ((0,1), 2): 0,
            ((0,1), 3): 0,
            ((0,1), 4): 0,
            ((0,1), 5): 0,

            ((1,2), 0): 0,
            ((1,2), 1): 0,
            ((1,2), 2): 0,
            ((1,2), 3): 0,
            ((1,2), 4): 0,
            ((1,2), 5): 0,

            ((1,3), 0): 0,
            ((1,3), 1): 0,
            ((1,3), 2): 0,
            ((1,3), 3): 0,
            ((1,3), 4): 0,
            ((1,3), 5): 0,

            ((1,4), 0): 0,
            ((1,4), 1): 0,
            ((1,4), 2): 0,
            ((1,4), 3): 0,
            ((1,4), 4): 0,
            ((1,4), 5): 0,

            ((1,5), 0): 0,
            ((1,5), 1): 0,
            ((1,5), 2): 0,
            ((1,5), 3): 0,
            ((1,5), 4): 0,
            ((1,5), 5): 0,

            ((2,2), 0): 0,
            ((2,2), 1): 0,
            ((2,2), 2): 0,
            ((2,2), 3): 0,
            ((2,2), 4): 0,
            ((2,2), 5): 0,

            ((2,3), 0): 0,
            ((2,3), 1): 0,
            ((2,3), 2): 0,
            ((2,3), 3): 0,
            ((2,3), 4): 0,
            ((2,3), 5): 0,

            ((3,4), 0): 0,
            ((3,4), 1): 0,
            ((3,4), 2): 0,
            ((3,4), 3): 0,
            ((3,4), 4): 0,
            ((3,4), 5): 0,

            ((3,5), 0): 0,
            ((3,5), 1): 0,
            ((3,5), 2): 0,
            ((3,5), 3): 0,
            ((3,5), 4): 0,
            ((3,5), 5): 0,

        }
        self.rewards = {
    # absorbing state gets no reward
            ((4,-1), 0): 0,
            ((4,-1), 1): 0,
            ((4,-1), 2): 0,
            ((4,-1), 3): 0,
            ((4,-1), 4): 0,
            ((4,-1), 5): 0,


            ((0,0), 0): 0,
            ((0,0), 1): 0,
            ((0,0), 2): 0,
            ((0,0), 3): 0,
            ((0,0), 4): 0,
            ((0,0), 5): 0,

            ((0,1), 0): 0,
            ((0,1), 1): 0,
            ((0,1), 2): 0,
            ((0,1), 3): 0,
            ((0,1), 4): 0,
            ((0,1), 5): 0,

            ((1,2), 0): 0,
            ((1,2), 1): 0,
            ((1,2), 2): 0,
            ((1,2), 3): 0,
            ((1,2), 4): 0,
            ((1,2), 5): 0,

            ((1,3), 0): 0,
            ((1,3), 1): 0,
            ((1,3), 2): 0,
            ((1,3), 3): 0,
            ((1,3), 4): 0,
            ((1,3), 5): 0,

            ((1,4), 0): 0,
            ((1,4), 1): 0,
            ((1,4), 2): 0,
            ((1,4), 3): 0,
            ((1,4), 4): self.big_reward,
            ((1,4), 5): 0,

            ((1,5), 0): 0,
            ((1,5), 1): 0,
            ((1,5), 2): 0,
            ((1,5), 3): 0,
            ((1,5), 4): 0,
            ((1,5), 5): 0,

            ((2,2), 0): 0,
            ((2,2), 1): 0,
            ((2,2), 2): 0,
            ((2,2), 3): 0,
            ((2,2), 4): 0,
            ((2,2), 5): 0,

            ((2,3), 0): 0,
            ((2,3), 1): 0,
            ((2,3), 2): 0,
            ((2,3), 3): 0,
            ((2,3), 4): 0,
            ((2,3), 5): 0,

            ((3,4), 0): 0,
            ((3,4), 1): 0,
            ((3,4), 2): 0,
            ((3,4), 3): 0,
            ((3,4), 4): 0,
            ((3,4), 5): 0,

            ((3,5), 0): 0,
            ((3,5), 1): 0,
            ((3,5), 2): 0,
            ((3,5), 3): 0,
            ((3,5), 4): 0,
            ((3,5), 5): 0,

        }
        self.starting_states = [(0,0)]
        self.terminal_states = [(3,4),(4,-1)]

class MultiTaskDebugCommonAncestorAF(MultiTaskDebug):
    def __init__(self):
        MultiTaskDebug.__init__(self)
        self.states = [0, 1, 2, 3]  # 8 is an absorbing state
        self.keys=[0,1,2,3,4,5]
        self.actions = [0, 1, 2, 3, 4, 5] # 0->A , 1->B, 2->C, 3->D, 4->E, 5->F
        self.transitions = {
            # absorbing state
            ((4,-1), 0): (4,-1),
            ((4,-1), 1): (4,-1),
            ((4,-1), 2): (4,-1),
            ((4,-1), 3): (4,-1),
            ((4,-1), 4): (4,-1),
            ((4,-1), 5): (4,-1),


            ((0,0), 0): (1,5),
            ((0,0), 1): (0,0),
            ((0,0), 2): (0,0),
            ((0,0), 3): (0,0),
            ((0,0), 4): (0,0),
            ((0,0), 5): (0,0),

            ((0,1), 0): (0,1),
            ((0,1), 1): (1,5),
            ((0,1), 2): (0,1),
            ((0,1), 3): (0,1),
            ((0,1), 4): (0,1),
            ((0,1), 5): (0,1),

            ((1,2), 0): (1,2),
            ((1,2), 1): (1,2),
            ((1,2), 2): (2,2),
            ((1,2), 3): (1,2),
            ((1,2), 4): (1,2),
            ((1,2), 5): (1,2),

            ((1,3), 0): (1,3),
            ((1,3), 1): (1,3),
            ((1,3), 2): (1,3),
            ((1,3), 3): (1,3),
            ((1,3), 4): (1,3),
            ((1,3), 5): (1,3),

            ((1,4), 0): (1,4),
            ((1,4), 1): (1,4),
            ((1,4), 2): (1,4),
            ((1,4), 3): (1,4),
            ((1,4), 4): (1,4),
            ((1,4), 5): (1,4),

            ((1,5), 0): (1,5),
            ((1,5), 1): (1,5),
            ((1,5), 2): (1,5),
            ((1,5), 3): (1,5),
            ((1,5), 4): (1,5),
            ((1,5), 5): (3,5),

            ((2,2), 0): (4,-1),
            ((2,2), 1): (4,-1),
            ((2,2), 2): (4,-1),
            ((2,2), 3): (4,-1),
            ((2,2), 4): (4,-1),
            ((2,2), 5): (4,-1),

            ((2,3), 0): (4,-1),
            ((2,3), 1): (4,-1),
            ((2,3), 2): (4,-1),
            ((2,3), 3): (4,-1),
            ((2,3), 4): (4,-1),
            ((2,3), 5): (4,-1),

            ((3,4), 0): (4,-1),
            ((3,4), 1): (4,-1),
            ((3,4), 2): (4,-1),
            ((3,4), 3): (4,-1),
            ((3,4), 4): (4,-1),
            ((3,4), 5): (4,-1),

            ((3,5), 0): (4,-1),
            ((3,5), 1): (4,-1),
            ((3,5), 2): (4,-1),
            ((3,5), 3): (4,-1),
            ((3,5), 4): (4,-1),
            ((3,5), 5): (4,-1),

        }
        # Qs are actually not correct
        self.Qs = {
            ((4,-1), 0): 0,
            ((4,-1), 1): 0,
            ((4,-1), 2): 0,
            ((4,-1), 3): 0,
            ((4,-1), 4): 0,
            ((4,-1), 5): 0,


            ((0,0), 0): 0,
            ((0,0), 1): 0,
            ((0,0), 2): 0,
            ((0,0), 3): 0,
            ((0,0), 4): 0,
            ((0,0), 5): 0,

            ((0,1), 0): 0,
            ((0,1), 1): 0,
            ((0,1), 2): 0,
            ((0,1), 3): 0,
            ((0,1), 4): 0,
            ((0,1), 5): 0,

            ((1,2), 0): 0,
            ((1,2), 1): 0,
            ((1,2), 2): 0,
            ((1,2), 3): 0,
            ((1,2), 4): 0,
            ((1,2), 5): 0,

            ((1,3), 0): 0,
            ((1,3), 1): 0,
            ((1,3), 2): 0,
            ((1,3), 3): 0,
            ((1,3), 4): 0,
            ((1,3), 5): 0,

            ((1,4), 0): 0,
            ((1,4), 1): 0,
            ((1,4), 2): 0,
            ((1,4), 3): 0,
            ((1,4), 4): 0,
            ((1,4), 5): 0,

            ((1,5), 0): 0,
            ((1,5), 1): 0,
            ((1,5), 2): 0,
            ((1,5), 3): 0,
            ((1,5), 4): 0,
            ((1,5), 5): 0,

            ((2,2), 0): 0,
            ((2,2), 1): 0,
            ((2,2), 2): 0,
            ((2,2), 3): 0,
            ((2,2), 4): 0,
            ((2,2), 5): 0,

            ((2,3), 0): 0,
            ((2,3), 1): 0,
            ((2,3), 2): 0,
            ((2,3), 3): 0,
            ((2,3), 4): 0,
            ((2,3), 5): 0,

            ((3,4), 0): 0,
            ((3,4), 1): 0,
            ((3,4), 2): 0,
            ((3,4), 3): 0,
            ((3,4), 4): 0,
            ((3,4), 5): 0,

            ((3,5), 0): 0,
            ((3,5), 1): 0,
            ((3,5), 2): 0,
            ((3,5), 3): 0,
            ((3,5), 4): 0,
            ((3,5), 5): 0,

        }
        self.rewards = {
    # absorbing state gets no reward
            ((4,-1), 0): 0,
            ((4,-1), 1): 0,
            ((4,-1), 2): 0,
            ((4,-1), 3): 0,
            ((4,-1), 4): 0,
            ((4,-1), 5): 0,


            ((0,0), 0): 0,
            ((0,0), 1): 0,
            ((0,0), 2): 0,
            ((0,0), 3): 0,
            ((0,0), 4): 0,
            ((0,0), 5): 0,

            ((0,1), 0): 0,
            ((0,1), 1): 0,
            ((0,1), 2): 0,
            ((0,1), 3): 0,
            ((0,1), 4): 0,
            ((0,1), 5): 0,

            ((1,2), 0): 0,
            ((1,2), 1): 0,
            ((1,2), 2): 0,
            ((1,2), 3): 0,
            ((1,2), 4): 0,
            ((1,2), 5): 0,

            ((1,3), 0): 0,
            ((1,3), 1): 0,
            ((1,3), 2): 0,
            ((1,3), 3): 0,
            ((1,3), 4): 0,
            ((1,3), 5): 0,

            ((1,4), 0): 0,
            ((1,4), 1): 0,
            ((1,4), 2): 0,
            ((1,4), 3): 0,
            ((1,4), 4): 0,
            ((1,4), 5): 0,

            ((1,5), 0): 0,
            ((1,5), 1): 0,
            ((1,5), 2): 0,
            ((1,5), 3): 0,
            ((1,5), 4): 0,
            ((1,5), 5): self.big_reward,

            ((2,2), 0): 0,
            ((2,2), 1): 0,
            ((2,2), 2): 0,
            ((2,2), 3): 0,
            ((2,2), 4): 0,
            ((2,2), 5): 0,

            ((2,3), 0): 0,
            ((2,3), 1): 0,
            ((2,3), 2): 0,
            ((2,3), 3): 0,
            ((2,3), 4): 0,
            ((2,3), 5): 0,

            ((3,4), 0): 0,
            ((3,4), 1): 0,
            ((3,4), 2): 0,
            ((3,4), 3): 0,
            ((3,4), 4): 0,
            ((3,4), 5): 0,

            ((3,5), 0): 0,
            ((3,5), 1): 0,
            ((3,5), 2): 0,
            ((3,5), 3): 0,
            ((3,5), 4): 0,
            ((3,5), 5): 0,

        }
        self.starting_states = [(0,0)]
        self.terminal_states = [(3,5),(4,-1)]

class MultiTaskDebugCommonAncestorAC(MultiTaskDebug):
    def __init__(self):
        MultiTaskDebug.__init__(self)
        self.states = [0, 1, 2, 3]  # 8 is an absorbing state
        self.keys=[0,1,2,3,4,5]
        self.actions = [0, 1, 2, 3, 4, 5] # 0->A , 1->B, 2->C, 3->D, 4->E, 5->F
        self.transitions = {
            # absorbing state
            ((4,-1), 0): (4,-1),
            ((4,-1), 1): (4,-1),
            ((4,-1), 2): (4,-1),
            ((4,-1), 3): (4,-1),
            ((4,-1), 4): (4,-1),
            ((4,-1), 5): (4,-1),


            ((0,0), 0): (1,3),
            ((0,0), 1): (0,0),
            ((0,0), 2): (0,0),
            ((0,0), 3): (0,0),
            ((0,0), 4): (0,0),
            ((0,0), 5): (0,0),

            ((0,1), 0): (0,1),
            ((0,1), 1): (1,3),
            ((0,1), 2): (0,1),
            ((0,1), 3): (0,1),
            ((0,1), 4): (0,1),
            ((0,1), 5): (0,1),

            ((1,2), 0): (1,2),
            ((1,2), 1): (1,2),
            ((1,2), 2): (1,2),
            ((1,2), 3): (1,2),
            ((1,2), 4): (1,2),
            ((1,2), 5): (1,2),

            ((1,3), 0): (1,3),
            ((1,3), 1): (1,3),
            ((1,3), 2): (1,3),
            ((1,3), 3): (2,3),
            ((1,3), 4): (1,3),
            ((1,3), 5): (1,3),

            ((1,4), 0): (1,4),
            ((1,4), 1): (1,4),
            ((1,4), 2): (1,4),
            ((1,4), 3): (1,4),
            ((1,4), 4): (1,4),
            ((1,4), 5): (1,4),

            ((1,5), 0): (1,5),
            ((1,5), 1): (1,5),
            ((1,5), 2): (1,5),
            ((1,5), 3): (1,5),
            ((1,5), 4): (1,5),
            ((1,5), 5): (1,5),

            ((2,2), 0): (4,-1),
            ((2,2), 1): (4,-1),
            ((2,2), 2): (4,-1),
            ((2,2), 3): (4,-1),
            ((2,2), 4): (4,-1),
            ((2,2), 5): (4,-1),

            ((2,3), 0): (4,-1),
            ((2,3), 1): (4,-1),
            ((2,3), 2): (4,-1),
            ((2,3), 3): (4,-1),
            ((2,3), 4): (4,-1),
            ((2,3), 5): (4,-1),

            ((3,4), 0): (4,-1),
            ((3,4), 1): (4,-1),
            ((3,4), 2): (4,-1),
            ((3,4), 3): (4,-1),
            ((3,4), 4): (4,-1),
            ((3,4), 5): (4,-1),

            ((3,5), 0): (4,-1),
            ((3,5), 1): (4,-1),
            ((3,5), 2): (4,-1),
            ((3,5), 3): (4,-1),
            ((3,5), 4): (4,-1),
            ((3,5), 5): (4,-1),

        }
        # Qs are actually not correct
        self.Qs = {
            ((4,-1), 0): 0,
            ((4,-1), 1): 0,
            ((4,-1), 2): 0,
            ((4,-1), 3): 0,
            ((4,-1), 4): 0,
            ((4,-1), 5): 0,


            ((0,0), 0): 0,
            ((0,0), 1): 0,
            ((0,0), 2): 0,
            ((0,0), 3): 0,
            ((0,0), 4): 0,
            ((0,0), 5): 0,

            ((0,1), 0): 0,
            ((0,1), 1): 0,
            ((0,1), 2): 0,
            ((0,1), 3): 0,
            ((0,1), 4): 0,
            ((0,1), 5): 0,

            ((1,2), 0): 0,
            ((1,2), 1): 0,
            ((1,2), 2): 0,
            ((1,2), 3): 0,
            ((1,2), 4): 0,
            ((1,2), 5): 0,

            ((1,3), 0): 0,
            ((1,3), 1): 0,
            ((1,3), 2): 0,
            ((1,3), 3): 0,
            ((1,3), 4): 0,
            ((1,3), 5): 0,

            ((1,4), 0): 0,
            ((1,4), 1): 0,
            ((1,4), 2): 0,
            ((1,4), 3): 0,
            ((1,4), 4): 0,
            ((1,4), 5): 0,

            ((1,5), 0): 0,
            ((1,5), 1): 0,
            ((1,5), 2): 0,
            ((1,5), 3): 0,
            ((1,5), 4): 0,
            ((1,5), 5): 0,

            ((2,2), 0): 0,
            ((2,2), 1): 0,
            ((2,2), 2): 0,
            ((2,2), 3): 0,
            ((2,2), 4): 0,
            ((2,2), 5): 0,

            ((2,3), 0): 0,
            ((2,3), 1): 0,
            ((2,3), 2): 0,
            ((2,3), 3): 0,
            ((2,3), 4): 0,
            ((2,3), 5): 0,

            ((3,4), 0): 0,
            ((3,4), 1): 0,
            ((3,4), 2): 0,
            ((3,4), 3): 0,
            ((3,4), 4): 0,
            ((3,4), 5): 0,

            ((3,5), 0): 0,
            ((3,5), 1): 0,
            ((3,5), 2): 0,
            ((3,5), 3): 0,
            ((3,5), 4): 0,
            ((3,5), 5): 0,

        }
        self.rewards = {
    # absorbing state gets no reward
            ((4,-1), 0): 0,
            ((4,-1), 1): 0,
            ((4,-1), 2): 0,
            ((4,-1), 3): 0,
            ((4,-1), 4): 0,
            ((4,-1), 5): 0,


            ((0,0), 0): 0,
            ((0,0), 1): 0,
            ((0,0), 2): 0,
            ((0,0), 3): 0,
            ((0,0), 4): 0,
            ((0,0), 5): 0,

            ((0,1), 0): 0,
            ((0,1), 1): 0,
            ((0,1), 2): 0,
            ((0,1), 3): 0,
            ((0,1), 4): 0,
            ((0,1), 5): 0,

            ((1,2), 0): 0,
            ((1,2), 1): 0,
            ((1,2), 2): 0,
            ((1,2), 3): 0,
            ((1,2), 4): 0,
            ((1,2), 5): 0,

            ((1,3), 0): 0,
            ((1,3), 1): 0,
            ((1,3), 2): 0,
            ((1,3), 3): self.big_reward,
            ((1,3), 4): 0,
            ((1,3), 5): 0,

            ((1,4), 0): 0,
            ((1,4), 1): 0,
            ((1,4), 2): 0,
            ((1,4), 3): 0,
            ((1,4), 4): 0,
            ((1,4), 5): 0,

            ((1,5), 0): 0,
            ((1,5), 1): 0,
            ((1,5), 2): 0,
            ((1,5), 3): 0,
            ((1,5), 4): 0,
            ((1,5), 5): 0,

            ((2,2), 0): 0,
            ((2,2), 1): 0,
            ((2,2), 2): 0,
            ((2,2), 3): 0,
            ((2,2), 4): 0,
            ((2,2), 5): 0,

            ((2,3), 0): 0,
            ((2,3), 1): 0,
            ((2,3), 2): 0,
            ((2,3), 3): 0,
            ((2,3), 4): 0,
            ((2,3), 5): 0,

            ((3,4), 0): 0,
            ((3,4), 1): 0,
            ((3,4), 2): 0,
            ((3,4), 3): 0,
            ((3,4), 4): 0,
            ((3,4), 5): 0,

            ((3,5), 0): 0,
            ((3,5), 1): 0,
            ((3,5), 2): 0,
            ((3,5), 3): 0,
            ((3,5), 4): 0,
            ((3,5), 5): 0,

        }
        self.starting_states = [(0,0)]
        self.terminal_states = [(2,3),(4,-1)]

class MultiTaskDebugCommonAncestorDB(MultiTaskDebug):
    def __init__(self):
        MultiTaskDebug.__init__(self)
        self.states = [0, 1, 2, 3]  # 8 is an absorbing state
        self.keys=[0,1,2,3,4,5]
        self.actions = [0, 1, 2, 3, 4, 5] # 0->A , 1->B, 2->C, 3->D, 4->E, 5->F
        self.transitions = {
            # absorbing state
            ((4,-1), 0): (4,-1),
            ((4,-1), 1): (4,-1),
            ((4,-1), 2): (4,-1),
            ((4,-1), 3): (4,-1),
            ((4,-1), 4): (4,-1),
            ((4,-1), 5): (4,-1),


            ((0,0), 0): (1,2),
            ((0,0), 1): (0,0),
            ((0,0), 2): (0,0),
            ((0,0), 3): (0,0),
            ((0,0), 4): (0,0),
            ((0,0), 5): (0,0),

            ((0,1), 0): (0,1),
            ((0,1), 1): (1,2),
            ((0,1), 2): (0,1),
            ((0,1), 3): (0,1),
            ((0,1), 4): (0,1),
            ((0,1), 5): (0,1),

            ((1,2), 0): (1,2),
            ((1,2), 1): (1,2),
            ((1,2), 2): (2,2),
            ((1,2), 3): (1,2),
            ((1,2), 4): (1,2),
            ((1,2), 5): (1,2),

            ((1,3), 0): (1,3),
            ((1,3), 1): (1,3),
            ((1,3), 2): (1,3),
            ((1,3), 3): (1,3),
            ((1,3), 4): (1,3),
            ((1,3), 5): (1,3),

            ((1,4), 0): (1,4),
            ((1,4), 1): (1,4),
            ((1,4), 2): (1,4),
            ((1,4), 3): (1,4),
            ((1,4), 4): (1,4),
            ((1,4), 5): (1,4),

            ((1,5), 0): (1,5),
            ((1,5), 1): (1,5),
            ((1,5), 2): (1,5),
            ((1,5), 3): (1,5),
            ((1,5), 4): (1,5),
            ((1,5), 5): (1,5),

            ((2,2), 0): (4,-1),
            ((2,2), 1): (4,-1),
            ((2,2), 2): (4,-1),
            ((2,2), 3): (4,-1),
            ((2,2), 4): (4,-1),
            ((2,2), 5): (4,-1),

            ((2,3), 0): (4,-1),
            ((2,3), 1): (4,-1),
            ((2,3), 2): (4,-1),
            ((2,3), 3): (4,-1),
            ((2,3), 4): (4,-1),
            ((2,3), 5): (4,-1),

            ((3,4), 0): (4,-1),
            ((3,4), 1): (4,-1),
            ((3,4), 2): (4,-1),
            ((3,4), 3): (4,-1),
            ((3,4), 4): (4,-1),
            ((3,4), 5): (4,-1),

            ((3,5), 0): (4,-1),
            ((3,5), 1): (4,-1),
            ((3,5), 2): (4,-1),
            ((3,5), 3): (4,-1),
            ((3,5), 4): (4,-1),
            ((3,5), 5): (4,-1),

        }
        # Qs are actually not correct
        self.Qs = {
            ((4,-1), 0): 0,
            ((4,-1), 1): 0,
            ((4,-1), 2): 0,
            ((4,-1), 3): 0,
            ((4,-1), 4): 0,
            ((4,-1), 5): 0,


            ((0,0), 0): 0,
            ((0,0), 1): 0,
            ((0,0), 2): 0,
            ((0,0), 3): 0,
            ((0,0), 4): 0,
            ((0,0), 5): 0,

            ((0,1), 0): 0,
            ((0,1), 1): 0,
            ((0,1), 2): 0,
            ((0,1), 3): 0,
            ((0,1), 4): 0,
            ((0,1), 5): 0,

            ((1,2), 0): 0,
            ((1,2), 1): 0,
            ((1,2), 2): 0,
            ((1,2), 3): 0,
            ((1,2), 4): 0,
            ((1,2), 5): 0,

            ((1,3), 0): 0,
            ((1,3), 1): 0,
            ((1,3), 2): 0,
            ((1,3), 3): 0,
            ((1,3), 4): 0,
            ((1,3), 5): 0,

            ((1,4), 0): 0,
            ((1,4), 1): 0,
            ((1,4), 2): 0,
            ((1,4), 3): 0,
            ((1,4), 4): 0,
            ((1,4), 5): 0,

            ((1,5), 0): 0,
            ((1,5), 1): 0,
            ((1,5), 2): 0,
            ((1,5), 3): 0,
            ((1,5), 4): 0,
            ((1,5), 5): 0,

            ((2,2), 0): 0,
            ((2,2), 1): 0,
            ((2,2), 2): 0,
            ((2,2), 3): 0,
            ((2,2), 4): 0,
            ((2,2), 5): 0,

            ((2,3), 0): 0,
            ((2,3), 1): 0,
            ((2,3), 2): 0,
            ((2,3), 3): 0,
            ((2,3), 4): 0,
            ((2,3), 5): 0,

            ((3,4), 0): 0,
            ((3,4), 1): 0,
            ((3,4), 2): 0,
            ((3,4), 3): 0,
            ((3,4), 4): 0,
            ((3,4), 5): 0,

            ((3,5), 0): 0,
            ((3,5), 1): 0,
            ((3,5), 2): 0,
            ((3,5), 3): 0,
            ((3,5), 4): 0,
            ((3,5), 5): 0,

        }
        self.rewards = {
    # absorbing state gets no reward
            ((4,-1), 0): 0,
            ((4,-1), 1): 0,
            ((4,-1), 2): 0,
            ((4,-1), 3): 0,
            ((4,-1), 4): 0,
            ((4,-1), 5): 0,


            ((0,0), 0): 0,
            ((0,0), 1): 0,
            ((0,0), 2): 0,
            ((0,0), 3): 0,
            ((0,0), 4): 0,
            ((0,0), 5): 0,

            ((0,1), 0): 0,
            ((0,1), 1): 0,
            ((0,1), 2): 0,
            ((0,1), 3): 0,
            ((0,1), 4): 0,
            ((0,1), 5): 0,

            ((1,2), 0): 0,
            ((1,2), 1): 0,
            ((1,2), 2): self.big_reward,
            ((1,2), 3): 0,
            ((1,2), 4): 0,
            ((1,2), 5): 0,

            ((1,3), 0): 0,
            ((1,3), 1): 0,
            ((1,3), 2): 0,
            ((1,3), 3): 0,
            ((1,3), 4): 0,
            ((1,3), 5): 0,

            ((1,4), 0): 0,
            ((1,4), 1): 0,
            ((1,4), 2): 0,
            ((1,4), 3): 0,
            ((1,4), 4): 0,
            ((1,4), 5): 0,

            ((1,5), 0): 0,
            ((1,5), 1): 0,
            ((1,5), 2): 0,
            ((1,5), 3): 0,
            ((1,5), 4): 0,
            ((1,5), 5): 0,

            ((2,2), 0): 0,
            ((2,2), 1): 0,
            ((2,2), 2): 0,
            ((2,2), 3): 0,
            ((2,2), 4): 0,
            ((2,2), 5): 0,

            ((2,3), 0): 0,
            ((2,3), 1): 0,
            ((2,3), 2): 0,
            ((2,3), 3): 0,
            ((2,3), 4): 0,
            ((2,3), 5): 0,

            ((3,4), 0): 0,
            ((3,4), 1): 0,
            ((3,4), 2): 0,
            ((3,4), 3): 0,
            ((3,4), 4): 0,
            ((3,4), 5): 0,

            ((3,5), 0): 0,
            ((3,5), 1): 0,
            ((3,5), 2): 0,
            ((3,5), 3): 0,
            ((3,5), 4): 0,
            ((3,5), 5): 0,

        }
        self.starting_states = [(0,1)]
        self.terminal_states = [(2,2),(4,-1)]


class MultiTaskDebugCommonAncestorDC(MultiTaskDebug):
    def __init__(self):
        MultiTaskDebug.__init__(self)
        self.states = [0, 1, 2, 3]  # 8 is an absorbing state
        self.keys=[0,1,2,3,4,5]
        self.actions = [0, 1, 2, 3, 4, 5] # 0->A , 1->B, 2->C, 3->D, 4->E, 5->F
        self.transitions = {
            # absorbing state
            ((4,-1), 0): (4,-1),
            ((4,-1), 1): (4,-1),
            ((4,-1), 2): (4,-1),
            ((4,-1), 3): (4,-1),
            ((4,-1), 4): (4,-1),
            ((4,-1), 5): (4,-1),


            ((0,0), 0): (1,3),
            ((0,0), 1): (0,0),
            ((0,0), 2): (0,0),
            ((0,0), 3): (0,0),
            ((0,0), 4): (0,0),
            ((0,0), 5): (0,0),

            ((0,1), 0): (0,1),
            ((0,1), 1): (1,3),
            ((0,1), 2): (0,1),
            ((0,1), 3): (0,1),
            ((0,1), 4): (0,1),
            ((0,1), 5): (0,1),

            ((1,2), 0): (1,2),
            ((1,2), 1): (1,2),
            ((1,2), 2): (1,2),
            ((1,2), 3): (1,2),
            ((1,2), 4): (1,2),
            ((1,2), 5): (1,2),

            ((1,3), 0): (1,3),
            ((1,3), 1): (1,3),
            ((1,3), 2): (1,3),
            ((1,3), 3): (2,3),
            ((1,3), 4): (1,3),
            ((1,3), 5): (1,3),

            ((1,4), 0): (1,4),
            ((1,4), 1): (1,4),
            ((1,4), 2): (1,4),
            ((1,4), 3): (1,4),
            ((1,4), 4): (1,4),
            ((1,4), 5): (1,4),

            ((1,5), 0): (1,5),
            ((1,5), 1): (1,5),
            ((1,5), 2): (1,5),
            ((1,5), 3): (1,5),
            ((1,5), 4): (1,5),
            ((1,5), 5): (1,5),

            ((2,2), 0): (4,-1),
            ((2,2), 1): (4,-1),
            ((2,2), 2): (4,-1),
            ((2,2), 3): (4,-1),
            ((2,2), 4): (4,-1),
            ((2,2), 5): (4,-1),

            ((2,3), 0): (4,-1),
            ((2,3), 1): (4,-1),
            ((2,3), 2): (4,-1),
            ((2,3), 3): (4,-1),
            ((2,3), 4): (4,-1),
            ((2,3), 5): (4,-1),

            ((3,4), 0): (4,-1),
            ((3,4), 1): (4,-1),
            ((3,4), 2): (4,-1),
            ((3,4), 3): (4,-1),
            ((3,4), 4): (4,-1),
            ((3,4), 5): (4,-1),

            ((3,5), 0): (4,-1),
            ((3,5), 1): (4,-1),
            ((3,5), 2): (4,-1),
            ((3,5), 3): (4,-1),
            ((3,5), 4): (4,-1),
            ((3,5), 5): (4,-1),

        }
        # Qs are actually not correct
        self.Qs = {
            ((4,-1), 0): 0,
            ((4,-1), 1): 0,
            ((4,-1), 2): 0,
            ((4,-1), 3): 0,
            ((4,-1), 4): 0,
            ((4,-1), 5): 0,


            ((0,0), 0): 0,
            ((0,0), 1): 0,
            ((0,0), 2): 0,
            ((0,0), 3): 0,
            ((0,0), 4): 0,
            ((0,0), 5): 0,

            ((0,1), 0): 0,
            ((0,1), 1): 0,
            ((0,1), 2): 0,
            ((0,1), 3): 0,
            ((0,1), 4): 0,
            ((0,1), 5): 0,

            ((1,2), 0): 0,
            ((1,2), 1): 0,
            ((1,2), 2): 0,
            ((1,2), 3): 0,
            ((1,2), 4): 0,
            ((1,2), 5): 0,

            ((1,3), 0): 0,
            ((1,3), 1): 0,
            ((1,3), 2): 0,
            ((1,3), 3): 0,
            ((1,3), 4): 0,
            ((1,3), 5): 0,

            ((1,4), 0): 0,
            ((1,4), 1): 0,
            ((1,4), 2): 0,
            ((1,4), 3): 0,
            ((1,4), 4): 0,
            ((1,4), 5): 0,

            ((1,5), 0): 0,
            ((1,5), 1): 0,
            ((1,5), 2): 0,
            ((1,5), 3): 0,
            ((1,5), 4): 0,
            ((1,5), 5): 0,

            ((2,2), 0): 0,
            ((2,2), 1): 0,
            ((2,2), 2): 0,
            ((2,2), 3): 0,
            ((2,2), 4): 0,
            ((2,2), 5): 0,

            ((2,3), 0): 0,
            ((2,3), 1): 0,
            ((2,3), 2): 0,
            ((2,3), 3): 0,
            ((2,3), 4): 0,
            ((2,3), 5): 0,

            ((3,4), 0): 0,
            ((3,4), 1): 0,
            ((3,4), 2): 0,
            ((3,4), 3): 0,
            ((3,4), 4): 0,
            ((3,4), 5): 0,

            ((3,5), 0): 0,
            ((3,5), 1): 0,
            ((3,5), 2): 0,
            ((3,5), 3): 0,
            ((3,5), 4): 0,
            ((3,5), 5): 0,

        }
        self.rewards = {
    # absorbing state gets no reward
            ((4,-1), 0): 0,
            ((4,-1), 1): 0,
            ((4,-1), 2): 0,
            ((4,-1), 3): 0,
            ((4,-1), 4): 0,
            ((4,-1), 5): 0,


            ((0,0), 0): 0,
            ((0,0), 1): 0,
            ((0,0), 2): 0,
            ((0,0), 3): 0,
            ((0,0), 4): 0,
            ((0,0), 5): 0,

            ((0,1), 0): 0,
            ((0,1), 1): 0,
            ((0,1), 2): 0,
            ((0,1), 3): 0,
            ((0,1), 4): 0,
            ((0,1), 5): 0,

            ((1,2), 0): 0,
            ((1,2), 1): 0,
            ((1,2), 2): 0,
            ((1,2), 3): 0,
            ((1,2), 4): 0,
            ((1,2), 5): 0,

            ((1,3), 0): 0,
            ((1,3), 1): 0,
            ((1,3), 2): 0,
            ((1,3), 3): self.big_reward,
            ((1,3), 4): 0,
            ((1,3), 5): 0,

            ((1,4), 0): 0,
            ((1,4), 1): 0,
            ((1,4), 2): 0,
            ((1,4), 3): 0,
            ((1,4), 4): 0,
            ((1,4), 5): 0,

            ((1,5), 0): 0,
            ((1,5), 1): 0,
            ((1,5), 2): 0,
            ((1,5), 3): 0,
            ((1,5), 4): 0,
            ((1,5), 5): 0,

            ((2,2), 0): 0,
            ((2,2), 1): 0,
            ((2,2), 2): 0,
            ((2,2), 3): 0,
            ((2,2), 4): 0,
            ((2,2), 5): 0,

            ((2,3), 0): 0,
            ((2,3), 1): 0,
            ((2,3), 2): 0,
            ((2,3), 3): 0,
            ((2,3), 4): 0,
            ((2,3), 5): 0,

            ((3,4), 0): 0,
            ((3,4), 1): 0,
            ((3,4), 2): 0,
            ((3,4), 3): 0,
            ((3,4), 4): 0,
            ((3,4), 5): 0,

            ((3,5), 0): 0,
            ((3,5), 1): 0,
            ((3,5), 2): 0,
            ((3,5), 3): 0,
            ((3,5), 4): 0,
            ((3,5), 5): 0,

        }
        self.starting_states = [(0,1)]
        self.terminal_states = [(2,3),(4,-1)]

class MultiTaskDebugTaskTwoStepInvarianceV1(MultiTaskDebug):
    def __init__(self):
        MultiTaskDebug.__init__(self)
        self.states = [0, 1, 2, 3, 4]  # 5 is an absorbing state
        self.keys=[0,1] #For room 2
        self.actions = [0, 1, 2, 3] # 0->A , 1->B, 2->C, 3->D, 4->E, 5->F
        self.transitions = {
            # absorbing state
            ((5,-1), 0): (5,-1),
            ((5,-1), 1): (5,-1),
            ((5,-1), 2): (5,-1),
            ((5,-1), 3): (5,-1),



            ((0,-1), 0): (2,0),
            ((0,-1), 1): (0,-1),
            ((0,-1), 2): (0,-1),
            ((0,-1), 3): (0,-1),

            ((1,-1), 0): (1,-1),
            ((1,-1), 1): (2,0),
            ((1,-1), 2): (1,-1),
            ((1,-1), 3): (1,-1),


            ((2,0), 0): (2,0),
            ((2,0), 1): (2,0),
            ((2,0), 2): (3,-1),
            ((2,0), 3): (2,0),

            ((2,1), 0): (2,1),
            ((2,1), 1): (2,1),
            ((2,1), 2): (2,1),
            ((2,1), 3): (2,1),


            ((3,-1), 0): (5,-1),
            ((3,-1), 1): (5,-1),
            ((3,-1), 2): (5,-1),
            ((3,-1), 3): (5,-1),

            ((4,-1), 0): (5,-1),
            ((4,-1), 1): (5,-1),
            ((4,-1), 2): (5,-1),
            ((4,-1), 3): (5,-1),

        }
        # Qs are actually not correct
        self.Qs = {
            ((5,-1), 0): 0,
            ((5,-1), 1): 0,
            ((5,-1), 2): 0,
            ((5,-1), 3): 0,



            ((0,-1), 0): 0,
            ((0,-1), 1): 0,
            ((0,-1), 2): 0,
            ((0,-1), 3): 0,

            ((1,-1), 0): 0,
            ((1,-1), 1): 0,
            ((1,-1), 2): 0,
            ((1,-1), 3): 0,


            ((2,0), 0): 0,
            ((2,0), 1): 0,
            ((2,0), 2): 0,
            ((2,0), 3): 0,

            ((2,1), 0): 0,
            ((2,1), 1): 0,
            ((2,1), 2): 0,
            ((2,1), 3): 0,


            ((3,-1), 0): 0,
            ((3,-1), 1): 0,
            ((3,-1), 2): 0,
            ((3,-1), 3): 0,

            ((4,-1), 0): 0,
            ((4,-1), 1): 0,
            ((4,-1), 2): 0,
            ((4,-1), 3): 0,


        }
        self.rewards = {
    # absorbing state gets no reward
            ((5,-1), 0): 0,
            ((5,-1), 1): 0,
            ((5,-1), 2): 0,
            ((5,-1), 3): 0,



            ((0,-1), 0): 0,
            ((0,-1), 1): 0,
            ((0,-1), 2): 0,
            ((0,-1), 3): 0,

            ((1,-1), 0): 0,
            ((1,-1), 1): 0,
            ((1,-1), 2): 0,
            ((1,-1), 3): 0,


            ((2,0), 0): 0,
            ((2,0), 1): 0,
            ((2,0), 2): self.big_reward,
            ((2,0), 3): 0,

            ((2,1), 0): 0,
            ((2,1), 1): 0,
            ((2,1), 2): 0,
            ((2,1), 3): 0,


            ((3,-1), 0): 0,
            ((3,-1), 1): 0,
            ((3,-1), 2): 0,
            ((3,-1), 3): 0,

            ((4,-1), 0): 0,
            ((4,-1), 1): 0,
            ((4,-1), 2): 0,
            ((4,-1), 3): 0,


        }
        self.starting_states = [(0,-1)]
        self.terminal_states = [(3,-1),(5,-1)]

class MultiTaskDebugTaskTwoStepInvarianceV2(MultiTaskDebug):
    def __init__(self):
        MultiTaskDebug.__init__(self)
        self.states = [0, 1, 2, 3, 4]  # 5 is an absorbing state
        self.keys=[0,1] #For room 2
        self.actions = [0, 1, 2, 3] # 0->A , 1->B, 2->C, 3->D, 4->E, 5->F
        self.transitions = {
            # absorbing state
            ((5,-1), 0): (5,-1),
            ((5,-1), 1): (5,-1),
            ((5,-1), 2): (5,-1),
            ((5,-1), 3): (5,-1),



            ((0,-1), 0): (2,1),
            ((0,-1), 1): (0,-1),
            ((0,-1), 2): (0,-1),
            ((0,-1), 3): (0,-1),

            ((1,-1), 0): (1,-1),
            ((1,-1), 1): (2,1),
            ((1,-1), 2): (1,-1),
            ((1,-1), 3): (1,-1),


            ((2,0), 0): (2,0),
            ((2,0), 1): (2,0),
            ((2,0), 2): (2,0),
            ((2,0), 3): (2,0),

            ((2,1), 0): (2,1),
            ((2,1), 1): (2,1),
            ((2,1), 2): (2,1),
            ((2,1), 3): (4,-1),


            ((3,-1), 0): (5,-1),
            ((3,-1), 1): (5,-1),
            ((3,-1), 2): (5,-1),
            ((3,-1), 3): (5,-1),

            ((4,-1), 0): (5,-1),
            ((4,-1), 1): (5,-1),
            ((4,-1), 2): (5,-1),
            ((4,-1), 3): (5,-1),

        }
        # Qs are actually not correct
        self.Qs = {
            ((5,-1), 0): 0,
            ((5,-1), 1): 0,
            ((5,-1), 2): 0,
            ((5,-1), 3): 0,



            ((0,-1), 0): 0,
            ((0,-1), 1): 0,
            ((0,-1), 2): 0,
            ((0,-1), 3): 0,

            ((1,-1), 0): 0,
            ((1,-1), 1): 0,
            ((1,-1), 2): 0,
            ((1,-1), 3): 0,


            ((2,0), 0): 0,
            ((2,0), 1): 0,
            ((2,0), 2): 0,
            ((2,0), 3): 0,

            ((2,1), 0): 0,
            ((2,1), 1): 0,
            ((2,1), 2): 0,
            ((2,1), 3): 0,


            ((3,-1), 0): 0,
            ((3,-1), 1): 0,
            ((3,-1), 2): 0,
            ((3,-1), 3): 0,

            ((4,-1), 0): 0,
            ((4,-1), 1): 0,
            ((4,-1), 2): 0,
            ((4,-1), 3): 0,


        }
        self.rewards = {
    # absorbing state gets no reward
            ((5,-1), 0): 0,
            ((5,-1), 1): 0,
            ((5,-1), 2): 0,
            ((5,-1), 3): 0,



            ((0,-1), 0): 0,
            ((0,-1), 1): 0,
            ((0,-1), 2): 0,
            ((0,-1), 3): 0,

            ((1,-1), 0): 0,
            ((1,-1), 1): 0,
            ((1,-1), 2): 0,
            ((1,-1), 3): 0,


            ((2,0), 0): 0,
            ((2,0), 1): 0,
            ((2,0), 2): 0,
            ((2,0), 3): 0,

            ((2,1), 0): 0,
            ((2,1), 1): 0,
            ((2,1), 2): 0,
            ((2,1), 3): self.big_reward,


            ((3,-1), 0): 0,
            ((3,-1), 1): 0,
            ((3,-1), 2): 0,
            ((3,-1), 3): 0,

            ((4,-1), 0): 0,
            ((4,-1), 1): 0,
            ((4,-1), 2): 0,
            ((4,-1), 3): 0,


        }
        self.starting_states = [(1,-1)]
        self.terminal_states = [(4,-1),(5,-1)]

class MultiTaskDebugTaskTwoStepParallelismV1(MultiTaskDebug):
    def __init__(self):
        MultiTaskDebug.__init__(self)
        self.states = [0, 1, 2, 3, 4]  # 5 is an absorbing state
        self.keys=[0,1] #For room 2
        self.actions = [0, 1, 2, 3] # 0->A , 1->B, 2->C, 3->D, 4->E, 5->F
        self.transitions = {
            # absorbing state
            ((5,-1), 0): (5,-1),
            ((5,-1), 1): (5,-1),
            ((5,-1), 2): (5,-1),
            ((5,-1), 3): (5,-1),



            ((0,-1), 0): (2,1),
            ((0,-1), 1): (0,-1),
            ((0,-1), 2): (0,-1),
            ((0,-1), 3): (0,-1),

            ((1,-1), 0): (1,-1),
            ((1,-1), 1): (2,1),
            ((1,-1), 2): (1,-1),
            ((1,-1), 3): (1,-1),


            ((2,0), 0): (2,0),
            ((2,0), 1): (2,0),
            ((2,0), 2): (2,0),
            ((2,0), 3): (2,0),

            ((2,1), 0): (2,1),
            ((2,1), 1): (2,1),
            ((2,1), 2): (2,1),
            ((2,1), 3): (4,-1),


            ((3,-1), 0): (5,-1),
            ((3,-1), 1): (5,-1),
            ((3,-1), 2): (5,-1),
            ((3,-1), 3): (5,-1),

            ((4,-1), 0): (5,-1),
            ((4,-1), 1): (5,-1),
            ((4,-1), 2): (5,-1),
            ((4,-1), 3): (5,-1),

        }
        # Qs are actually not correct
        self.Qs = {
            ((5,-1), 0): 0,
            ((5,-1), 1): 0,
            ((5,-1), 2): 0,
            ((5,-1), 3): 0,



            ((0,-1), 0): 0,
            ((0,-1), 1): 0,
            ((0,-1), 2): 0,
            ((0,-1), 3): 0,

            ((1,-1), 0): 0,
            ((1,-1), 1): 0,
            ((1,-1), 2): 0,
            ((1,-1), 3): 0,


            ((2,0), 0): 0,
            ((2,0), 1): 0,
            ((2,0), 2): 0,
            ((2,0), 3): 0,

            ((2,1), 0): 0,
            ((2,1), 1): 0,
            ((2,1), 2): 0,
            ((2,1), 3): 0,


            ((3,-1), 0): 0,
            ((3,-1), 1): 0,
            ((3,-1), 2): 0,
            ((3,-1), 3): 0,

            ((4,-1), 0): 0,
            ((4,-1), 1): 0,
            ((4,-1), 2): 0,
            ((4,-1), 3): 0,


        }
        self.rewards = {
    # absorbing state gets no reward
            ((5,-1), 0): 0,
            ((5,-1), 1): 0,
            ((5,-1), 2): 0,
            ((5,-1), 3): 0,



            ((0,-1), 0): 0,
            ((0,-1), 1): 0,
            ((0,-1), 2): 0,
            ((0,-1), 3): 0,

            ((1,-1), 0): 0,
            ((1,-1), 1): 0,
            ((1,-1), 2): 0,
            ((1,-1), 3): 0,


            ((2,0), 0): 0,
            ((2,0), 1): 0,
            ((2,0), 2): 0,
            ((2,0), 3): 0,

            ((2,1), 0): 0,
            ((2,1), 1): 0,
            ((2,1), 2): 0,
            ((2,1), 3): self.big_reward,


            ((3,-1), 0): 0,
            ((3,-1), 1): 0,
            ((3,-1), 2): 0,
            ((3,-1), 3): 0,

            ((4,-1), 0): 0,
            ((4,-1), 1): 0,
            ((4,-1), 2): 0,
            ((4,-1), 3): 0,


        }
        self.starting_states = [(1,-1)]
        self.terminal_states = [(3,-1),(5,-1)]

class MultiTaskDebugTaskTwoStepParallelismV2(MultiTaskDebug):
    def __init__(self):
        MultiTaskDebug.__init__(self)
        self.states = [0, 1, 2, 3, 4]  # 5 is an absorbing state
        self.keys=[0,1] #For room 2
        self.actions = [0, 1, 2, 3] # 0->A , 1->B, 2->C, 3->D, 4->E, 5->F
        self.transitions = {
            # absorbing state
            ((5,-1), 0): (5,-1),
            ((5,-1), 1): (5,-1),
            ((5,-1), 2): (5,-1),
            ((5,-1), 3): (5,-1),



            ((0,-1), 0): (2,0),
            ((0,-1), 1): (0,-1),
            ((0,-1), 2): (0,-1),
            ((0,-1), 3): (0,-1),

            ((1,-1), 0): (1,-1),
            ((1,-1), 1): (2,0),
            ((1,-1), 2): (1,-1),
            ((1,-1), 3): (1,-1),


            ((2,0), 0): (2,0),
            ((2,0), 1): (2,0),
            ((2,0), 2): (3,-1),
            ((2,0), 3): (2,0),

            ((2,1), 0): (2,1),
            ((2,1), 1): (2,1),
            ((2,1), 2): (2,1),
            ((2,1), 3): (2,1),


            ((3,-1), 0): (5,-1),
            ((3,-1), 1): (5,-1),
            ((3,-1), 2): (5,-1),
            ((3,-1), 3): (5,-1),

            ((4,-1), 0): (5,-1),
            ((4,-1), 1): (5,-1),
            ((4,-1), 2): (5,-1),
            ((4,-1), 3): (5,-1),

        }
        # Qs are actually not correct
        self.Qs = {
            ((5,-1), 0): 0,
            ((5,-1), 1): 0,
            ((5,-1), 2): 0,
            ((5,-1), 3): 0,



            ((0,-1), 0): 0,
            ((0,-1), 1): 0,
            ((0,-1), 2): 0,
            ((0,-1), 3): 0,

            ((1,-1), 0): 0,
            ((1,-1), 1): 0,
            ((1,-1), 2): 0,
            ((1,-1), 3): 0,


            ((2,0), 0): 0,
            ((2,0), 1): 0,
            ((2,0), 2): 0,
            ((2,0), 3): 0,

            ((2,1), 0): 0,
            ((2,1), 1): 0,
            ((2,1), 2): 0,
            ((2,1), 3): 0,


            ((3,-1), 0): 0,
            ((3,-1), 1): 0,
            ((3,-1), 2): 0,
            ((3,-1), 3): 0,

            ((4,-1), 0): 0,
            ((4,-1), 1): 0,
            ((4,-1), 2): 0,
            ((4,-1), 3): 0,


        }
        self.rewards = {
    # absorbing state gets no reward
            ((5,-1), 0): 0,
            ((5,-1), 1): 0,
            ((5,-1), 2): 0,
            ((5,-1), 3): 0,



            ((0,-1), 0): 0,
            ((0,-1), 1): 0,
            ((0,-1), 2): 0,
            ((0,-1), 3): 0,

            ((1,-1), 0): 0,
            ((1,-1), 1): 0,
            ((1,-1), 2): 0,
            ((1,-1), 3): 0,


            ((2,0), 0): 0,
            ((2,0), 1): 0,
            ((2,0), 2): self.big_reward,
            ((2,0), 3): 0,

            ((2,1), 0): 0,
            ((2,1), 1): 0,
            ((2,1), 2): 0,
            ((2,1), 3): 0,


            ((3,-1), 0): 0,
            ((3,-1), 1): 0,
            ((3,-1), 2): 0,
            ((3,-1), 3): 0,

            ((4,-1), 0): 0,
            ((4,-1), 1): 0,
            ((4,-1), 2): 0,
            ((4,-1), 3): 0,


        }
        self.starting_states = [(1,-1)]
        self.terminal_states = [(4,-1),(5,-1)]

class MultiTaskDebugTaskABCSIX(MultiTaskDebug):
    def __init__(self):
        MultiTaskDebug.__init__(self)
        self.states = [0, 1, 2, 3]  # 4 is an absorbing state
        self.keys=[0,1,2,3,4,5]
        self.actions = [0, 1, 2, 3, 4, 5] # 0->A , 1->B, 2->C, 3->D, 4->E, 5->F
        self.transitions = {
            # absorbing state
            ((4,-1), 0): (4,-1),
            ((4,-1), 1): (4,-1),
            ((4,-1), 2): (4,-1),
            ((4,-1), 3): (4,-1),
            ((4,-1), 4): (4,-1),
            ((4,-1), 5): (4,-1),


            ((0,0), 0): (1,2),
            ((0,0), 1): (0,0),
            ((0,0), 2): (0,0),
            ((0,0), 3): (0,0),
            ((0,0), 4): (0,0),
            ((0,0), 5): (0,0),

            ((1,2), 0): (1,2),
            ((1,2), 1): (2,-1),
            ((1,2), 2): (1,2),
            ((1,2), 3): (1,2),
            ((1,2), 4): (1,2),
            ((1,2), 5): (1,2),

            ((2,-1), 0): (2,-1),
            ((2,-1), 1): (2,-1),
            ((2,-1), 2): (3,4),
            ((2,-1), 3): (3,5),
            ((2,-1), 4): (2,-1),
            ((2,-1), 5): (2,-1),

            ((3,4), 0): (4,-1),
            ((3,4), 1): (4,-1),
            ((3,4), 2): (4,-1),
            ((3,4), 3): (4,-1),
            ((3,4), 4): (4,-1),
            ((3,4), 5): (4,-1),

            ((3,5), 0): (2,-1),
            ((3,5), 1): (2,-1),
            ((3,5), 2): (3, 4),
            ((3,5), 3): (3, 5),
            ((3,5), 4): (2,-1),
            ((3,5), 5): (2,-1),


        }
        # Qs are actually not correct
        self.Qs = {
            ((4,-1), 0): 0,
            ((4,-1), 1): 0,
            ((4,-1), 2): 0,
            ((4,-1), 3): 0,
            ((4,-1), 4): 0,
            ((4,-1), 5): 0,


            ((0,0), 0): 0,
            ((0,0), 1): 0,
            ((0,0), 2): 0,
            ((0,0), 3): 0,
            ((0,0), 4): 0,
            ((0,0), 5): 0,

            ((1,2), 0): 0,
            ((1,2), 1): 0,
            ((1,2), 2): 0,
            ((1,2), 3): 0,
            ((1,2), 4): 0,
            ((1,2), 5): 0,

            ((2,-1), 0): 0,
            ((2,-1), 1): 0,
            ((2,-1), 2): 0,
            ((2,-1), 3): 0,
            ((2,-1), 4): 0,
            ((2,-1), 5): 0,

            ((3,4), 0): 0,
            ((3,4), 1): 0,
            ((3,4), 2): 0,
            ((3,4), 3): 0,
            ((3,4), 4): 0,
            ((3,4), 5): 0,

            ((3,5), 0): 0,
            ((3,5), 1): 0,
            ((3,5), 2): 0,
            ((3,5), 3): 0,
            ((3,5), 4): 0,
            ((3,5), 5): 0,

        }
        self.rewards = {
    # absorbing state gets no reward
            ((4,-1), 0): 0,
            ((4,-1), 1): 0,
            ((4,-1), 2): 0,
            ((4,-1), 3): 0,
            ((4,-1), 4): 0,
            ((4,-1), 5): 0,


            ((0,0), 0): 0,
            ((0,0), 1): 0,
            ((0,0), 2): 0,
            ((0,0), 3): 0,
            ((0,0), 4): 0,
            ((0,0), 5): 0,

            ((1,2), 0): 0,
            ((1,2), 1): 0,
            ((1,2), 2): 0,
            ((1,2), 3): 0,
            ((1,2), 4): 0,
            ((1,2), 5): 0,

            ((2,-1), 0): 0,
            ((2,-1), 1): 0,
            ((2,-1), 2): self.big_reward,
            ((2,-1), 3): 0,
            ((2,-1), 4): 0,
            ((2,-1), 5): 0,

            ((3,4), 0): 0,
            ((3,4), 1): 0,
            ((3,4), 2): 0,
            ((3,4), 3): 0,
            ((3,4), 4): 0,
            ((3,4), 5): 0,

            ((3,5), 0): 0,
            ((3,5), 1): 0,
            ((3,5), 2): 0,
            ((3,5), 3): 0,
            ((3,5), 4): 0,
            ((3,5), 5): 0,

        }
        self.starting_states = [(0,0)]
        self.terminal_states = [(3,4),(4,-1)]


class MultiTaskDebugTaskABDSIX(MultiTaskDebug):
    def __init__(self):
        MultiTaskDebug.__init__(self)
        self.states = [0, 1, 2, 3]  # 4 is an absorbing state
        self.keys=[0,1,2,3,4,5]
        self.actions = [0, 1, 2, 3, 4, 5] # 0->A , 1->B, 2->C, 3->D, 4->E, 5->F
        self.transitions = {
            # absorbing state
            ((4,-1), 0): (4,-1),
            ((4,-1), 1): (4,-1),
            ((4,-1), 2): (4,-1),
            ((4,-1), 3): (4,-1),
            ((4,-1), 4): (4,-1),
            ((4,-1), 5): (4,-1),


            ((0,0), 0): (1,2),
            ((0,0), 1): (0,0),
            ((0,0), 2): (0,0),
            ((0,0), 3): (0,0),
            ((0,0), 4): (0,0),
            ((0,0), 5): (0,0),

            ((1,2), 0): (1,2),
            ((1,2), 1): (2,-1),
            ((1,2), 2): (1,2),
            ((1,2), 3): (1,2),
            ((1,2), 4): (1,2),
            ((1,2), 5): (1,2),

            ((2,-1), 0): (2,-1),
            ((2,-1), 1): (2,-1),
            ((2,-1), 2): (3,4),
            ((2,-1), 3): (3,5),
            ((2,-1), 4): (2,-1),
            ((2,-1), 5): (2,-1),

            ((3,4), 0): (2,-1),
            ((3,4), 1): (2,-1),
            ((3,4), 2): (3, 4),
            ((3,4), 3): (3, 5),
            ((3,4), 4): (2,-1),
            ((3,4), 5): (2,-1),

            ((3,5), 0): (4,-1),
            ((3,5), 1): (4,-1),
            ((3,5), 2): (4,-1),
            ((3,5), 3): (4,-1),
            ((3,5), 4): (4,-1),
            ((3,5), 5): (4,-1),



        }
        # Qs are actually not correct
        self.Qs = {
            ((4,-1), 0): 0,
            ((4,-1), 1): 0,
            ((4,-1), 2): 0,
            ((4,-1), 3): 0,
            ((4,-1), 4): 0,
            ((4,-1), 5): 0,


            ((0,0), 0): 0,
            ((0,0), 1): 0,
            ((0,0), 2): 0,
            ((0,0), 3): 0,
            ((0,0), 4): 0,
            ((0,0), 5): 0,

            ((1,2), 0): 0,
            ((1,2), 1): 0,
            ((1,2), 2): 0,
            ((1,2), 3): 0,
            ((1,2), 4): 0,
            ((1,2), 5): 0,

            ((2,-1), 0): 0,
            ((2,-1), 1): 0,
            ((2,-1), 2): 0,
            ((2,-1), 3): 0,
            ((2,-1), 4): 0,
            ((2,-1), 5): 0,

            ((3,4), 0): 0,
            ((3,4), 1): 0,
            ((3,4), 2): 0,
            ((3,4), 3): 0,
            ((3,4), 4): 0,
            ((3,4), 5): 0,

            ((3,5), 0): 0,
            ((3,5), 1): 0,
            ((3,5), 2): 0,
            ((3,5), 3): 0,
            ((3,5), 4): 0,
            ((3,5), 5): 0,

        }
        self.rewards = {
    # absorbing state gets no reward
            ((4,-1), 0): 0,
            ((4,-1), 1): 0,
            ((4,-1), 2): 0,
            ((4,-1), 3): 0,
            ((4,-1), 4): 0,
            ((4,-1), 5): 0,


            ((0,0), 0): 0,
            ((0,0), 1): 0,
            ((0,0), 2): 0,
            ((0,0), 3): 0,
            ((0,0), 4): 0,
            ((0,0), 5): 0,

            ((1,2), 0): 0,
            ((1,2), 1): 0,
            ((1,2), 2): 0,
            ((1,2), 3): 0,
            ((1,2), 4): 0,
            ((1,2), 5): 0,

            ((2,-1), 0): 0,
            ((2,-1), 1): 0,
            ((2,-1), 2): 0,
            ((2,-1), 3): self.big_reward,
            ((2,-1), 4): 0,
            ((2,-1), 5): 0,

            ((3,4), 0): 0,
            ((3,4), 1): 0,
            ((3,4), 2): 0,
            ((3,4), 3): 0,
            ((3,4), 4): 0,
            ((3,4), 5): 0,

            ((3,5), 0): 0,
            ((3,5), 1): 0,
            ((3,5), 2): 0,
            ((3,5), 3): 0,
            ((3,5), 4): 0,
            ((3,5), 5): 0,

        }
        self.starting_states = [(0,0)]
        self.terminal_states = [(3,5),(4,-1)]

class MultiTaskDebugTaskAECSIX(MultiTaskDebug):
    def __init__(self):
        MultiTaskDebug.__init__(self)
        self.states = [0, 1, 2, 3]  # 4 is an absorbing state
        self.keys=[0,1,2,3,4,5]
        self.actions = [0, 1, 2, 3, 4, 5] # 0->A , 1->B, 2->C, 3->D, 4->E, 5->F
        self.transitions = {
            # absorbing state
            ((4,-1), 0): (4,-1),
            ((4,-1), 1): (4,-1),
            ((4,-1), 2): (4,-1),
            ((4,-1), 3): (4,-1),
            ((4,-1), 4): (4,-1),
            ((4,-1), 5): (4,-1),


            ((0,0), 0): (1,3),
            ((0,0), 1): (0,0),
            ((0,0), 2): (0,0),
            ((0,0), 3): (0,0),
            ((0,0), 4): (0,0),
            ((0,0), 5): (0,0),

            ((1,3), 0): (1,3),
            ((1,3), 1): (1,3),
            ((1,3), 2): (1,3),
            ((1,3), 3): (1,3),
            ((1,3), 4): (2,-1),
            ((1,3), 5): (1,3),

            ((2,-1), 0): (2,-1),
            ((2,-1), 1): (2,-1),
            ((2,-1), 2): (3,4),
            ((2,-1), 3): (3,5),
            ((2,-1), 4): (2,-1),
            ((2,-1), 5): (2,-1),

            ((3,5), 0): (2,-1),
            ((3,5), 1): (2,-1),
            ((3,5), 2): (3, 4),
            ((3,5), 3): (3, 5),
            ((3,5), 4): (2,-1),
            ((3,5), 5): (2,-1),

            ((3,4), 0): (4,-1),
            ((3,4), 1): (4,-1),
            ((3,4), 2): (4,-1),
            ((3,4), 3): (4,-1),
            ((3,4), 4): (4,-1),
            ((3,4), 5): (4,-1),


        }
        # Qs are actually not correct
        self.Qs = {
            ((4,-1), 0): 0,
            ((4,-1), 1): 0,
            ((4,-1), 2): 0,
            ((4,-1), 3): 0,
            ((4,-1), 4): 0,
            ((4,-1), 5): 0,


            ((0,0), 0): 0,
            ((0,0), 1): 0,
            ((0,0), 2): 0,
            ((0,0), 3): 0,
            ((0,0), 4): 0,
            ((0,0), 5): 0,

            ((1,3), 0): 0,
            ((1,3), 1): 0,
            ((1,3), 2): 0,
            ((1,3), 3): 0,
            ((1,3), 4): 0,
            ((1,3), 5): 0,

            ((2,-1), 0): 0,
            ((2,-1), 1): 0,
            ((2,-1), 2): 0,
            ((2,-1), 3): 0,
            ((2,-1), 4): 0,
            ((2,-1), 5): 0,

            ((3,4), 0): 0,
            ((3,4), 1): 0,
            ((3,4), 2): 0,
            ((3,4), 3): 0,
            ((3,4), 4): 0,
            ((3,4), 5): 0,

            ((3,5), 0): 0,
            ((3,5), 1): 0,
            ((3,5), 2): 0,
            ((3,5), 3): 0,
            ((3,5), 4): 0,
            ((3,5), 5): 0,

        }
        self.rewards = {
    # absorbing state gets no reward
            ((4,-1), 0): 0,
            ((4,-1), 1): 0,
            ((4,-1), 2): 0,
            ((4,-1), 3): 0,
            ((4,-1), 4): 0,
            ((4,-1), 5): 0,


            ((0,0), 0): 0,
            ((0,0), 1): 0,
            ((0,0), 2): 0,
            ((0,0), 3): 0,
            ((0,0), 4): 0,
            ((0,0), 5): 0,

            ((1,3), 0): 0,
            ((1,3), 1): 0,
            ((1,3), 2): 0,
            ((1,3), 3): 0,
            ((1,3), 4): 0,
            ((1,3), 5): 0,

            ((2,-1), 0): 0,
            ((2,-1), 1): 0,
            ((2,-1), 2): self.big_reward,
            ((2,-1), 3): 0,
            ((2,-1), 4): 0,
            ((2,-1), 5): 0,

            ((3,4), 0): 0,
            ((3,4), 1): 0,
            ((3,4), 2): 0,
            ((3,4), 3): 0,
            ((3,4), 4): 0,
            ((3,4), 5): 0,

            ((3,5), 0): 0,
            ((3,5), 1): 0,
            ((3,5), 2): 0,
            ((3,5), 3): 0,
            ((3,5), 4): 0,
            ((3,5), 5): 0,

        }
        self.starting_states = [(0,0)]
        self.terminal_states = [(3,4),(4,-1)]

class MultiTaskDebugTaskFBCSIX(MultiTaskDebug):
    def __init__(self):
        MultiTaskDebug.__init__(self)
        self.states = [0, 1, 2, 3]  # 4 is an absorbing state
        self.keys=[0,1,2,3,4,5]
        self.actions = [0, 1, 2, 3, 4, 5] # 0->A , 1->B, 2->C, 3->D, 4->E, 5->F
        self.transitions = {
            # absorbing state
            ((4,-1), 0): (4,-1),
            ((4,-1), 1): (4,-1),
            ((4,-1), 2): (4,-1),
            ((4,-1), 3): (4,-1),
            ((4,-1), 4): (4,-1),
            ((4,-1), 5): (4,-1),


            ((0,1), 0): (0,1),
            ((0,1), 1): (0,1),
            ((0,1), 2): (0,1),
            ((0,1), 3): (0,1),
            ((0,1), 4): (0,1),
            ((0,1), 5): (1,2),

            ((1,2), 0): (1,2),
            ((1,2), 1): (2,-1),
            ((1,2), 2): (1,2),
            ((1,2), 3): (1,2),
            ((1,2), 4): (1,2),
            ((1,2), 5): (1,2),

            ((2,-1), 0): (2,-1),
            ((2,-1), 1): (2,-1),
            ((2,-1), 2): (3,4),
            ((2,-1), 3): (3,5),
            ((2,-1), 4): (2,-1),
            ((2,-1), 5): (2,-1),

            ((3,4), 0): (4,-1),
            ((3,4), 1): (4,-1),
            ((3,4), 2): (4,-1),
            ((3,4), 3): (4,-1),
            ((3,4), 4): (4,-1),
            ((3,4), 5): (4,-1),

            ((3,5), 0): (2,-1),
            ((3,5), 1): (2,-1),
            ((3,5), 2): (3, 4),
            ((3,5), 3): (3, 5),
            ((3,5), 4): (2,-1),
            ((3,5), 5): (2,-1),



        }
        # Qs are actually not correct
        self.Qs = {
            ((4,-1), 0): 0,
            ((4,-1), 1): 0,
            ((4,-1), 2): 0,
            ((4,-1), 3): 0,
            ((4,-1), 4): 0,
            ((4,-1), 5): 0,


            ((0,1), 0): 0,
            ((0,1), 1): 0,
            ((0,1), 2): 0,
            ((0,1), 3): 0,
            ((0,1), 4): 0,
            ((0,1), 5): 0,

            ((1,2), 0): 0,
            ((1,2), 1): 0,
            ((1,2), 2): 0,
            ((1,2), 3): 0,
            ((1,2), 4): 0,
            ((1,2), 5): 0,

            ((2,-1), 0): 0,
            ((2,-1), 1): 0,
            ((2,-1), 2): 0,
            ((2,-1), 3): 0,
            ((2,-1), 4): 0,
            ((2,-1), 5): 0,

            ((3,4), 0): 0,
            ((3,4), 1): 0,
            ((3,4), 2): 0,
            ((3,4), 3): 0,
            ((3,4), 4): 0,
            ((3,4), 5): 0,

            ((3,5), 0): 0,
            ((3,5), 1): 0,
            ((3,5), 2): 0,
            ((3,5), 3): 0,
            ((3,5), 4): 0,
            ((3,5), 5): 0,

        }
        self.rewards = {
    # absorbing state gets no reward
            ((4,-1), 0): 0,
            ((4,-1), 1): 0,
            ((4,-1), 2): 0,
            ((4,-1), 3): 0,
            ((4,-1), 4): 0,
            ((4,-1), 5): 0,


            ((0,1), 0): 0,
            ((0,1), 1): 0,
            ((0,1), 2): 0,
            ((0,1), 3): 0,
            ((0,1), 4): 0,
            ((0,1), 5): 0,

            ((1,2), 0): 0,
            ((1,2), 1): 0,
            ((1,2), 2): 0,
            ((1,2), 3): 0,
            ((1,2), 4): 0,
            ((1,2), 5): 0,

            ((2,-1), 0): 0,
            ((2,-1), 1): 0,
            ((2,-1), 2): self.big_reward,
            ((2,-1), 3): 0,
            ((2,-1), 4): 0,
            ((2,-1), 5): 0,

            ((3,4), 0): 0,
            ((3,4), 1): 0,
            ((3,4), 2): 0,
            ((3,4), 3): 0,
            ((3,4), 4): 0,
            ((3,4), 5): 0,

            ((3,5), 0): 0,
            ((3,5), 1): 0,
            ((3,5), 2): 0,
            ((3,5), 3): 0,
            ((3,5), 4): 0,
            ((3,5), 5): 0,

        }
        self.starting_states = [(0,1)]
        self.terminal_states = [(3,4),(4,-1)]

class MultiTaskDebugTaskABCDLenFOUR(MultiTaskDebug):
    def __init__(self):
        MultiTaskDebug.__init__(self)
        self.states = [0, 1, 2, 3, 4]  # 5 is an absorbing state
        self.keys=[0,1,2,3,4,5,6,7]
        self.actions = [0, 1, 2, 3, 4, 5, 6, 7] # 0->A , 1->B, 2->C, 3->D, 4->E, 5->F, 6->G, 7->H
        self.transitions = {
            # absorbing state
            ((5,-1), 0): (5,-1),
            ((5,-1), 1): (5,-1),
            ((5,-1), 2): (5,-1),
            ((5,-1), 3): (5,-1),
            ((5,-1), 4): (5,-1),
            ((5,-1), 5): (5,-1),
            ((5,-1), 6): (5,-1),
            ((5,-1), 7): (5,-1),


            ((0,0), 0): (1,2),
            ((0,0), 1): (0,0),
            ((0,0), 2): (0,0),
            ((0,0), 3): (0,0),
            ((0,0), 4): (0,0),
            ((0,0), 5): (0,0),
            ((0,0), 6): (0,0),
            ((0,0), 7): (0,0),

            ((1,2), 0): (1,2),
            ((1,2), 1): (2,4),
            ((1,2), 2): (1,2),
            ((1,2), 3): (1,2),
            ((1,2), 4): (1,2),
            ((1,2), 5): (1,2),
            ((1,2), 6): (1,2),
            ((1,2), 7): (1,2),

            ((2,4), 0): (2,4),
            ((2,4), 1): (2,4),
            ((2,4), 2): (3,-1),
            ((2,4), 3): (2,4),
            ((2,4), 4): (2,4),
            ((2,4), 5): (2,4),
            ((2,4), 6): (2,4),
            ((2,4), 7): (2,4),

            ((3,-1), 0): (3,-1),
            ((3,-1), 1): (3,-1),
            ((3,-1), 2): (3,-1),
            ((3,-1), 3): (4,6),
            ((3,-1), 4): (3,-1),
            ((3,-1), 5): (3,-1),
            ((3,-1), 6): (3,-1),
            ((3,-1), 7): (4,7),

            ((4,6), 0): (5,-1),
            ((4,6), 1): (5,-1),
            ((4,6), 2): (5,-1),
            ((4,6), 3): (5,-1),
            ((4,6), 4): (5,-1),
            ((4,6), 5): (5,-1),
            ((4,6), 6): (5,-1),
            ((4,6), 7): (5,-1),

            ((4,7), 0): (3,-1),
            ((4,7), 1): (3,-1),
            ((4,7), 2): (3,-1),
            ((4,7), 3): (4, 6),
            ((4,7), 4): (3,-1),
            ((4,7), 5): (3,-1),
            ((4,7), 6): (3,-1),
            ((4,7), 7): (4, 7),


        }
        # Qs are actually not correct
        self.Qs = {
            ((5,-1), 0): 0,
            ((5,-1), 1): 0,
            ((5,-1), 2): 0,
            ((5,-1), 3): 0,
            ((5,-1), 4): 0,
            ((5,-1), 5): 0,
            ((5,-1), 6): 0,
            ((5,-1), 7): 0,


            ((0,0), 0): 0,
            ((0,0), 1): 0,
            ((0,0), 2): 0,
            ((0,0), 3): 0,
            ((0,0), 4): 0,
            ((0,0), 5): 0,
            ((0,0), 6): 0,
            ((0,0), 7): 0,

            ((1,2), 0): 0,
            ((1,2), 1): 0,
            ((1,2), 2): 0,
            ((1,2), 3): 0,
            ((1,2), 4): 0,
            ((1,2), 5): 0,
            ((1,2), 6): 0,
            ((1,2), 7): 0,

            ((2,4), 0): 0,
            ((2,4), 1): 0,
            ((2,4), 2): 0,
            ((2,4), 3): 0,
            ((2,4), 4): 0,
            ((2,4), 5): 0,
            ((2,4), 6): 0,
            ((2,4), 7): 0,

            ((3,-1), 0): 0,
            ((3,-1), 1): 0,
            ((3,-1), 2): 0,
            ((3,-1), 3): 0,
            ((3,-1), 4): 0,
            ((3,-1), 5): 0,
            ((3,-1), 6): 0,
            ((3,-1), 7): 0,

            ((4,6), 0): 0,
            ((4,6), 1): 0,
            ((4,6), 2): 0,
            ((4,6), 3): 0,
            ((4,6), 4): 0,
            ((4,6), 5): 0,
            ((4,6), 6): 0,
            ((4,6), 7): 0,

            ((4,7), 0): 0,
            ((4,7), 1): 0,
            ((4,7), 2): 0,
            ((4,7), 3): 0,
            ((4,7), 4): 0,
            ((4,7), 5): 0,
            ((4,7), 6): 0,
            ((4,7), 7): 0,

        }
        self.rewards = {
    # absorbing state gets no reward
            ((5,-1), 0): 0,
            ((5,-1), 1): 0,
            ((5,-1), 2): 0,
            ((5,-1), 3): 0,
            ((5,-1), 4): 0,
            ((5,-1), 5): 0,
            ((5,-1), 6): 0,
            ((5,-1), 7): 0,


            ((0,0), 0): 0,
            ((0,0), 1): 0,
            ((0,0), 2): 0,
            ((0,0), 3): 0,
            ((0,0), 4): 0,
            ((0,0), 5): 0,
            ((0,0), 6): 0,
            ((0,0), 7): 0,

            ((1,2), 0): 0,
            ((1,2), 1): 0,
            ((1,2), 2): 0,
            ((1,2), 3): 0,
            ((1,2), 4): 0,
            ((1,2), 5): 0,
            ((1,2), 6): 0,
            ((1,2), 7): 0,

            ((2,4), 0): 0,
            ((2,4), 1): 0,
            ((2,4), 2): 0,
            ((2,4), 3): 0,
            ((2,4), 4): 0,
            ((2,4), 5): 0,
            ((2,4), 6): 0,
            ((2,4), 7): 0,

            ((3,-1), 0): 0,
            ((3,-1), 1): 0,
            ((3,-1), 2): 0,
            ((3,-1), 3): self.big_reward,
            ((3,-1), 4): 0,
            ((3,-1), 5): 0,
            ((3,-1), 6): 0,
            ((3,-1), 7): 0,

            ((4,6), 0): 0,
            ((4,6), 1): 0,
            ((4,6), 2): 0,
            ((4,6), 3): 0,
            ((4,6), 4): 0,
            ((4,6), 5): 0,
            ((4,6), 6): 0,
            ((4,6), 7): 0,

            ((4,7), 0): 0,
            ((4,7), 1): 0,
            ((4,7), 2): 0,
            ((4,7), 3): 0,
            ((4,7), 4): 0,
            ((4,7), 5): 0,
            ((4,7), 6): 0,
            ((4,7), 7): 0,
        }
        self.starting_states = [(0,0)]
        self.terminal_states = [(4,6),(5,-1)]

class MultiTaskDebugTaskABCHLenFOUR(MultiTaskDebug):
    def __init__(self):
        MultiTaskDebug.__init__(self)
        self.states = [0, 1, 2, 3, 4]  # 5 is an absorbing state
        self.keys=[0,1,2,3,4,5,6,7]
        self.actions = [0, 1, 2, 3, 4, 5, 6, 7] # 0->A , 1->B, 2->C, 3->D, 4->E, 5->F, 6->G, 7->H
        self.transitions = {
            # absorbing state
            ((5,-1), 0): (5,-1),
            ((5,-1), 1): (5,-1),
            ((5,-1), 2): (5,-1),
            ((5,-1), 3): (5,-1),
            ((5,-1), 4): (5,-1),
            ((5,-1), 5): (5,-1),
            ((5,-1), 6): (5,-1),
            ((5,-1), 7): (5,-1),


            ((0,0), 0): (1,2),
            ((0,0), 1): (0,0),
            ((0,0), 2): (0,0),
            ((0,0), 3): (0,0),
            ((0,0), 4): (0,0),
            ((0,0), 5): (0,0),
            ((0,0), 6): (0,0),
            ((0,0), 7): (0,0),

            ((1,2), 0): (1,2),
            ((1,2), 1): (2,4),
            ((1,2), 2): (1,2),
            ((1,2), 3): (1,2),
            ((1,2), 4): (1,2),
            ((1,2), 5): (1,2),
            ((1,2), 6): (1,2),
            ((1,2), 7): (1,2),

            ((2,4), 0): (2,4),
            ((2,4), 1): (2,4),
            ((2,4), 2): (3,-1),
            ((2,4), 3): (2,4),
            ((2,4), 4): (2,4),
            ((2,4), 5): (2,4),
            ((2,4), 6): (2,4),
            ((2,4), 7): (2,4),

            ((3,-1), 0): (3,-1),
            ((3,-1), 1): (3,-1),
            ((3,-1), 2): (3,-1),
            ((3,-1), 3): (4,6),
            ((3,-1), 4): (3,-1),
            ((3,-1), 5): (3,-1),
            ((3,-1), 6): (3,-1),
            ((3,-1), 7): (4,7),

            ((4,7), 0): (5,-1),
            ((4,7), 1): (5,-1),
            ((4,7), 2): (5,-1),
            ((4,7), 3): (5,-1),
            ((4,7), 4): (5,-1),
            ((4,7), 5): (5,-1),
            ((4,7), 6): (5,-1),
            ((4,7), 7): (5,-1),

            ((4,6), 0): (3,-1),
            ((4,6), 1): (3,-1),
            ((4,6), 2): (3,-1),
            ((4,6), 3): (4, 6),
            ((4,6), 4): (3,-1),
            ((4,6), 5): (3,-1),
            ((4,6), 6): (3,-1),
            ((4,6), 7): (4, 7),


        }
        # Qs are actually not correct
        self.Qs = {
            ((5,-1), 0): 0,
            ((5,-1), 1): 0,
            ((5,-1), 2): 0,
            ((5,-1), 3): 0,
            ((5,-1), 4): 0,
            ((5,-1), 5): 0,
            ((5,-1), 6): 0,
            ((5,-1), 7): 0,


            ((0,0), 0): 0,
            ((0,0), 1): 0,
            ((0,0), 2): 0,
            ((0,0), 3): 0,
            ((0,0), 4): 0,
            ((0,0), 5): 0,
            ((0,0), 6): 0,
            ((0,0), 7): 0,

            ((1,2), 0): 0,
            ((1,2), 1): 0,
            ((1,2), 2): 0,
            ((1,2), 3): 0,
            ((1,2), 4): 0,
            ((1,2), 5): 0,
            ((1,2), 6): 0,
            ((1,2), 7): 0,

            ((2,4), 0): 0,
            ((2,4), 1): 0,
            ((2,4), 2): 0,
            ((2,4), 3): 0,
            ((2,4), 4): 0,
            ((2,4), 5): 0,
            ((2,4), 6): 0,
            ((2,4), 7): 0,

            ((3,-1), 0): 0,
            ((3,-1), 1): 0,
            ((3,-1), 2): 0,
            ((3,-1), 3): 0,
            ((3,-1), 4): 0,
            ((3,-1), 5): 0,
            ((3,-1), 6): 0,
            ((3,-1), 7): 0,

            ((4,6), 0): 0,
            ((4,6), 1): 0,
            ((4,6), 2): 0,
            ((4,6), 3): 0,
            ((4,6), 4): 0,
            ((4,6), 5): 0,
            ((4,6), 6): 0,
            ((4,6), 7): 0,

            ((4,7), 0): 0,
            ((4,7), 1): 0,
            ((4,7), 2): 0,
            ((4,7), 3): 0,
            ((4,7), 4): 0,
            ((4,7), 5): 0,
            ((4,7), 6): 0,
            ((4,7), 7): 0,

        }
        self.rewards = {
    # absorbing state gets no reward
            ((5,-1), 0): 0,
            ((5,-1), 1): 0,
            ((5,-1), 2): 0,
            ((5,-1), 3): 0,
            ((5,-1), 4): 0,
            ((5,-1), 5): 0,
            ((5,-1), 6): 0,
            ((5,-1), 7): 0,


            ((0,0), 0): 0,
            ((0,0), 1): 0,
            ((0,0), 2): 0,
            ((0,0), 3): 0,
            ((0,0), 4): 0,
            ((0,0), 5): 0,
            ((0,0), 6): 0,
            ((0,0), 7): 0,

            ((1,2), 0): 0,
            ((1,2), 1): 0,
            ((1,2), 2): 0,
            ((1,2), 3): 0,
            ((1,2), 4): 0,
            ((1,2), 5): 0,
            ((1,2), 6): 0,
            ((1,2), 7): 0,

            ((2,4), 0): 0,
            ((2,4), 1): 0,
            ((2,4), 2): 0,
            ((2,4), 3): 0,
            ((2,4), 4): 0,
            ((2,4), 5): 0,
            ((2,4), 6): 0,
            ((2,4), 7): 0,

            ((3,-1), 0): 0,
            ((3,-1), 1): 0,
            ((3,-1), 2): 0,
            ((3,-1), 3): 0,
            ((3,-1), 4): 0,
            ((3,-1), 5): 0,
            ((3,-1), 6): 0,
            ((3,-1), 7): self.big_reward,

            ((4,6), 0): 0,
            ((4,6), 1): 0,
            ((4,6), 2): 0,
            ((4,6), 3): 0,
            ((4,6), 4): 0,
            ((4,6), 5): 0,
            ((4,6), 6): 0,
            ((4,6), 7): 0,

            ((4,7), 0): 0,
            ((4,7), 1): 0,
            ((4,7), 2): 0,
            ((4,7), 3): 0,
            ((4,7), 4): 0,
            ((4,7), 5): 0,
            ((4,7), 6): 0,
            ((4,7), 7): 0,
        }
        self.starting_states = [(0,0)]
        self.terminal_states = [(4,7),(5,-1)]

class MultiTaskDebugTaskABCDELenFIVE(MultiTaskDebug):
    def __init__(self):
        MultiTaskDebug.__init__(self)
        self.states = [0, 1, 2, 3, 4, 5]  # 6 is an absorbing state
        self.keys=[0,1,2,3,4,5,6,7,8, 9]
        self.actions = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] # 0->A , 1->B, 2->C, 3->D, 4->E, 5->F, 6->G, 7->H
        self.transitions = {
            # absorbing state
            ((5,-1), 0): (5,-1),
            ((5,-1), 1): (5,-1),
            ((5,-1), 2): (5,-1),
            ((5,-1), 3): (5,-1),
            ((5,-1), 4): (5,-1),
            ((5,-1), 5): (5,-1),
            ((5,-1), 6): (5,-1),
            ((5,-1), 7): (5,-1),
            ((5,-1), 8): (5,-1),
            ((5,-1), 9): (5,-1),

            ((0,0), 0): (1,2),
            ((0,0), 1): (0,0),
            ((0,0), 2): (0,0),
            ((0,0), 3): (0,0),
            ((0,0), 4): (0,0),
            ((0,0), 5): (0,0),
            ((0,0), 6): (0,0),
            ((0,0), 7): (0,0),
            ((0,0), 8): (0,0),
            ((0,0), 9): (0,0),

            ((1,2), 0): (1,2),
            ((1,2), 1): (2,4),
            ((1,2), 2): (1,2),
            ((1,2), 3): (1,2),
            ((1,2), 4): (1,2),
            ((1,2), 5): (1,2),
            ((1,2), 6): (1,2),
            ((1,2), 7): (1,2),
            ((1,2), 8): (1,2),
            ((1,2), 9): (1,2),

            ((2,4), 0): (2,4),
            ((2,4), 1): (2,4),
            ((2,4), 2): (3,6),
            ((2,4), 3): (2,4),
            ((2,4), 4): (2,4),
            ((2,4), 5): (2,4),
            ((2,4), 6): (2,4),
            ((2,4), 7): (2,4),
            ((2,4), 8): (2,4),
            ((2,4), 9): (2,4),

            ((3,6), 0): (3,6),
            ((3,6), 1): (3,6),
            ((3,6), 2): (3,6),
            ((3,6), 3): (4,-1),
            ((3,6), 4): (3,6),
            ((3,6), 5): (3,6),
            ((3,6), 6): (3,6),
            ((3,6), 7): (3,6),
            ((3,6), 8): (3,6),
            ((3,6), 9): (3,6),

            ((4,-1), 0): (4,-1),
            ((4,-1), 1): (4,-1),
            ((4,-1), 2): (4,-1),
            ((4,-1), 3): (4,-1),
            ((4,-1), 4): (5, 8),
            ((4,-1), 5): (4,-1),
            ((4,-1), 6): (4,-1),
            ((4,-1), 7): (4,-1),
            ((4,-1), 8): (4,-1),
            ((4,-1), 9): (5, 9),

            ((5,8), 0): (6,-1),
            ((5,8), 1): (6,-1),
            ((5,8), 2): (6,-1),
            ((5,8), 3): (6,-1),
            ((5,8), 4): (6,-1),
            ((5,8), 5): (6,-1),
            ((5,8), 6): (6,-1),
            ((5,8), 7): (6,-1),
            ((5,8), 8): (6,-1),
            ((5,8), 9): (6,-1),

            ((5,9), 0): (4,-1),
            ((5,9), 1): (4,-1),
            ((5,9), 2): (4,-1),
            ((5,9), 3): (4,-1),
            ((5,9), 4): (5, 8),
            ((5,9), 5): (4,-1),
            ((5,9), 6): (4,-1),
            ((5,9), 7): (4,-1),
            ((5,9), 8): (4,-1),
            ((5,9), 9): (5, 9),

        }
        # Qs are actually not correct
        self.Qs = {
            ((5,-1), 0): 0,
            ((5,-1), 1): 0,
            ((5,-1), 2): 0,
            ((5,-1), 3): 0,
            ((5,-1), 4): 0,
            ((5,-1), 5): 0,
            ((5,-1), 6): 0,
            ((5,-1), 7): 0,
            ((5,-1), 8): 0,
            ((5,-1), 9): 0,

            ((0,0), 0): 0,
            ((0,0), 1): 0,
            ((0,0), 2): 0,
            ((0,0), 3): 0,
            ((0,0), 4): 0,
            ((0,0), 5): 0,
            ((0,0), 6): 0,
            ((0,0), 7): 0,
            ((0,0), 8): 0,
            ((0,0), 9): 0,

            ((1,2), 0): 0,
            ((1,2), 1): 0,
            ((1,2), 2): 0,
            ((1,2), 3): 0,
            ((1,2), 4): 0,
            ((1,2), 5): 0,
            ((1,2), 6): 0,
            ((1,2), 7): 0,
            ((1,2), 8): 0,
            ((1,2), 9): 0,

            ((2,4), 0): 0,
            ((2,4), 1): 0,
            ((2,4), 2): 0,
            ((2,4), 3): 0,
            ((2,4), 4): 0,
            ((2,4), 5): 0,
            ((2,4), 6): 0,
            ((2,4), 7): 0,
            ((2,4), 8): 0,
            ((2,4), 9): 0,

            ((3,6), 0): 0,
            ((3,6), 1): 0,
            ((3,6), 2): 0,
            ((3,6), 3): 0,
            ((3,6), 4): 0,
            ((3,6), 5): 0,
            ((3,6), 6): 0,
            ((3,6), 7): 0,
            ((3,6), 8): 0,
            ((3,6), 9): 0,

            ((4,-1), 0): 0,
            ((4,-1), 1): 0,
            ((4,-1), 2): 0,
            ((4,-1), 3): 0,
            ((4,-1), 4): 0,
            ((4,-1), 5): 0,
            ((4,-1), 6): 0,
            ((4,-1), 7): 0,
            ((4,-1), 8): 0,
            ((4,-1), 9): 0,

            ((5,8), 0): 0,
            ((5,8), 1): 0,
            ((5,8), 2): 0,
            ((5,8), 3): 0,
            ((5,8), 4): 0,
            ((5,8), 5): 0,
            ((5,8), 6): 0,
            ((5,8), 7): 0,
            ((5,8), 8): 0,
            ((5,8), 9): 0,

            ((5,9), 0): 0,
            ((5,9), 1): 0,
            ((5,9), 2): 0,
            ((5,9), 3): 0,
            ((5,9), 4): 0,
            ((5,9), 5): 0,
            ((5,9), 6): 0,
            ((5,9), 7): 0,
            ((5,9), 8): 0,
            ((5,9), 9): 0,

        }
        self.rewards = {
    # absorbing state gets no reward
            ((5,-1), 0): 0,
            ((5,-1), 1): 0,
            ((5,-1), 2): 0,
            ((5,-1), 3): 0,
            ((5,-1), 4): 0,
            ((5,-1), 5): 0,
            ((5,-1), 6): 0,
            ((5,-1), 7): 0,
            ((5,-1), 8): 0,
            ((5,-1), 9): 0,

            ((0,0), 0): 0,
            ((0,0), 1): 0,
            ((0,0), 2): 0,
            ((0,0), 3): 0,
            ((0,0), 4): 0,
            ((0,0), 5): 0,
            ((0,0), 6): 0,
            ((0,0), 7): 0,
            ((0,0), 8): 0,
            ((0,0), 9): 0,

            ((1,2), 0): 0,
            ((1,2), 1): 0,
            ((1,2), 2): 0,
            ((1,2), 3): 0,
            ((1,2), 4): 0,
            ((1,2), 5): 0,
            ((1,2), 6): 0,
            ((1,2), 7): 0,
            ((1,2), 8): 0,
            ((1,2), 9): 0,

            ((2,4), 0): 0,
            ((2,4), 1): 0,
            ((2,4), 2): 0,
            ((2,4), 3): 0,
            ((2,4), 4): 0,
            ((2,4), 5): 0,
            ((2,4), 6): 0,
            ((2,4), 7): 0,
            ((2,4), 8): 0,
            ((2,4), 9): 0,

            ((3,6), 0): 0,
            ((3,6), 1): 0,
            ((3,6), 2): 0,
            ((3,6), 3): 0,
            ((3,6), 4): 0,
            ((3,6), 5): 0,
            ((3,6), 6): 0,
            ((3,6), 7): 0,
            ((3,6), 8): 0,
            ((3,6), 9): 0,

            ((4,-1), 0): 0,
            ((4,-1), 1): 0,
            ((4,-1), 2): 0,
            ((4,-1), 3): 0,
            ((4,-1), 4): self.big_reward,
            ((4,-1), 5): 0,
            ((4,-1), 6): 0,
            ((4,-1), 7): 0,
            ((4,-1), 8): 0,
            ((4,-1), 9): 0,

            ((5,8), 0): 0,
            ((5,8), 1): 0,
            ((5,8), 2): 0,
            ((5,8), 3): 0,
            ((5,8), 4): 0,
            ((5,8), 5): 0,
            ((5,8), 6): 0,
            ((5,8), 7): 0,
            ((5,8), 8): 0,
            ((5,8), 9): 0,

            ((5,9), 0): 0,
            ((5,9), 1): 0,
            ((5,9), 2): 0,
            ((5,9), 3): 0,
            ((5,9), 4): 0,
            ((5,9), 5): 0,
            ((5,9), 6): 0,
            ((5,9), 7): 0,
            ((5,9), 8): 0,
            ((5,9), 9): 0,
        }
        self.starting_states = [(0,0)]
        self.terminal_states = [(5,8),(6,-1)]
    
class MultiTaskDebugTaskABCDJLenFIVE(MultiTaskDebug):
    def __init__(self):
        MultiTaskDebug.__init__(self)
        self.states = [0, 1, 2, 3, 4, 5]  # 6 is an absorbing state
        self.keys=[0,1,2,3,4,5,6,7,8, 9]
        self.actions = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] # 0->A , 1->B, 2->C, 3->D, 4->E, 5->F, 6->G, 7->H
        self.transitions = {
            # absorbing state
            ((5,-1), 0): (5,-1),
            ((5,-1), 1): (5,-1),
            ((5,-1), 2): (5,-1),
            ((5,-1), 3): (5,-1),
            ((5,-1), 4): (5,-1),
            ((5,-1), 5): (5,-1),
            ((5,-1), 6): (5,-1),
            ((5,-1), 7): (5,-1),
            ((5,-1), 8): (5,-1),
            ((5,-1), 9): (5,-1),

            ((0,0), 0): (1,2),
            ((0,0), 1): (0,0),
            ((0,0), 2): (0,0),
            ((0,0), 3): (0,0),
            ((0,0), 4): (0,0),
            ((0,0), 5): (0,0),
            ((0,0), 6): (0,0),
            ((0,0), 7): (0,0),
            ((0,0), 8): (0,0),
            ((0,0), 9): (0,0),

            ((1,2), 0): (1,2),
            ((1,2), 1): (2,4),
            ((1,2), 2): (1,2),
            ((1,2), 3): (1,2),
            ((1,2), 4): (1,2),
            ((1,2), 5): (1,2),
            ((1,2), 6): (1,2),
            ((1,2), 7): (1,2),
            ((1,2), 8): (1,2),
            ((1,2), 9): (1,2),

            ((2,4), 0): (2,4),
            ((2,4), 1): (2,4),
            ((2,4), 2): (3,6),
            ((2,4), 3): (2,4),
            ((2,4), 4): (2,4),
            ((2,4), 5): (2,4),
            ((2,4), 6): (2,4),
            ((2,4), 7): (2,4),
            ((2,4), 8): (2,4),
            ((2,4), 9): (2,4),

            ((3,6), 0): (3,6),
            ((3,6), 1): (3,6),
            ((3,6), 2): (3,6),
            ((3,6), 3): (4,-1),
            ((3,6), 4): (3,6),
            ((3,6), 5): (3,6),
            ((3,6), 6): (3,6),
            ((3,6), 7): (3,6),
            ((3,6), 8): (3,6),
            ((3,6), 9): (3,6),

            ((4,-1), 0): (4,-1),
            ((4,-1), 1): (4,-1),
            ((4,-1), 2): (4,-1),
            ((4,-1), 3): (4,-1),
            ((4,-1), 4): (5, 8),
            ((4,-1), 5): (4,-1),
            ((4,-1), 6): (4,-1),
            ((4,-1), 7): (4,-1),
            ((4,-1), 8): (4,-1),
            ((4,-1), 9): (5, 9),

            ((5,9), 0): (6,-1),
            ((5,9), 1): (6,-1),
            ((5,9), 2): (6,-1),
            ((5,9), 3): (6,-1),
            ((5,9), 4): (6,-1),
            ((5,9), 5): (6,-1),
            ((5,9), 6): (6,-1),
            ((5,9), 7): (6,-1),
            ((5,9), 8): (6,-1),
            ((5,9), 9): (6,-1),

            ((5,8), 0): (4,-1),
            ((5,8), 1): (4,-1),
            ((5,8), 2): (4,-1),
            ((5,8), 3): (4,-1),
            ((5,8), 4): (5, 8),
            ((5,8), 5): (4,-1),
            ((5,8), 6): (4,-1),
            ((5,8), 7): (4,-1),
            ((5,8), 8): (4,-1),
            ((5,8), 9): (5, 9),

        }
        # Qs are actually not correct
        self.Qs = {
            ((5,-1), 0): 0,
            ((5,-1), 1): 0,
            ((5,-1), 2): 0,
            ((5,-1), 3): 0,
            ((5,-1), 4): 0,
            ((5,-1), 5): 0,
            ((5,-1), 6): 0,
            ((5,-1), 7): 0,
            ((5,-1), 8): 0,
            ((5,-1), 9): 0,

            ((0,0), 0): 0,
            ((0,0), 1): 0,
            ((0,0), 2): 0,
            ((0,0), 3): 0,
            ((0,0), 4): 0,
            ((0,0), 5): 0,
            ((0,0), 6): 0,
            ((0,0), 7): 0,
            ((0,0), 8): 0,
            ((0,0), 9): 0,

            ((1,2), 0): 0,
            ((1,2), 1): 0,
            ((1,2), 2): 0,
            ((1,2), 3): 0,
            ((1,2), 4): 0,
            ((1,2), 5): 0,
            ((1,2), 6): 0,
            ((1,2), 7): 0,
            ((1,2), 8): 0,
            ((1,2), 9): 0,

            ((2,4), 0): 0,
            ((2,4), 1): 0,
            ((2,4), 2): 0,
            ((2,4), 3): 0,
            ((2,4), 4): 0,
            ((2,4), 5): 0,
            ((2,4), 6): 0,
            ((2,4), 7): 0,
            ((2,4), 8): 0,
            ((2,4), 9): 0,

            ((3,6), 0): 0,
            ((3,6), 1): 0,
            ((3,6), 2): 0,
            ((3,6), 3): 0,
            ((3,6), 4): 0,
            ((3,6), 5): 0,
            ((3,6), 6): 0,
            ((3,6), 7): 0,
            ((3,6), 8): 0,
            ((3,6), 9): 0,

            ((4,-1), 0): 0,
            ((4,-1), 1): 0,
            ((4,-1), 2): 0,
            ((4,-1), 3): 0,
            ((4,-1), 4): 0,
            ((4,-1), 5): 0,
            ((4,-1), 6): 0,
            ((4,-1), 7): 0,
            ((4,-1), 8): 0,
            ((4,-1), 9): 0,

            ((5,8), 0): 0,
            ((5,8), 1): 0,
            ((5,8), 2): 0,
            ((5,8), 3): 0,
            ((5,8), 4): 0,
            ((5,8), 5): 0,
            ((5,8), 6): 0,
            ((5,8), 7): 0,
            ((5,8), 8): 0,
            ((5,8), 9): 0,

            ((5,9), 0): 0,
            ((5,9), 1): 0,
            ((5,9), 2): 0,
            ((5,9), 3): 0,
            ((5,9), 4): 0,
            ((5,9), 5): 0,
            ((5,9), 6): 0,
            ((5,9), 7): 0,
            ((5,9), 8): 0,
            ((5,9), 9): 0,

        }
        self.rewards = {
    # absorbing state gets no reward
            ((5,-1), 0): 0,
            ((5,-1), 1): 0,
            ((5,-1), 2): 0,
            ((5,-1), 3): 0,
            ((5,-1), 4): 0,
            ((5,-1), 5): 0,
            ((5,-1), 6): 0,
            ((5,-1), 7): 0,
            ((5,-1), 8): 0,
            ((5,-1), 9): 0,

            ((0,0), 0): 0,
            ((0,0), 1): 0,
            ((0,0), 2): 0,
            ((0,0), 3): 0,
            ((0,0), 4): 0,
            ((0,0), 5): 0,
            ((0,0), 6): 0,
            ((0,0), 7): 0,
            ((0,0), 8): 0,
            ((0,0), 9): 0,

            ((1,2), 0): 0,
            ((1,2), 1): 0,
            ((1,2), 2): 0,
            ((1,2), 3): 0,
            ((1,2), 4): 0,
            ((1,2), 5): 0,
            ((1,2), 6): 0,
            ((1,2), 7): 0,
            ((1,2), 8): 0,
            ((1,2), 9): 0,

            ((2,4), 0): 0,
            ((2,4), 1): 0,
            ((2,4), 2): 0,
            ((2,4), 3): 0,
            ((2,4), 4): 0,
            ((2,4), 5): 0,
            ((2,4), 6): 0,
            ((2,4), 7): 0,
            ((2,4), 8): 0,
            ((2,4), 9): 0,

            ((3,6), 0): 0,
            ((3,6), 1): 0,
            ((3,6), 2): 0,
            ((3,6), 3): 0,
            ((3,6), 4): 0,
            ((3,6), 5): 0,
            ((3,6), 6): 0,
            ((3,6), 7): 0,
            ((3,6), 8): 0,
            ((3,6), 9): 0,

            ((4,-1), 0): 0,
            ((4,-1), 1): 0,
            ((4,-1), 2): 0,
            ((4,-1), 3): 0,
            ((4,-1), 4): 0,
            ((4,-1), 5): 0,
            ((4,-1), 6): 0,
            ((4,-1), 7): 0,
            ((4,-1), 8): 0,
            ((4,-1), 9): self.big_reward,

            ((5,8), 0): 0,
            ((5,8), 1): 0,
            ((5,8), 2): 0,
            ((5,8), 3): 0,
            ((5,8), 4): 0,
            ((5,8), 5): 0,
            ((5,8), 6): 0,
            ((5,8), 7): 0,
            ((5,8), 8): 0,
            ((5,8), 9): 0,

            ((5,9), 0): 0,
            ((5,9), 1): 0,
            ((5,9), 2): 0,
            ((5,9), 3): 0,
            ((5,9), 4): 0,
            ((5,9), 5): 0,
            ((5,9), 6): 0,
            ((5,9), 7): 0,
            ((5,9), 8): 0,
            ((5,9), 9): 0,
        }
        self.starting_states = [(0,0)]
        self.terminal_states = [(5,9),(6,-1)]


"""
The goal state transitions to the terminal state
The terminal state transitions to itself
The terminal self-transition gets 0 reward and 0 Q-value
The episode ends after you transition from goal state to terminal state
"""
class TransferDebug(MultiStepEnv):
    def __init__(self):
        MultiStepEnv.__init__(self)
        self.big_reward = 0.8
        self.eplen = 10

        self.states = [0, 1, 2, 3, 4]  # 5 is an absorbing state
        self.actions = [0, 1]
        self.transitions = {
            # absorbing state
            (5, 0): 5,
            (5, 1): 5,

            (0, 0): 0,  # self-transition
            (0, 1): 1,  # goes to the bottleneck state

            (3, 0): 1,  # goes to the bottleneck state
            (3, 1): 3,  # self-transition

            (1, 0): 4,
            (1, 1): 2,

            # both 2 and 4 go to the terminal state
            (2, 0): 5,
            (2, 1): 5,

            (4, 0): 5,
            (4, 1): 5,
        }
        # Qs are actually not correct
        self.Qs = {
            (5, 0): 0,
            (5, 1): 0,

            (0, 0): 0,
            (0, 1): 0,

            (1, 0): 0,
            (1, 1): 0,

            (2, 0): 0,
            (2, 1): 0,

            (3, 0): 0,
            (3, 1): 0,

            (4, 0): 0,
            (4, 1): 0,
        }

    def step(self, action):
        next_state = self.transitions[self.state, action]
        self.counter += 1
        done = next_state in self.terminal_states
        reward = self.rewards[self.state, action]
        self.state = next_state
        return self.to_onehot(next_state), reward, done, {}

class TransferDebugInvar(MultiStepEnv):
    def __init__(self):
        MultiStepEnv.__init__(self)
        self.big_reward = 0.8
        self.eplen = 10

        self.states = [0, 1, 2, 3, 4, 5]  # 6 is an absorbing state
        self.actions = [0, 1, 2, 3]
        self.transitions = {
            # absorbing state
            (6, 0): 6,
            (6, 1): 6,
            (6, 2): 6,
            (6, 3): 6,

            (0, 0): 0,
            (0, 1): 0,
            (0, 2): 0,
            (0, 3): 0,

            (1, 0): 1,
            (1, 1): 1,
            (1, 2): 1,
            (1, 3): 1,

            (2, 0): 2,
            (2, 1): 2,
            (2, 2): 2,
            (2, 3): 2,

            (3, 0): 3,
            (3, 1): 3,
            (3, 2): 3,
            (3, 3): 3,

            (4, 0): 4,
            (4, 1): 4,
            (4, 2): 4,
            (4, 3): 4,

            (5, 0): 5,
            (5, 1): 5,
            (5, 2): 5,
            (5, 3): 5,

        }
        # Qs are actually not correct
        self.Qs = {
            (6, 0): 0,
            (6, 1): 0,
            (6, 2): 0,
            (6, 3): 0,

            (0, 0): 0,
            (0, 1): 0,
            (0, 2): 0,
            (0, 3): 0,

            (1, 0): 0,
            (1, 1): 0,
            (1, 2): 0,
            (1, 3): 0,

            (2, 0): 0,
            (2, 1): 0,
            (2, 2): 0,
            (2, 3): 0,

            (3, 0): 0,
            (3, 1): 0,
            (3, 2): 0,
            (3, 3): 0,

            (4, 0): 0,
            (4, 1): 0,
            (4, 2): 0,
            (4, 3): 0,

            (5, 0): 0,
            (5, 1): 0,
            (5, 2): 0,
            (5, 3): 0,
        }
        self.rewards = {
            (6, 0): 0,
            (6, 1): 0,
            (6, 2): 0,
            (6, 3): 0,

            (0, 0): 0,
            (0, 1): 0,
            (0, 2): 0,
            (0, 3): 0,

            (1, 0): 0,
            (1, 1): 0,
            (1, 2): 0,
            (1, 3): 0,

            (2, 0): 0,
            (2, 1): 0,
            (2, 2): 0,
            (2, 3): 0,

            (3, 0): 0,
            (3, 1): 0,
            (3, 2): 0,
            (3, 3): 0,

            (4, 0): 0,
            (4, 1): 0,
            (4, 2): 0,
            (4, 3): 0,

            (5, 0): 0,
            (5, 1): 0,
            (5, 2): 0,
            (5, 3): 0,

            }

        self.starting_states = [0]
        self.terminal_states = [4, 5, 6]

    def step(self, action):
        next_state = self.transitions[self.state, action]
        self.counter += 1
        done = next_state in self.terminal_states
        reward = self.rewards[self.state, action]
        self.state = next_state
        return self.to_onehot(next_state), reward, done, {}

class TransferDebugInvarOneHotV1(TransferDebugInvar):
    def __init__(self):
        TransferDebugInvar.__init__(self)

        self.transitions[(0, 0)]= 2
        self.transitions[(2, 2)]= 4

        self.rewards[(2, 2)]= self.big_reward

        self.starting_states = [0]
        self.terminal_states = [4, 5, 6]

class TransferDebugInvarOneHotV2(TransferDebugInvar):
    def __init__(self):
        TransferDebugInvar.__init__(self)

        self.transitions[(1, 1)]= 3
        self.transitions[(3, 3)]= 5

        self.rewards[(3, 3)]= self.big_reward

        self.starting_states = [1]
        self.terminal_states = [4, 5, 6]

class TransferDebugParallelOneHotV1(TransferDebugInvar):
    def __init__(self):
        TransferDebugInvar.__init__(self)

        self.transitions[(0, 0)]= 3
        self.transitions[(3, 3)]= 5

        self.rewards[(3, 3)]= self.big_reward

        self.starting_states = [0]
        self.terminal_states = [4, 5, 6]

class TransferDebugParallelOneHotV2(TransferDebugInvar):
    def __init__(self):
        TransferDebugInvar.__init__(self)

        self.transitions[(1, 1)]= 2
        self.transitions[(2, 2)]= 4

        self.rewards[(2, 2)]= self.big_reward

        self.starting_states = [1]
        self.terminal_states = [4, 5, 6]


class TransferDebugCommonAncestor(MultiStepEnv):
    def __init__(self):
        MultiStepEnv.__init__(self)
        self.big_reward = 0.8
        self.eplen = 10

        self.states = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # 10 is an absorbing state
        self.actions = [0, 1, 2, 3, 4, 5]
        self.transitions = {
            # absorbing state
            (10, 0): 10,
            (10, 1): 10,
            (10, 2): 10,
            (10, 3): 10,
            (10, 4): 10,
            (10, 5): 10,

            (0, 0): 0,
            (0, 1): 0,
            (0, 2): 0,
            (0, 3): 0,
            (0, 4): 0,
            (0, 5): 0,

            (1, 0): 1,
            (1, 1): 1,
            (1, 2): 1,
            (1, 3): 1,
            (1, 4): 1,
            (1, 5): 1,

            (2, 0): 2,
            (2, 1): 2,
            (2, 2): 2,
            (2, 3): 2,
            (2, 4): 2,
            (2, 5): 2,

            (3, 0): 3,
            (3, 1): 3,
            (3, 2): 3,
            (3, 3): 3,
            (3, 4): 3,
            (3, 5): 3,

            (4, 0): 4,
            (4, 1): 4,
            (4, 2): 4,
            (4, 3): 4,
            (4, 4): 4,
            (4, 5): 4,

            (5, 0): 5,
            (5, 1): 5,
            (5, 2): 5,
            (5, 3): 5,
            (5, 4): 5,
            (5, 5): 5,

            (6, 0): 10,
            (6, 1): 10,
            (6, 2): 10,
            (6, 3): 10,
            (6, 4): 10,
            (6, 5): 10,

            (7, 0): 10,
            (7, 1): 10,
            (7, 2): 10,
            (7, 3): 10,
            (7, 4): 10,
            (7, 5): 10,

            (8, 0): 10,
            (8, 1): 10,
            (8, 2): 10,
            (8, 3): 10,
            (8, 4): 10,
            (8, 5): 10,

            (9, 0): 10,
            (9, 1): 10,
            (9, 2): 10,
            (9, 3): 10,
            (9, 4): 10,
            (9, 5): 10,

        }
        # Qs are actually not correct
        self.Qs = {
            (10, 0): 0,
            (10, 1): 0,
            (10, 2): 0,
            (10, 3): 0,
            (10, 4): 0,
            (10, 5): 0,

            (0, 0): 0,
            (0, 1): 0,
            (0, 2): 0,
            (0, 3): 0,
            (0, 4): 0,
            (0, 5): 0,

            (1, 0): 0,
            (1, 1): 0,
            (1, 2): 0,
            (1, 3): 0,
            (1, 4): 0,
            (1, 5): 0,

            (2, 0): 0,
            (2, 1): 0,
            (2, 2): 0,
            (2, 3): 0,
            (2, 4): 0,
            (2, 5): 0,

            (3, 0): 0,
            (3, 1): 0,
            (3, 2): 0,
            (3, 3): 0,
            (3, 4): 0,
            (3, 5): 0,

            (4, 0): 0,
            (4, 1): 0,
            (4, 2): 0,
            (4, 3): 0,
            (4, 4): 0,
            (4, 5): 0,

            (5, 0): 0,
            (5, 1): 0,
            (5, 2): 0,
            (5, 3): 0,
            (5, 4): 0,
            (5, 5): 0,

            (6, 0): 0,
            (6, 1): 0,
            (6, 2): 0,
            (6, 3): 0,
            (6, 4): 0,
            (6, 5): 0,

            (7, 0): 0,
            (7, 1): 0,
            (7, 2): 0,
            (7, 3): 0,
            (7, 4): 0,
            (7, 5): 0,

            (8, 0): 0,
            (8, 1): 0,
            (8, 2): 0,
            (8, 3): 0,
            (8, 4): 0,
            (8, 5): 0,

            (9, 0): 0,
            (9, 1): 0,
            (9, 2): 0,
            (9, 3): 0,
            (9, 4): 0,
            (9, 5): 0,
        }
        self.rewards = {
            # absorbing state gets no reward
            (10, 0): 0,
            (10, 1): 0,
            (10, 2): 0,
            (10, 3): 0,
            (10, 4): 0,
            (10, 5): 0,

            (0, 0): 0,
            (0, 1): 0,
            (0, 2): 0,
            (0, 3): 0,
            (0, 4): 0,
            (0, 5): 0,

            (1, 0): 0,
            (1, 1): 0,
            (1, 2): 0,
            (1, 3): 0,
            (1, 4): 0,
            (1, 5): 0,

            (2, 0): 0,
            (2, 1): 0,
            (2, 2): 0,
            (2, 3): 0,
            (2, 4): 0,
            (2, 5): 0,

            (3, 0): 0,
            (3, 1): 0,
            (3, 2): 0,
            (3, 3): 0,
            (3, 4): 0,
            (3, 5): 0,

            (4, 0): 0,
            (4, 1): 0,
            (4, 2): 0,
            (4, 3): 0,
            (4, 4): 0,
            (4, 5): 0,

            (5, 0): 0,
            (5, 1): 0,
            (5, 2): 0,
            (5, 3): 0,
            (5, 4): 0,
            (5, 5): 0,

            (6, 0): 0,
            (6, 1): 0,
            (6, 2): 0,
            (6, 3): 0,
            (6, 4): 0,
            (6, 5): 0,

            (7, 0): 0,
            (7, 1): 0,
            (7, 2): 0,
            (7, 3): 0,
            (7, 4): 0,
            (7, 5): 0,

            (8, 0): 0,
            (8, 1): 0,
            (8, 2): 0,
            (8, 3): 0,
            (8, 4): 0,
            (8, 5): 0,

            (9, 0): 0,
            (9, 1): 0,
            (9, 2): 0,
            (9, 3): 0,
            (9, 4): 0,
            (9, 5): 0,
        }
        self.starting_states = [0]
        self.terminal_states = [6, 7, 8, 9, 10]

    def step(self, action):
        next_state = self.transitions[self.state, action]
        self.counter += 1
        done = next_state in self.terminal_states
        reward = self.rewards[self.state, action]
        self.state = next_state
        return self.to_onehot(next_state), reward, done, {}

class TransferDebugCommonAncestorOneHotAB(TransferDebugCommonAncestor):
    def __init__(self):
        TransferDebugCommonAncestor.__init__(self)

        self.transitions[(0, 0)]= 2
        self.transitions[(2, 1)]= 6


        self.rewards[(2, 1)]= self.big_reward

        self.starting_states = [0]
        self.terminal_states = [6, 7, 8, 9, 10]

class TransferDebugCommonAncestorOneHotAC(TransferDebugCommonAncestor):
    def __init__(self):
        TransferDebugCommonAncestor.__init__(self)

        self.transitions[(0, 0)]= 3
        self.transitions[(3, 2)]= 7


        self.rewards[(3, 2)]= self.big_reward

        self.starting_states = [0]
        self.terminal_states = [6, 7, 8, 9, 10]

class TransferDebugCommonAncestorOneHotDB(TransferDebugCommonAncestor):
    def __init__(self):
        TransferDebugCommonAncestor.__init__(self)

        self.transitions[(1, 3)]= 2
        self.transitions[(2, 1)]= 6


        self.rewards[(2, 1)]= self.big_reward

        self.starting_states = [1]
        self.terminal_states = [6, 7, 8, 9, 10]

class TransferDebugCommonAncestorOneHotDC(TransferDebugCommonAncestor):
    def __init__(self):
        TransferDebugCommonAncestor.__init__(self)

        self.transitions[(1, 3)]= 3
        self.transitions[(3, 2)]= 7


        self.rewards[(3, 2)]= self.big_reward

        self.starting_states = [1]
        self.terminal_states = [6, 7, 8, 9, 10]


class TransferDebugCommonAncestorOneHotAE(TransferDebugCommonAncestor):
    def __init__(self):
        TransferDebugCommonAncestor.__init__(self)

        self.transitions[(0, 0)]= 4
        self.transitions[(4, 4)]= 8


        self.rewards[(4, 4)]= self.big_reward

        self.starting_states = [0]
        self.terminal_states = [6, 7, 8, 9, 10]

class TransferDebugCommonAncestorOneHotAF(TransferDebugCommonAncestor):
    def __init__(self):
        TransferDebugCommonAncestor.__init__(self)

        self.transitions[(0, 0)]= 4
        self.transitions[(5, 5)]= 9


        self.rewards[(5, 5)]= self.big_reward

        self.starting_states = [0]
        self.terminal_states = [6, 7, 8, 9, 10]

class TransferDebugCommonDescedent(MultiStepEnv):
    def __init__(self):
        MultiStepEnv.__init__(self)
        self.big_reward = 0.8
        self.eplen = 10

        self.states = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # 10 is an absorbing state
        self.actions = [0, 1, 2, 3, 4, 5]
        self.transitions = {
            # absorbing state
            (10, 0): 10,
            (10, 1): 10,
            (10, 2): 10,
            (10, 3): 10,
            (10, 4): 10,
            (10, 5): 10,

            (0, 0): 0,
            (0, 1): 0,
            (0, 2): 0,
            (0, 3): 0,
            (0, 4): 0,
            (0, 5): 0,

            (1, 0): 1,
            (1, 1): 1,
            (1, 2): 1,
            (1, 3): 1,
            (1, 4): 1,
            (1, 5): 1,

            (2, 0): 0,
            (2, 1): 0,
            (2, 2): 0,
            (2, 3): 0,
            (2, 4): 0,
            (2, 5): 0,

            (3, 0): 3,
            (3, 1): 3,
            (3, 2): 3,
            (3, 3): 3,
            (3, 4): 3,
            (3, 5): 3,

            (4, 0): 4,
            (4, 1): 4,
            (4, 2): 4,
            (4, 3): 4,
            (4, 4): 4,
            (4, 5): 4,

            (5, 0): 5,
            (5, 1): 5,
            (5, 2): 5,
            (5, 3): 5,
            (5, 4): 5,
            (5, 5): 5,

            (6, 0): 6,
            (6, 1): 6,
            (6, 2): 6,
            (6, 3): 6,
            (6, 4): 6,
            (6, 5): 6,

            (7, 0): 7,
            (7, 1): 7,
            (7, 2): 7,
            (7, 3): 7,
            (7, 4): 7,
            (7, 5): 7,

            (8, 0): 10,
            (8, 1): 10,
            (8, 2): 10,
            (8, 3): 10,
            (8, 4): 10,
            (8, 5): 10,

            (9, 0): 10,
            (9, 1): 10,
            (9, 2): 10,
            (9, 3): 10,
            (9, 4): 10,
            (9, 5): 10,

        }
        # Qs are actually not correct
        self.Qs = {
            (10, 0): 0,
            (10, 1): 0,
            (10, 2): 0,
            (10, 3): 0,
            (10, 4): 0,
            (10, 5): 0,

            (0, 0): 0,
            (0, 1): 0,
            (0, 2): 0,
            (0, 3): 0,
            (0, 4): 0,
            (0, 5): 0,

            (1, 0): 0,
            (1, 1): 0,
            (1, 2): 0,
            (1, 3): 0,
            (1, 4): 0,
            (1, 5): 0,

            (2, 0): 0,
            (2, 1): 0,
            (2, 2): 0,
            (2, 3): 0,
            (2, 4): 0,
            (2, 5): 0,

            (3, 0): 0,
            (3, 1): 0,
            (3, 2): 0,
            (3, 3): 0,
            (3, 4): 0,
            (3, 5): 0,

            (4, 0): 0,
            (4, 1): 0,
            (4, 2): 0,
            (4, 3): 0,
            (4, 4): 0,
            (4, 5): 0,

            (5, 0): 0,
            (5, 1): 0,
            (5, 2): 0,
            (5, 3): 0,
            (5, 4): 0,
            (5, 5): 0,

            (6, 0): 0,
            (6, 1): 0,
            (6, 2): 0,
            (6, 3): 0,
            (6, 4): 0,
            (6, 5): 0,

            (7, 0): 0,
            (7, 1): 0,
            (7, 2): 0,
            (7, 3): 0,
            (7, 4): 0,
            (7, 5): 0,

            (8, 0): 0,
            (8, 1): 0,
            (8, 2): 0,
            (8, 3): 0,
            (8, 4): 0,
            (8, 5): 0,

            (9, 0): 0,
            (9, 1): 0,
            (9, 2): 0,
            (9, 3): 0,
            (9, 4): 0,
            (9, 5): 0,
        }
        self.rewards = {
            # absorbing state gets no reward
            (10, 0): 0,
            (10, 1): 0,
            (10, 2): 0,
            (10, 3): 0,
            (10, 4): 0,
            (10, 5): 0,

            (0, 0): 0,
            (0, 1): 0,
            (0, 2): 0,
            (0, 3): 0,
            (0, 4): 0,
            (0, 5): 0,

            (1, 0): 0,
            (1, 1): 0,
            (1, 2): 0,
            (1, 3): 0,
            (1, 4): 0,
            (1, 5): 0,

            (2, 0): 0,
            (2, 1): 0,
            (2, 2): 0,
            (2, 3): 0,
            (2, 4): 0,
            (2, 5): 0,

            (3, 0): 0,
            (3, 1): 0,
            (3, 2): 0,
            (3, 3): 0,
            (3, 4): 0,
            (3, 5): 0,

            (4, 0): 0,
            (4, 1): 0,
            (4, 2): 0,
            (4, 3): 0,
            (4, 4): 0,
            (4, 5): 0,

            (5, 0): 0,
            (5, 1): 0,
            (5, 2): 0,
            (5, 3): 0,
            (5, 4): 0,
            (5, 5): 0,

            (6, 0): 0,
            (6, 1): 0,
            (6, 2): 0,
            (6, 3): 0,
            (6, 4): 0,
            (6, 5): 0,

            (7, 0): 0,
            (7, 1): 0,
            (7, 2): 0,
            (7, 3): 0,
            (7, 4): 0,
            (7, 5): 0,

            (8, 0): 0,
            (8, 1): 0,
            (8, 2): 0,
            (8, 3): 0,
            (8, 4): 0,
            (8, 5): 0,

            (9, 0): 0,
            (9, 1): 0,
            (9, 2): 0,
            (9, 3): 0,
            (9, 4): 0,
            (9, 5): 0,
        }
        self.starting_states = [0]
        self.terminal_states = [6, 7, 8, 9, 10]

    def step(self, action):
        next_state = self.transitions[self.state, action]
        self.counter += 1
        done = next_state in self.terminal_states
        reward = self.rewards[self.state, action]
        self.state = next_state
        return self.to_onehot(next_state), reward, done, {}

class TransferDebugCommonDescedentOneHotAC(TransferDebugCommonDescedent):
    def __init__(self):
        TransferDebugCommonDescedent.__init__(self)
        self.big_reward = 0.8
        self.eplen = 10

        self.states = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # 10 is an absorbing state
        self.actions = [0, 1, 2, 3, 4, 5]
        self.transitions[(0, 0)]= 6
        self.transitions[(6, 2)]= 8

        self.rewards[(6, 2)]= self.big_reward

        self.starting_states = [0]
        self.terminal_states = [8, 9, 10]

class TransferDebugCommonDescedentOneHotBC(TransferDebugCommonDescedent):
    def __init__(self):
        TransferDebugCommonDescedent.__init__(self)
        self.big_reward = 0.8
        self.eplen = 10

        self.states = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # 10 is an absorbing state
        self.actions = [0, 1, 2, 3, 4, 5]
        self.transitions[(1, 1)]= 6
        self.transitions[(6, 2)]= 8

        self.rewards[(6, 2)]= self.big_reward

        self.starting_states = [1]
        self.terminal_states = [8, 9, 10]

class TransferDebugCommonDescedentOneHotAF(TransferDebugCommonDescedent):
    def __init__(self):
        TransferDebugCommonDescedent.__init__(self)
        self.big_reward = 0.8
        self.eplen = 10

        self.states = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # 10 is an absorbing state
        self.actions = [0, 1, 2, 3, 4, 5]
        self.transitions[(0, 0)]= 7
        self.transitions[(7, 5)]= 9

        self.rewards[(7, 5)]= self.big_reward

        self.starting_states = [0]
        self.terminal_states = [8, 9, 10]

class TransferDebugCommonDescedentOneHotBF(TransferDebugCommonDescedent):
    def __init__(self):
        TransferDebugCommonDescedent.__init__(self)
        self.big_reward = 0.8
        self.eplen = 10

        self.states = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # 10 is an absorbing state
        self.actions = [0, 1, 2, 3, 4, 5]
        self.transitions[(1, 1)]= 7
        self.transitions[(7, 5)]= 9

        self.rewards[(7, 5)]= self.big_reward

        self.starting_states = [1]
        self.terminal_states = [8, 9, 10]

class TransferDebugCommonDescedentOneHotDC(TransferDebugCommonDescedent):
    def __init__(self):
        TransferDebugCommonDescedent.__init__(self)
        self.big_reward = 0.8
        self.eplen = 10

        self.states = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # 10 is an absorbing state
        self.actions = [0, 1, 2, 3, 4, 5]
        self.transitions[(3, 3)]= 6
        self.transitions[(6, 2)]= 8

        self.rewards[(6, 2)]= self.big_reward

        self.starting_states = [3]
        self.terminal_states = [8, 9, 10]

class TransferDebugCommonDescedentOneHotEC(TransferDebugCommonDescedent):
    def __init__(self):
        TransferDebugCommonDescedent.__init__(self)
        self.big_reward = 0.8
        self.eplen = 10

        self.states = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # 10 is an absorbing state
        self.actions = [0, 1, 2, 3, 4, 5]
        self.transitions[(4, 4)]= 6
        self.transitions[(6, 2)]= 8

        self.rewards[(6, 2)]= self.big_reward

        self.starting_states = [4]
        self.terminal_states = [8, 9, 10]


class TransferDebugTask1(TransferDebug):
    def __init__(self):
        TransferDebug.__init__(self)
        self.rewards = {
            # absorbing state gets no reward
            (5, 0): 0,
            (5, 1): 0,

            (0, 0): 0,
            (0, 1): 0,

            (1, 0): 0,
            (1, 1): self.big_reward,

            (2, 0): 0,
            (2, 1): 0,

            (4, 0): 0,
            (4, 1): 0,
        }
        self.starting_states = [0]
        self.terminal_states = [2, 4, 5]



class TransferDebugTask2(TransferDebug):
    def __init__(self):
        TransferDebug.__init__(self)
        self.rewards = {
            # absorbing state gets no reward
            (5, 0): 0,
            (5, 1): 0,

            (3, 0): 0,
            (3, 1): 0,

            (1, 0): self.big_reward,
            (1, 1): 0,

            (2, 0): 0,
            (2, 1): 0,

            (4, 0): 0,
            (4, 1): 0,
        }
        self.starting_states = [3]
        self.terminal_states = [2, 4, 5]

class TransferDebugTask3(TransferDebug):
    def __init__(self):
        TransferDebug.__init__(self)
        self.rewards = {
            # absorbing state gets no reward
            (5, 0): 0,
            (5, 1): 0,

            (0, 0): 0,
            (0, 1): 0,

            (3, 0): 0,
            (3, 1): 0,

            (1, 0): self.big_reward,
            (1, 1): 0,

            (2, 0): 0,
            (2, 1): 0,

            (4, 0): 0,
            (4, 1): 0,
        }
        self.starting_states = [0]
        self.terminal_states = [2, 4, 5]


class LongerTransferDebugTask(MultiStepEnv):
    def __init__(self):
        MultiStepEnv.__init__(self)
        self.big_reward = 0.8
        self.eplen = 10

        self.states = [0, 1, 2, 3]
        self.actions = [0, 1, 2, 3]

        self.rewards = {
            # absorbing state gets no reward
            (3, 0): 0,
            (3, 1): 0,
            (3, 2): 0,
            (3, 3): 0,

            # state 0
            (0, 0): 0,
            (0, 1): 0,
            (0, 2): 0,
            (0, 3): 0,

            # state 1
            (1, 0): 0,
            (1, 1): 0,
            (1, 2): 0,
            (1, 3): 0,

            # state 2
            (2, 0): 0,
            (2, 1): 0,
            (2, 2): self.big_reward,
            (2, 3): 0,
        }
        self.Qs = {
            # absorbing state gets no reward
            (3, 0): 0,
            (3, 1): 0,
            (3, 2): 0,
            (3, 3): 0,

            # state 0
            (0, 0): 0,
            (0, 1): 0,
            (0, 2): 0,
            (0, 3): 0,

            # state 1
            (1, 0): 0,
            (1, 1): 0,
            (1, 2): 0,
            (1, 3): 0,

            # state 2
            (2, 0): 0,
            (2, 1): 0,
            (2, 2): 0,
            (2, 3): 0,
        }
        self.starting_states = [0]
        self.terminal_states = [3]

    def step(self, action):
        next_state = self.transitions[self.state, action]
        self.counter += 1
        done = next_state in self.terminal_states
        reward = self.rewards[self.state, action]
        self.state = next_state
        return self.to_onehot(next_state), reward, done, {}

class LongerTransferDebugTask1(LongerTransferDebugTask):
    def __init__(self):
        LongerTransferDebugTask.__init__(self)
        self.transitions = {
            # absorbing state
            (3, 0): 3,
            (3, 1): 3,
            (3, 2): 3,
            (3, 3): 3,

            # state 0
            (0, 0): 1,
            (0, 1): 0,  # self-transtion
            (0, 2): 0,  # self-transtion
            (0, 3): 0,  # self-transtion

            # state 1
            (1, 0): 1,  # self-transition
            (1, 1): 2,  # changed
            (1, 2): 1,  # self-transition
            (1, 3): 1,  # self-transition

            # state 2
            (2, 0): 2,  # self-transition
            (2, 1): 2,  # self-transition
            (2, 2): 3,
            (2, 3): 2,  # self-transition
        }

class LongerTransferDebugTask2(LongerTransferDebugTask):
    def __init__(self):
        LongerTransferDebugTask.__init__(self)
        self.transitions = {
            # absorbing state
            (3, 0): 3,
            (3, 1): 3,
            (3, 2): 3,
            (3, 3): 3,

            # state 0
            (0, 0): 1,
            (0, 1): 0,  # self-transtion
            (0, 2): 0,  # self-transtion
            (0, 3): 0,  # self-transtion

            # state 1
            (1, 0): 1,  # self-transition
            (1, 1): 1,  # self-transition
            (1, 2): 1,  # self-transition
            (1, 3): 2,  # changed

            # state 2
            (2, 0): 2,  # self-transition
            (2, 1): 2,  # self-transition
            (2, 2): 3,
            (2, 3): 2,  # self-transition
        }

class ABCEDPretrain(MultiStepEnv):
    def __init__(self):
        MultiStepEnv.__init__(self)
        self.big_reward = 0.8
        self.eplen = 10

        self.states = [0, 1, 2, 3, 4]  # 10 is an absorbing state
        self.actions = [0, 1, 2, 3, 4] # A-0, B-1, C-2, D-3, E-4
        self.transitions = {
            # absorbing state
            (5, 0): 5,
            (5, 1): 5,
            (5, 2): 5,
            (5, 3): 5,
            (5, 4): 5,

            (0, 0): 1,
            (0, 1): 0,
            (0, 2): 0,
            (0, 3): 0,
            (0, 4): 0,

            (1, 0): 1,
            (1, 1): 2,
            (1, 2): 1,
            (1, 3): 1,
            (1, 4): 1,

            (2, 0): 2,
            (2, 1): 2,
            (2, 2): 3,
            (2, 3): 2,
            (2, 4): 3,

            (3, 0): 3,
            (3, 1): 3,
            (3, 2): 3,
            (3, 3): 4,
            (3, 4): 3,

            (4, 0): 5,
            (4, 1): 5,
            (4, 2): 5,
            (4, 3): 5,
            (4, 4): 5,
        }
        # Qs are actually not correct
        self.Qs = {
            (5, 0): 0,
            (5, 1): 0,
            (5, 2): 0,
            (5, 3): 0,
            (5, 4): 0,

            (0, 0): 0,
            (0, 1): 0,
            (0, 2): 0,
            (0, 3): 0,
            (0, 4): 0,

            (1, 0): 0,
            (1, 1): 0,
            (1, 2): 0,
            (1, 3): 0,
            (1, 4): 0,

            (2, 0): 0,
            (2, 1): 0,
            (2, 2): 0,
            (2, 3): 0,
            (2, 4): 0,

            (3, 0): 0,
            (3, 1): 0,
            (3, 2): 0,
            (3, 3): 0,
            (3, 4): 0,

            (4, 0): 0,
            (4, 1): 0,
            (4, 2): 0,
            (4, 3): 0,
            (4, 4): 0,
        }
        self.rewards = {
            # absorbing state gets no reward
            (5, 0): 0,
            (5, 1): 0,
            (5, 2): 0,
            (5, 3): 0,
            (5, 4): 0,

            (0, 0): 0,
            (0, 1): 0,
            (0, 2): 0,
            (0, 3): 0,
            (0, 4): 0,

            (1, 0): 0,
            (1, 1): 0,
            (1, 2): 0,
            (1, 3): 0,
            (1, 4): 0,

            (2, 0): 0,
            (2, 1): 0,
            (2, 2): 0,
            (2, 3): 0,
            (2, 4): 0,

            (3, 0): 0,
            (3, 1): 0,
            (3, 2): 0,
            (3, 3): self.big_reward,
            (3, 4): 0,

            (4, 0): 0,
            (4, 1): 0,
            (4, 2): 0,
            (4, 3): 0,
            (4, 4): 0,
        }
        self.starting_states = [0]
        self.terminal_states = [4,5]

    def step(self, action):
        next_state = self.transitions[self.state, action]
        self.counter += 1
        done = next_state in self.terminal_states
        reward = self.rewards[self.state, action]
        self.state = next_state
        return self.to_onehot(next_state), reward, done, {}

class ABEDTransfer(ABCEDPretrain):
    def __init__(self):
        ABCEDPretrain.__init__(self)

        self.transitions[(2, 2)]= 2

        self.starting_states = [0]
        self.terminal_states = [4, 5]

class CondensedLinearABC(MultiStepEnv):
    def __init__(self):
        MultiStepEnv.__init__(self)
        self.big_reward = 0.8
        self.eplen = 10

        self.states = [0, 1, 2, 3]  # 10 is an absorbing state
        self.actions = [0, 1] # A-0, B-1, C-2, D-3, E-4
        self.transitions = {
            # absorbing state
            (4, 0): 4,
            (4, 1): 4,

            (0, 0): 1,
            (0, 1): 0,

            (1, 0): 2,
            (1, 1): 1,  

            (2, 0): 3,
            (2, 1): 2,

            (3, 0): 4,
            (3, 1): 4,



        }
        # Qs are actually not correct
        self.Qs = {
            (4, 0): 0,
            (4, 1): 0,

            (0, 0): 0,
            (0, 1): 0,

            (1, 0): 0,
            (1, 1): 0,

            (2, 0): 0,
            (2, 1): 0,
 
            (3, 0): 0,
            (3, 1): 0,


        }
        self.rewards = {
            # absorbing state gets no reward
            (4, 0): 0,
            (4, 1): 0,

            (0, 0): 0,
            (0, 1): 0,

            (1, 0): 0,
            (1, 1): 0,

            (2, 0): self.big_reward,
            (2, 1): 0,
 
            (3, 0): 0,
            (3, 1): 0,
        }
        self.starting_states = [0]
        self.terminal_states = [3,4]

    def step(self, action):
        next_state = self.transitions[self.state, action]
        self.counter += 1
        done = next_state in self.terminal_states
        reward = self.rewards[self.state, action]
        self.state = next_state
        return self.to_onehot(next_state), reward, done, {}

class CondensedLinABD(CondensedLinearABC):
    def __init__(self):
        CondensedLinearABC.__init__(self)

        self.transitions = {
            # absorbing state
            (4, 0): 4,
            (4, 1): 4,

            (0, 0): 1,
            (0, 1): 0,

            (1, 0): 2,
            (1, 1): 1,  

            (2, 0): 2,
            (2, 1): 3,

            (3, 0): 4,
            (3, 1): 4,
            }
        self.rewards[(2, 1)]= self.big_reward
        self.rewards[(2,0)]=0
        self.starting_states = [0]
        self.terminal_states = [4, 5]

class LinearOneHotABC(MultiStepEnv):
    def __init__(self):
        MultiStepEnv.__init__(self)
        self.big_reward = 0.8
        self.eplen = 10

        self.states = [0, 1, 2, 3]  # 10 is an absorbing state
        self.actions = [0, 1, 2, 3, 4, 5] # A-0, B-1, C-2, D-3, E-4, F-5
        self.transitions = {
            # absorbing state
            (4, 0): 4,
            (4, 1): 4,
            (4, 2): 4,
            (4, 3): 4,
            (4, 4): 4,
            (4, 5): 4,

            (0, 0): 1,
            (0, 1): 0,
            (0, 2): 0,
            (0, 3): 0,
            (0, 4): 0,
            (0, 5): 0,

            (1, 0): 1,
            (1, 1): 2,  
            (1, 2): 1,
            (1, 3): 1, 
            (1, 4): 1,
            (1, 5): 1, 

            (2, 0): 2,
            (2, 1): 2,
            (2, 2): 3,
            (2, 3): 2,
            (2, 4): 2,
            (2, 5): 2,

            (3, 0): 4,
            (3, 1): 4,
            (3, 2): 4,
            (3, 3): 4,
            (3, 4): 4,
            (3, 5): 4,

        }
        # Qs are actually not correct
        self.Qs = {
            (4, 0): 0,
            (4, 1): 0,
            (4, 2): 0,
            (4, 3): 0,
            (4, 4): 0,
            (4, 5): 0,

            (0, 0): 0,
            (0, 1): 0,
            (0, 2): 0,
            (0, 3): 0,
            (0, 4): 0,
            (0, 5): 0,

            (1, 0): 0,
            (1, 1): 0,  
            (1, 2): 0,
            (1, 3): 0, 
            (1, 4): 0,
            (1, 5): 0, 

            (2, 0): 0,
            (2, 1): 0,
            (2, 2): 0,
            (2, 3): 0,
            (2, 4): 0,
            (2, 5): 0,

            (3, 0): 0,
            (3, 1): 0,
            (3, 2): 0,
            (3, 3): 0,
            (3, 4): 0,
            (3, 5): 0,


        }
        self.rewards = {
            # absorbing state gets no reward
            (4, 0): 0,
            (4, 1): 0,
            (4, 2): 0,
            (4, 3): 0,
            (4, 4): 0,
            (4, 5): 0,

            (0, 0): 0,
            (0, 1): 0,
            (0, 2): 0,
            (0, 3): 0,
            (0, 4): 0,
            (0, 5): 0,

            (1, 0): 0,
            (1, 1): 0,  
            (1, 2): 0,
            (1, 3): 0, 
            (1, 4): 0,
            (1, 5): 0, 

            (2, 0): 0,
            (2, 1): 0,
            (2, 2): self.big_reward,
            (2, 3): 0,
            (2, 4): 0,
            (2, 5): 0,

            (3, 0): 0,
            (3, 1): 0,
            (3, 2): 0,
            (3, 3): 0,
            (3, 4): 0,
            (3, 5): 0,
        }
        self.starting_states = [0]
        self.terminal_states = [3,4]

    def step(self, action):
        next_state = self.transitions[self.state, action]
        self.counter += 1
        done = next_state in self.terminal_states
        reward = self.rewards[self.state, action]
        self.state = next_state
        return self.to_onehot(next_state), reward, done, {}

class LinearOneHotABD(LinearOneHotABC):
    def __init__(self):
        LinearOneHotABC.__init__(self)

        self.transitions = {
            # absorbing state
            (4, 0): 4,
            (4, 1): 4,
            (4, 2): 4,
            (4, 3): 4,
            (4, 4): 4,
            (4, 5): 4,

            (0, 0): 1,
            (0, 1): 0,
            (0, 2): 0,
            (0, 3): 0,
            (0, 4): 0,
            (0, 5): 0,

            (1, 0): 1,
            (1, 1): 2,  
            (1, 2): 1,
            (1, 3): 1, 
            (1, 4): 1,
            (1, 5): 1, 

            (2, 0): 2,
            (2, 1): 2,
            (2, 2): 2,
            (2, 3): 3,
            (2, 4): 2,
            (2, 5): 2,

            (3, 0): 4,
            (3, 1): 4,
            (3, 2): 4,
            (3, 3): 4,
            (3, 4): 4,
            (3, 5): 4,
            }
        self.rewards[(2, 3)]= self.big_reward
        self.rewards[(2,2)]=0
        self.starting_states = [0]
        self.terminal_states = [3, 4]

class LinearOneHotAEC(LinearOneHotABC):
    def __init__(self):
        LinearOneHotABC.__init__(self)

        self.transitions = {
            # absorbing state
            (4, 0): 4,
            (4, 1): 4,
            (4, 2): 4,
            (4, 3): 4,
            (4, 4): 4,
            (4, 5): 4,

            (0, 0): 1,
            (0, 1): 0,
            (0, 2): 0,
            (0, 3): 0,
            (0, 4): 0,
            (0, 5): 0,

            (1, 0): 1,
            (1, 1): 1,  
            (1, 2): 1,
            (1, 3): 1, 
            (1, 4): 2,
            (1, 5): 1, 

            (2, 0): 2,
            (2, 1): 2,
            (2, 2): 3,
            (2, 3): 2,
            (2, 4): 2,
            (2, 5): 2,

            (3, 0): 4,
            (3, 1): 4,
            (3, 2): 4,
            (3, 3): 4,
            (3, 4): 4,
            (3, 5): 4,
            }
        # self.rewards[(2, 3)]= self.big_reward
        # self.rewards[(2,2)]=0
        self.starting_states = [0]
        self.terminal_states = [3, 4]

class LinearOneHotFBC(LinearOneHotABC):
    def __init__(self):
        LinearOneHotABC.__init__(self)

        self.transitions = {
            # absorbing state
            (4, 0): 4,
            (4, 1): 4,
            (4, 2): 4,
            (4, 3): 4,
            (4, 4): 4,
            (4, 5): 4,

            (0, 0): 0,
            (0, 1): 0,
            (0, 2): 0,
            (0, 3): 0,
            (0, 4): 0,
            (0, 5): 1,

            (1, 0): 1,
            (1, 1): 2,  
            (1, 2): 1,
            (1, 3): 1, 
            (1, 4): 1,
            (1, 5): 1, 

            (2, 0): 2,
            (2, 1): 2,
            (2, 2): 3,
            (2, 3): 2,
            (2, 4): 2,
            (2, 5): 2,

            (3, 0): 4,
            (3, 1): 4,
            (3, 2): 4,
            (3, 3): 4,
            (3, 4): 4,
            (3, 5): 4,
            }
        # self.rewards[(2, 3)]= self.big_reward
        # self.rewards[(2,2)]=0
        self.starting_states = [0]
        self.terminal_states = [3, 4]



class CondensedLinAEC(CondensedLinearABC):
    def __init__(self):
        CondensedLinearABC.__init__(self)

        self.transitions = {
            # absorbing state
            (4, 0): 4,
            (4, 1): 4,

            (0, 0): 1,
            (0, 1): 0,

            (1, 0): 1,
            (1, 1): 2,  

            (2, 0): 3,
            (2, 1): 2,

            (3, 0): 4,
            (3, 1): 4,
            }
        # self.rewards[(2, 1)]= self.big_reward
        # self.rewards[(2,0)]=0
        self.starting_states = [0]
        self.terminal_states = [4, 5]

class CondensedLinFBC(CondensedLinearABC):
    def __init__(self):
        CondensedLinearABC.__init__(self)

        self.transitions = {
            # absorbing state
            (4, 0): 4,
            (4, 1): 4,

            (0, 0): 0,
            (0, 1): 1,

            (1, 0): 2,
            (1, 1): 1,  

            (2, 0): 3,
            (2, 1): 2,

            (3, 0): 4,
            (3, 1): 4,
            }
        # self.rewards[(2, 1)]= self.big_reward
        # self.rewards[(2,0)]=0
        self.starting_states = [0]
        self.terminal_states = [4, 5]

class CondensedLinearABCV2(MultiTaskDebug):
    def __init__(self):
        MultiTaskDebug.__init__(self)
        self.states = [0, 1, 2, 3]  # 8 is an absorbing state
        self.keys=[0,1,2,3,4,5]
        self.actions = [0, 1] # 0->A , 1->B, 2->C, 3->D, 4->E, 5->F
        self.transitions = {
            # absorbing state
            ((4,-1), 0): (4,-1),
            ((4,-1), 1): (4,-1),


            ((0,0), 0): (1,2),
            ((0,0), 1): (0,0),

            ((0,1), 0): (0,1),
            ((0,1), 1): (0,1),

            ((1,2), 0): (2,4),
            ((1,2), 1): (1,2),

            ((1,3), 0): (1,3),
            ((1,3), 1): (1,3),

            ((2,4), 0): (3,-1),
            ((2,4), 1): (2,4),

            ((2,5), 0): (2,5),
            ((2,5), 1): (2,5),

            ((3,-1), 0): (4,-1),
            ((3,-1), 1): (4,-1),

        }
        # Qs are actually not correct
        self.Qs = {
            ((4,-1), 0): 0,
            ((4,-1), 1): 0,


            ((0,0), 0): 0,
            ((0,0), 1): 0,

            ((0,1), 0): 0,
            ((0,1), 1): 0,

            ((1,2), 0): 0,
            ((1,2), 1): 0,

            ((1,3), 0): 0,
            ((1,3), 1): 0,

            ((2,4), 0): 0,
            ((2,4), 1): 0,

            ((2,5), 0): 0,
            ((2,5), 1): 0,

            ((3,-1), 0): 0,
            ((3,-1), 1): 0,

        }
        self.rewards = {
    # absorbing state gets no reward
            ((4,-1), 0): 0,
            ((4,-1), 1): 0,


            ((0,0), 0): 0,
            ((0,0), 1): 0,

            ((0,1), 0): 0,
            ((0,1), 1): 0,

            ((1,2), 0): 0,
            ((1,2), 1): 0,

            ((1,3), 0): 0,
            ((1,3), 1): 0,

            ((2,4), 0): self.big_reward,
            ((2,4), 1): 0,

            ((2,5), 0): 0,
            ((2,5), 1): 0,

            ((3,-1), 0): 0,
            ((3,-1), 1): 0,

        }
        self.starting_states = [(0,0)]
        self.terminal_states = [(3,-1),(4,-1)]


def test_MultiTaskDebugTaskABC():
    action_sequences = {

        (1, 0, 0): dict(states=[(0,0), (0,0), (1,2), (1,2)], rewards=[0,0,0], dones=[False, False, False]),
        (1, 0, 1): dict(states=[(0,0), (0,0), (1,2), (2,-1)], rewards=[0,0,0], dones=[False, False, False]),

        (1, 1, 0): dict(states=[(0,0), (0,0), (0,0), (1,2)], rewards=[0,0,0], dones=[False, False, False]),
        (1, 1, 1): dict(states=[(0,0), (0,0), (0,0), (0,0)], rewards=[0,0,0], dones=[False, False, False]),

        (0, 1, 0): dict(states=[(0,0), (1,2), (2,-1), (2,-1)], rewards=[0, 0,0], dones=[False, False, False]),
        (0, 1, 2): dict(states=[(0,0), (1,2), (2,-1), (3,4)], rewards=[0, 0,0.8], dones=[False, False, True]),

    }
    tabular_tester(MultiTaskDebugTaskABC, action_sequences)

def test_MultiTaskDebugTaskABCSIX():
    action_sequences = {

        (1, 0, 0): dict(states=[(0,0), (0,0), (1,2), (1,2)], rewards=[0,0,0], dones=[False, False, False]),
        (1, 0, 1): dict(states=[(0,0), (0,0), (1,2), (2,-1)], rewards=[0,0,0], dones=[False, False, False]),

        (1, 1, 0): dict(states=[(0,0), (0,0), (0,0), (1,2)], rewards=[0,0,0], dones=[False, False, False]),
        (1, 4, 5): dict(states=[(0,0), (0,0), (0,0), (0,0)], rewards=[0,0,0], dones=[False, False, False]),

        (0, 1, 0): dict(states=[(0,0), (1,2), (2,-1), (2,-1)], rewards=[0, 0,0], dones=[False, False, False]),
        (0, 1, 2): dict(states=[(0,0), (1,2), (2,-1), (3,4)], rewards=[0, 0,0.8], dones=[False, False, True]),

    }
    tabular_tester(MultiTaskDebugTaskABCSIX, action_sequences)


def test_TransferDebugTask1():
    action_sequences = {
        (1, 0): dict(states=[0, 1, 4], rewards=[0,0], dones=[False, True]),
        (1, 1): dict(states=[0, 1, 2], rewards=[0,0.8], dones=[False, True]),

        (1, 0, 0): dict(states=[0, 1, 4, 5], rewards=[0,0,0], dones=[False, True, True]),
        (1, 0, 1): dict(states=[0, 1, 4, 5], rewards=[0,0,0], dones=[False, True, True]),

        (1, 1, 0): dict(states=[0, 1, 2, 5], rewards=[0,0.8,0], dones=[False, True, True]),
        (1, 1, 1): dict(states=[0, 1, 2, 5], rewards=[0,0.8,0], dones=[False, True, True]),

        (0, 1, 0): dict(states=[0, 0, 1, 4], rewards=[0, 0,0], dones=[False, False, True]),
        (0, 1, 1): dict(states=[0, 0, 1, 2], rewards=[0, 0,0.8], dones=[False, False, True]),

        (0, 1, 0, 0): dict(states=[0, 0, 1, 4, 5], rewards=[0,0,0,0], dones=[False, False, True, True]),
        (0, 1, 0, 1): dict(states=[0, 0, 1, 4, 5], rewards=[0,0,0,0], dones=[False, False, True, True]),

        (0, 1, 1, 0): dict(states=[0, 0, 1, 2, 5], rewards=[0,0,0.8,0], dones=[False, False, True, True]),
        (0, 1, 1, 1): dict(states=[0, 0, 1, 2, 5], rewards=[0,0,0.8,0], dones=[False, False, True, True]),
    }
    tabular_tester(TransferDebugTask1, action_sequences)


def test_TransferDebugTask2():
    action_sequences = {
        (0, 0): dict(states=[3, 1, 4], rewards=[0,0.8], dones=[False, True]),
        (0, 1): dict(states=[3, 1, 2], rewards=[0,0], dones=[False, True]),

        (1, 0, 0): dict(states=[3, 3, 1, 4], rewards=[0,0,0.8], dones=[False, False, True]),
        (1, 0, 1): dict(states=[3, 3, 1, 2], rewards=[0,0,0], dones=[False, False, True]),

        (1, 1, 0): dict(states=[3, 3, 3, 1], rewards=[0,0,0], dones=[False, False, False]),
        (1, 1, 1): dict(states=[3, 3, 3, 3], rewards=[0,0,0], dones=[False, False, False]),

        (0, 1, 0): dict(states=[3, 1, 2, 5], rewards=[0,0,0], dones=[False, True, True]),
        (0, 1, 1): dict(states=[3, 1, 2, 5], rewards=[0,0,0], dones=[False, True, True]),

        (0, 0, 0): dict(states=[3, 1, 4, 5], rewards=[0,0.8,0], dones=[False, True, True]),
        (0, 0, 1): dict(states=[3, 1, 4, 5], rewards=[0,0.8,0], dones=[False, True, True]),

        (0, 0, 0): dict(states=[3, 1, 4, 5], rewards=[0,0.8,0], dones=[False, True, True]),
        (0, 1, 0): dict(states=[3, 1, 2, 5], rewards=[0,0,0], dones=[False, True, True]),
    }
    tabular_tester(TransferDebugTask2, action_sequences)


if __name__ == '__main__':
    
    # test_CounterExample1Env()
    # test_TransferDebugTask1()
    # test_TransferDebugTask2()
    # test_MultiTaskDebugTaskABC()
    test_MultiTaskDebugTaskABCSIX()