import numpy as np
from starter_code.infrastructure.log import create_logdir

class TaskNode(object):
    def __init__(self, task_name, parents=None):
        self.task_name
        self.parents = parents  # a list of TaskNodes


class TaskDistribution(object):
    """
        Contains a set of K environment managers
    """
    def __init__(self):
        self.environment_managers = []

    def __iter__(self):
        for env_manager in self.environment_managers:
            yield env_manager

    def __len__(self):
        return len(self.environment_managers)

    def __str__(self):
        return '{}\n'.format(type(self)) + '\n'.join('\t{}: {}'.format(e.env_name, e) for e in self.environment_managers)

    def get_canonical(self):
        return self.environment_managers[0]

    def append(self, environment_manager):
        self.environment_managers.append(environment_manager)
        assert self.consistent_spec()

    @property
    def state_dim(self):
        return self.get_canonical().state_dim

    @property
    def action_dim(self):
        return self.get_canonical().action_dim

    @property
    def is_disc_action(self):
        return self.get_canonical().is_disc_action

    def consistent_spec(self):
        same_state_dim = [e.state_dim == self.environment_managers[0].state_dim for e in self.environment_managers]
        same_action_dim = [e.action_dim == self.environment_managers[0].action_dim for e in self.environment_managers]
        return same_state_dim and same_action_dim

    def sample(self):
        return np.random.choice(self.environment_managers)


class TaskDistributionGroup(object):
    """
        Contains a group of train/val/test task distributions
    """
    def __init__(self):
        self.dists = {}

    def __setitem__(self, mode, dist):
        self.dists[mode] = dist
        assert self.consistent_spec()

    def __getitem__(self, mode):
        return self.dists[mode]

    def __str__(self):
        return '{}\n'.format(type(self)) + '\n'.join(['\t{}: {}'.format(k, str(v)) for k, v in self.dists.items()])

    def get_canonical(self):
        return list(self.dists.values())[0]

    @property
    def state_dim(self):
        return self.get_canonical().state_dim

    @property
    def action_dim(self):
        return self.get_canonical().action_dim

    @property
    def is_disc_action(self):
        return self.get_canonical().is_disc_action

    def consistent_spec(self):
        return all([v.consistent_spec() for v in self.dists.values()])

    def sample(self, mode):
        return self.dists[mode].sample()


class TaskProgression(object):
    """
        Contains a time series of TaskDistributionGroups
    """
    def __init__(self):
        self.task_dist_group_series = []

    def __len__(self):
        return len(self.task_dist_group_series)

    def __getitem__(self, i):
        idx = min(i, len(self.task_dist_group_series)-1)
        return self.task_dist_group_series[idx]

    def __str__(self):
        return '{}\n'.format(type(self)) + '\n'.join('\t'.format(str(g)) for g in self.task_dist_group_series)

    def get_canonical(self):
        return self.task_dist_group_series[0]

    @property
    def state_dim(self):
        return self.get_canonical().state_dim

    @property
    def action_dim(self):
        return self.get_canonical().action_dim

    @property
    def is_disc_action(self):
        return self.get_canonical().is_disc_action

    def append(self, task_dist_group):
        self.task_dist_group_series.append(task_dist_group)
        assert self.consistent_spec()

    def consistent_spec(self):
        return all([v.consistent_spec() for v in self.task_dist_group_series])

    def sample(self, i, mode):
        prog_group_id = min(i, len(self.task_dist_group_series)-1)
        return self.task_dist_group_series[prog_group_id].sample(mode)


def default_task_prog_spec(env_name):
    spec = {0: dict(train=[env_name], test=[env_name])}
    return spec


def task_prog_spec_multi(env_names):
    spec = {0: dict(train=env_names, test=env_names)}
    return spec


def construct_task_progression(task_prog_spec, env_manager_builder, logger, env_registry, args):
    """
    should be able to convert:
    {
        Multitask Training
        0: {
            train: [env0, env1, env2, env3],
            test: [env0, env1, env2, env3],
        },

        Continual Learning
        0: {
            train: [env0],
            test: [env0],
        },
        1: {
            train: [env1],
            test: [env1],
        },
        2: {
            train: [env2],
            test: [env2],
        },
        3: {
            train: [env3],
            test: [env3],
        },

        Mixed
        0: {
            train: [env0, env1, env2],
            test: [env0, env1, env2],
        },
        1: {
            train: [env3, env4, env5],
            test: [env3, env4, env5],
        },
        2: {
            train: [env6, env7, env8],
            test: [env6, env7, env8],
        },
        3: {
            train: [env9, env10, env11],
            test: [env9, env10, env11],
        },

    }
    to a TaskProgression object

    The directory structure is
        exp_root
            0
                train_folder_0_0
                train_folder_0_1
                test_folder_0_0
                test_folder_0_1
            1
                train_folder_1_0
                train_folder_1_1
                test_folder_1_0
                test_folder_1_1
    """
    task_progression = TaskProgression()
    for i, group in task_prog_spec.items():
        group_dir = create_logdir(
            root=logger.logdir, dirname='group_{}'.format(i), setdate=False)
        task_distribution_group = TaskDistributionGroup()
        for mode, envs in group.items():
            task_distribution = TaskDistribution()
            for env_name in envs:
                env_manager = env_manager_builder(env_name, env_registry, args)
                env_manager.set_logdir(create_logdir(
                    root=group_dir,
                    dirname='{}_{}_{}'.format(env_name, i, mode),
                    setdate=False))
                task_distribution.append(env_manager)
            task_distribution_group[mode] = task_distribution
        task_progression.append(task_distribution_group)
    return task_progression