from collections import OrderedDict
import pickle
import sys
import time
from starter_code.infrastructure.log import log_string, format_log_string

class Experiment():
    def __init__(self, learner, task_progression, logger, args):
        self.learner = learner
        self.task_progression = task_progression
        self.logger = logger
        self.args = args

    def main_loop(self, max_epochs):
        for epoch in range(max_epochs):
            if epoch % self.args.eval_every == 0:
                self.eval_step(epoch)
            self.train_step(epoch)
            
        self.finish_training()

    def train_step(self, epoch):
        assert len(self.task_progression) == 1
        train_env_manager = self.task_progression.sample(i=epoch, mode='train')
        epoch_stats = self.learner.collect_samples(
            epoch=epoch, 
            max_episode_length=train_env_manager.max_episode_length, 
            env=train_env_manager.env)
        if epoch % self.args.log_every == 0:
            self.logger.printf(format_log_string(self.log(epoch, epoch_stats, mode='train')))
        if epoch % self.args.visualize_every == 0:
            self.visualize(train_env_manager, epoch, epoch_stats, self.logger.expname)
        if epoch % self.args.save_every == 0:
            self.save(train_env_manager, epoch, epoch_stats)
        self.learner.update(epoch)


    def eval_step(self, epoch):
        self.logger.printf('Evaluating...')
        self.learner.organism.to('cpu')
        for env_manager in self.task_progression[epoch]['test']:
            t0 = time.time()
            stats = self.learner.test(epoch, env_manager, self.args.num_test)
            self.logger.printf(format_log_string(self.log(epoch, stats, mode='eval')))
            self.logger.printf('Time to sample test examples: {}'.format(time.time()-t0))

            self.visualize(env_manager, epoch, stats, env_manager.env_name, eval_mode=True)
            self.save(env_manager, epoch, stats)
        self.learner.organism.to(self.learner.device)

    def finish_training(self):
        if self.args.debug:
            self.logger.remove_exproot()

    def log(self, epoch, epoch_stats, mode):
        s = log_string(OrderedDict({
            '{} epoch'.format(mode): epoch,
            'env steps this batch': epoch_stats['total_steps'],
            'env steps taken': self.learner.steps,
            'avg return': epoch_stats['mean_return'],
            'std return': epoch_stats['std_return'],
            'min return': epoch_stats['min_return'],
            'max return': epoch_stats['max_return'],
            'min return ever': self.learner.min_return,
            'max return ever': self.learner.max_return
            }))
        return [s]

    def update_metrics(self, env_manager, epoch, stats):
        env_manager.update_variable(name='epoch', index=epoch, value=epoch)
        env_manager.update_variable(name='steps', index=epoch, value=self.learner.steps)
        for metric in env_manager.logged_metrics:
            env_manager.update_variable(
                name=metric, index=epoch, value=stats[metric], include_running_avg=True)

    def plot_metrics(self, env_manager, name):
        env_manager.plot(
            var_pairs=[(('steps', k)) for k in env_manager.logged_metrics],
            expname=name,
            pfunc=self.logger.printf)

    def save(self, env_manager, epoch, stats):
        env_manager.saver.save(epoch,
            {'args': self.args,
             'epoch': epoch,
             'logger': self.logger.get_state_dict(),
             'mean_return': stats['mean_return'],
             'organism': self.learner.organism.get_state_dict()},
             self.logger.printf)

    def visualize(self, env_manager, epoch, stats, name, eval_mode=False):
        self.update_metrics(env_manager, epoch, stats)
        self.plot_metrics(env_manager, name)