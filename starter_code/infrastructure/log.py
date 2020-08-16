from collections import defaultdict
import csv
import cv2
import datetime
import imageio
import glob
import h5py
import heapq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import mplcyberpunk
import numpy as np
import operator
import os
import pandas as pd
import pprint
import shutil
import torch
import ujson

from starter_code.environment.env_config import EnvRegistry
from starter_code.infrastructure.utils import is_float

plt.style.use("cyberpunk")
mplcyberpunk.add_glow_effects()


def mkdirp(logdir):
    try:
        os.makedirs(logdir)
    except FileExistsError:
        overwrite = 'o'
        while overwrite not in ['y', 'n']:
            overwrite = input('{} exists. Overwrite? [y/n] '.format(logdir))
        if overwrite == 'y':
            shutil.rmtree(logdir)
            os.mkdir(logdir)
        else:
            raise FileExistsError


def create_logdir(root, dirname, setdate):
    logdir = os.path.join(root, dirname)
    if setdate:
        if not dirname == '': logdir += '__'
        logdir += '{date:%Y-%m-%d_%H-%M-%S}'.format(
        date=datetime.datetime.now())
    mkdirp(logdir)
    return logdir


class RunningAverage(object):
    def __init__(self):
        super(RunningAverage, self).__init__()
        self.data = {}

    def update_counter(self, key, value):
        if 'counter_'+key not in self.data:
            self.data['counter_'+key] = 1
        else:
            self.data['counter_'+key] += 1
        return self.data['counter_'+key]

    def update_running(self, key, value, n):
        if 'running_'+key not in self.data:
            self.data['running_'+key] = value
        else:
            self.data['running_'+key] = float(n-1)/n * self.data['running_'+key] + 1.0/n * value

    def update_min(self, key, value):
        if 'min_'+key not in self.data:
            self.data['min_'+key] = value
        else:
            if value < self.data['min_'+key]:
                self.data['min_'+key] = value

    def update_max(self, key, value):
        if 'max_'+key not in self.data:
            self.data['max_'+key] = value
        else:
            if value > self.data['max_'+key]:
                self.data['max_'+key] = value

    def update_variable(self, key, value):
        self.data[key] = value # overwrite
        n = self.update_counter(key, value)
        self.update_running(key, value, n)

    def get_value(self, key):
        if 'running_'+key in self.data:
            return self.data['running_'+key]
        else:
            assert KeyError

    def get_last_value(self, key):
        if key in self.data:
            return self.data[key]
        else:
            assert KeyError


class Saver(object):
    def __init__(self, checkpoint_dir, heapsize=1):
        self.checkpoint_dir = checkpoint_dir
        self.heapsize = heapsize
        self.most_recents = []  # largest is most recent
        self.bests = []  # largest is best

        with open(os.path.join(self.checkpoint_dir, 'summary.csv'), 'w') as f:
            csv_writer = csv.DictWriter(f, fieldnames=['recent', 'best'])
            csv_writer.writeheader()

    def save(self, epoch, state_dict, pfunc):
        ckpt_id = epoch
        ckpt_return = float(state_dict['mean_return'])
        ckpt_name = os.path.join(
            self.checkpoint_dir, 'ckpt_batch{}.pth.tar'.format(epoch))
        heapq.heappush(self.most_recents, (ckpt_id, ckpt_name))
        heapq.heappush(self.bests, (ckpt_return, ckpt_name))
        torch.save(state_dict, ckpt_name)
        self.save_summary()
        pfunc('Saved to {}.'.format(ckpt_name))

    def save_summary(self):
        most_recent = os.path.basename(heapq.nlargest(1, self.most_recents)[0][-1])
        best = os.path.basename(heapq.nlargest(1, self.bests)[0][-1])
        with open(os.path.join(self.checkpoint_dir, 'summary.csv'), 'a') as f:
            csv_writer = csv.DictWriter(f, fieldnames=['recent', 'best'])
            csv_writer.writerow({'recent': most_recent, 'best': best})

    def evict(self):
        to_evict = set()
        if len(self.most_recents) > self.heapsize:
            most_recents = [x[-1] for x in heapq.nlargest(1, self.most_recents)]
            higest_returns = [x[-1] for x in heapq.nlargest(1, self.bests)]

            least_recent = heapq.heappop(self.most_recents)[-1]
            lowest_return = heapq.heappop(self.bests)[-1]

            # only evict least_recent if it is not in highest_returns
            if least_recent not in higest_returns and os.path.exists(least_recent):
                os.remove(least_recent)
            # only evict lowest_return if it is not in lowest_return
            if lowest_return not in most_recents and os.path.exists(lowest_return):
                os.remove(lowest_return)


class BaseLogger(object):
    def __init__(self, args):
        super(BaseLogger, self).__init__()
        self.data = {}
        self.metrics = {}
        self.run_avg = RunningAverage()

    def add_variable(self, name, incl_run_avg=False, metric=None):
        self.data[name] = []
        if incl_run_avg:
            self.data['running_{}'.format(name)] = []
        if metric is not None:
            self.add_metric(
                name='running_{}'.format(name),
                initial_val=metric['value'],
                comparator=metric['cmp'])

    def update_variable(self, name, index, value, include_running_avg=False):
        if include_running_avg:
            running_name = 'running_{}'.format(name)
            self.run_avg.update_variable(name, value)
            self.data[running_name].append((index, self.run_avg.get_value(name)))
        self.data[name].append((index, value))

    def get_recent_variable_value(self, name):
        index, recent_value = self.data[name][-1]
        return recent_value

    def has_running_avg(self, name):
        return self.run_avg.exists(name)

    def add_metric(self, name, initial_val, comparator):
        self.metrics[name] = {'value': initial_val, 'cmp': comparator}

    def plot(self, var_pairs, expname, pfunc):
        self.save_csv(expname, pfunc)
        self.plot_from_csv(
            var_pairs=var_pairs,
            expname=expname)
        self.clear_data()

    def save_csv(self, expname, pfunc):
        csv_dict = defaultdict(dict)
        for key, value in self.data.items():
            for index, e in value:
                csv_dict[index][key] = e
        filename = os.path.join(self.quantitative_dir,'global_stats.csv')
        pfunc('Saving to {}'.format(filename))
        file_exists = os.path.isfile(filename)
        with open(filename, 'a+') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=self.data.keys())
            if not file_exists:
                writer.writeheader()
            for i in sorted(csv_dict.keys()):
                writer.writerow(csv_dict[i])

    def save_print_csv(self, keys, values):
        filename = os.path.join(self.logdir,'eval_print_log.csv')
        with open(filename, 'a+') as outfile:
            csvwriter = csv.writer(outfile)
            csvwriter.writerow(keys)
            for v in values:
                csvwriter.writerow(v)

    def load_csv(self, expname):
        filename = os.path.join(self.quantitative_dir,'global_stats.csv')
        df = pd.read_csv(filename)
        return df

    def plot_from_csv(self, var_pairs, expname):
        df = self.load_csv(expname)
        for var1_name, var2_name in var_pairs:
            data = df[[var1_name, var2_name]].dropna()
            x = data[var1_name].tolist()
            y = data[var2_name].tolist()
            fname = '{}_{}'.format(expname, var2_name)
            plt.plot(x,y)
            plt.xlabel(var1_name)
            plt.ylabel(var2_name)
            plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
            plt.savefig(os.path.join(self.quantitative_dir,'_csv{}.png'.format(fname)))
            plt.close()

    def clear_data(self):
        for key in self.data:
            self.data[key] = []


class MultiBaseLogger(BaseLogger):
    def __init__(self, args):
        super(MultiBaseLogger, self).__init__(args)
        self.args = args
        self.expname = args.expname

        self.subroot = os.path.join('runs', args.subroot)
        self.exproot = os.path.join(self.subroot, self.expname)
        self.logdir = create_logdir(root=self.exproot, dirname='seed{}'.format(args.seed), setdate=True)
        self.code_dir = create_logdir(root=self.logdir, dirname='code', setdate=False)

        params_to_save = vars(args)
        params_to_save = {
            **params_to_save,
            **EnvRegistry().get_reward_normalization_info(args.env_name[0])}

        ujson.dump(params_to_save, open(os.path.join(self.code_dir, 'params.json'), 'w'), sort_keys=True, indent=4)
        self.save_source_code()

        self.print_dirs()
        self.initialize()

    def save_source_code(self):
        for src_folder in ['starter_code', 'launchers', 'mnist']:
            dest_src_folder = os.path.join(self.code_dir, src_folder)
            print('Copying {} to {}'.format(src_folder, dest_src_folder))
            shutil.copytree(src_folder, dest_src_folder)

        for src_file in [x for x in os.listdir('.') if '.py' in x]:
            print('Copying {} to {}'.format(
                src_file, self.code_dir))
            shutil.copy2(src_file, self.code_dir)

    def print_dirs(self):
        s = []
        s.append('subroot: {}'.format(self.subroot))
        s.append('exproot: {}'.format(self.exproot))
        s.append('logdir: {}'.format(self.logdir))
        self.printf('\n'.join(s))

    def initialize(self):
        self.add_variable('epoch')
        self.add_variable('steps')

        self.add_variable('min_return', incl_run_avg=True, metric={'value': -np.inf, 'cmp': operator.ge})
        self.add_variable('max_return', incl_run_avg=True, metric={'value': -np.inf, 'cmp': operator.ge})
        self.add_variable('mean_return', incl_run_avg=True, metric={'value': -np.inf, 'cmp': operator.ge})
        self.add_variable('std_return', incl_run_avg=True, metric={'value': np.inf, 'cmp': operator.le})

        self.add_variable('min_steps', incl_run_avg=True, metric={'value': np.inf, 'cmp': operator.le})
        self.add_variable('max_steps', incl_run_avg=True, metric={'value': np.inf, 'cmp': operator.le})
        self.add_variable('mean_steps', incl_run_avg=True, metric={'value': np.inf, 'cmp': operator.le})
        self.add_variable('std_steps', incl_run_avg=True, metric={'value': np.inf, 'cmp': operator.le})

        self.logged_metrics = ['min_return', 'max_return', 'mean_return', 'std_return',
                       'min_steps', 'max_steps', 'mean_steps', 'std_steps']

    def get_state_dict(self):
        return {'logdir': self.logdir, 'code_dir': self.code_dir}

    def printf(self, string):
        if self.args.printf:
            f = open(os.path.join(self.logdir, self.expname+'.txt'), 'a')
            print(string, file=f)
        else:
            print(string)

    def pprintf(self, string):
        if self.args.printf:
            f = open(os.path.join(self.logdir, self.expname+'.txt'), 'a')
            pprint.pprint(string, stream=f)
        else:
            pprint.pprint(string)

    def remove_exproot(self):
        if self.args.autorm:
            shutil.rmtree(self.exproot)
            print('Removed {}'.format(self.exproot))
        else:
            should_remove = input('Remove {}? [y/n] '.format(self.exproot))
            if should_remove == 'y':
                shutil.rmtree(self.exproot)
                print('Removed {}'.format(self.exproot))
            else:
                print('Did not remove {}'.format(self.exproot))


class EnvLogger(BaseLogger):
    def __init__(self, args):
        super(EnvLogger, self).__init__(args)

    def get_state_dict(self):
        state_dict = {
            **super(EnvLogger, self).get_state_dict(),
            **{'qualitative_dir': self.qualitative_dir, 'quantitative_dir': self.quantitative_dir}}
        return state_dict

    def set_logdir(self, logdir):
        self.logdir = logdir
        self.qualitative_dir = create_logdir(root=self.logdir, dirname='qualitative', setdate=False)
        self.quantitative_dir = create_logdir(root=self.logdir, dirname='quantitative', setdate=False)

        self.checkpoint_dir = create_logdir(root=self.logdir, dirname='checkpoints', setdate=False)
        self.saver = Saver(self.checkpoint_dir)
        print('Qualitative Directory: {}\nQuantitative Directory: {}\nLog Directory: {}\nCheckpoint Directory: {}'.format(self.qualitative_dir, self.quantitative_dir, self.logdir, self.checkpoint_dir))


class EnvManager(EnvLogger):
    def __init__(self, env_name, env_registry, args):
        super(EnvManager, self).__init__(args)
        self.env_name = env_name
        self.env_type = env_registry.get_env_type(env_name)
        self.env = env_registry.get_env_constructor(env_name)()
        self.visual = False  # default
        self.initialize()

    def initialize(self):
        self.add_variable('epoch')
        self.add_variable('steps')

        self.add_variable('min_return', incl_run_avg=True, metric={'value': -np.inf, 'cmp': operator.ge})
        self.add_variable('max_return', incl_run_avg=True, metric={'value': -np.inf, 'cmp': operator.ge})
        self.add_variable('mean_return', incl_run_avg=True, metric={'value': -np.inf, 'cmp': operator.ge})
        self.add_variable('std_return', incl_run_avg=True, metric={'value': np.inf, 'cmp': operator.le})

        self.add_variable('min_steps', incl_run_avg=True, metric={'value': np.inf, 'cmp': operator.le})
        self.add_variable('max_steps', incl_run_avg=True, metric={'value': np.inf, 'cmp': operator.le})
        self.add_variable('mean_steps', incl_run_avg=True, metric={'value': np.inf, 'cmp': operator.le})
        self.add_variable('std_steps', incl_run_avg=True, metric={'value': np.inf, 'cmp': operator.le})

        self.logged_metrics = ['min_return', 'max_return', 'mean_return', 'std_return',
                       'min_steps', 'max_steps', 'mean_steps', 'std_steps']


class HanoiEnvManager(EnvManager):
    def __init__(self, env_name, env_registry, args):
        super(HanoiEnvManager, self).__init__(env_name, env_registry, args)
        self.state_dim = self.env.state_dim
        self.action_dim = self.env.action_dim
        self.is_disc_action = True
        self.max_episode_length = self.env.max_steps


class TabularEnvManager(EnvManager):
    def __init__(self, env_name, env_registry, args):
        EnvManager.__init__(self, env_name, env_registry, args)
        self.state_dim = self.env.state_dim
        self.action_dim = len(self.env.actions)
        self.is_disc_action = True
        self.starting_states = self.env.starting_states
        self.max_episode_length = self.env.eplen
        self.agent_data = {}

    def save_json(self, fname):
        with open(os.path.join(self.quantitative_dir, fname), 'w') as fp:
            ujson.dump(self.agent_data, fp, sort_keys=True, indent=4)

    def record_state_variable(self, state, step, step_dict, metric):
        if metric not in self.agent_data:
            self.agent_data[metric] = {}
        if state not in self.agent_data[metric]:
            self.agent_data[metric][state] = {}
        for a_key in step_dict:
            a_id = int(a_key[len('agent_'):])
            if a_id not in self.agent_data[metric][state]:
                self.agent_data[metric][state][a_id] = {}
            self.agent_data[metric][state][a_id][step] = step_dict[a_key][metric]

    def visualize_data(self, state, title, metric):
        colors = plt.cm.viridis(np.linspace(0,1,len(self.agent_data[metric][state])))
        for i, a_id in sorted(enumerate(self.agent_data[metric][state])):
            data_indices, data_values = zip(*self.agent_data[metric][state][a_id].items())
            plt.plot(data_indices, data_values, label='{} for agent {}'.format(metric, a_id),
             color=colors[i]
                )

        plt.legend()
        if 'bid' in metric:
            plt.ylim(-0.1, 1.1)
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        plt.title(title)
        plt.tight_layout()
        fname = '{}_{}.png'.format(title.replace(' ', '_'), metric)
        plt.savefig(os.path.join(self.quantitative_dir, fname))
        plt.close()


class ComputationEnvManager(EnvManager):
    def __init__(self, env_name, env_registry, args):
        super(ComputationEnvManager, self).__init__(env_name, env_registry, args)
        self.state_dim = self.env.input_dim
        self.action_dim = self.env.output_dim
        self.is_disc_action = True
        self.max_episode_length = self.env.max_steps


class VisualEnvManager(EnvManager):
    def __init__(self, env_name, env_registry, args):
        super(VisualEnvManager, self).__init__(env_name, env_registry, args)
        self.visual = True

    def save_image(self, fname, i, ret, frame, bids_t):
        fig = plt.figure()
        ax1 = fig.add_subplot(121)
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.grid(False)
        ax1.set_title('Timestep: {}'.format(i))
        ax1.imshow(frame)

        ax2 = fig.add_subplot(122)
        agent_ids, bids = zip(*bids_t)
        ax2.bar(agent_ids, bids, align='center', alpha=0.5)
        ax2.set_xticks(range(len(agent_ids)), map(int, agent_ids))
        ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax2.set_ylim(0.0, 1.0)
        ax2.set_ylabel('Bid')
        ax2.set_xlabel('Action')
        fig.suptitle('Return: {}'.format(ret))
        plt.tight_layout(pad=3)
        plt.savefig(os.path.join(self.qualitative_dir, fname))
        plt.close()

    def save_json(self, fname, agent_data):
        with open(os.path.join(self.quantitative_dir, fname), 'w') as fp:
            ujson.dump(agent_data, fp, sort_keys=True, indent=4)

    def save_gif(self, prefix, gifname, epoch, test_example, remove_images):
        def get_key(fname):
            basename = os.path.basename(fname)
            delimiter = '_t'
            start = basename.rfind(delimiter)
            key = int(basename[start+len(delimiter):-len('.png')])
            return key
        fnames = sorted(glob.glob('{}/{}_e{}_n{}_t*.png'.format(self.qualitative_dir, prefix, epoch, test_example)), key=get_key)
        images = [imageio.imread(fname) for fname in fnames]
        imshape = images[0].shape
        for pad in range(2):
            images.append(imageio.core.util.Array(np.ones(imshape).astype(np.uint8)*255))
        imageio.mimsave(os.path.join(self.qualitative_dir, gifname), images)
        if remove_images:
            for fname in fnames:
                os.remove(fname)

    def save_video(self, epoch, test_example, bids, ret, frames, ext=''):
        for i, frame in enumerate(frames):
            fname = '{}_e{}_n{}_t{}.png'.format(self.env_name, epoch, test_example, i)
            agent_ids = sorted(bids.keys())
            bids_t = [(agent_id, bids[agent_id][i]) for agent_id in agent_ids]
            self.save_image(fname, i, ret, frame, bids_t)
        assert test_example == 0
        gifname = 'vid{}_{}_{}{}.gif'.format(self.env_name, epoch, test_example, ext)
        self.save_gif(self.env_name, gifname, epoch, test_example, remove_images=True)


class VisualComputationEnvManager(ComputationEnvManager, VisualEnvManager):
    def __init__(self, env_name, env_registry, args):
        ComputationEnvManager.__init__(self, env_name, env_registry, args)
        self.visual = True


class GymEnvManager(VisualEnvManager):
    def __init__(self, env_name, env_registry, args):
        super(GymEnvManager, self).__init__(env_name, env_registry, args)
        self.state_dim = self.env.observation_space.shape[0]
        self.is_disc_action = len(self.env.action_space.shape) == 0
        self.action_dim = self.env.action_space.n if self.is_disc_action else self.env.action_space.shape[0]
        self.max_episode_length = self.env._max_episode_steps
        

class MinigridEnvManager(VisualEnvManager):
    def __init__(self, env_name, env_registry, args):
        super(MinigridEnvManager, self).__init__(env_name, env_registry, args)
        full_state_dim = self.env.observation_space.shape  # (H, W, C)
        self.state_dim = full_state_dim
        self.is_disc_action = len(self.env.action_space.shape) == 0
        self.action_dim = self.env.action_space.n if self.is_disc_action else self.env.action_space.shape[0]
        self.max_episode_length = self.env.max_steps

    def save_bids_payoffs_by_room(self, room_stats, epoch):
        def create_room_group(hf, room, a_ids):
            roomgrp = hf.create_group(room)
            for a_id in a_ids:
                agent = roomgrp.create_group(a_id)
                agent.create_dataset('payoffs', data=np.array([]),  maxshape=(5000, ))
                agent.create_dataset('bids', data=np.array([]),  maxshape=(5000, ))
            return hf

        filename = os.path.join(self.quantitative_dir,'room_bids_payoffs.h5')
        if not os.path.isfile(filename):
            hf = h5py.File(filename, 'w')
            for room in room_stats:
                hf = create_room_group(hf, room, room_stats[room].keys())
        with h5py.File(filename, 'a') as hf:
            for room in room_stats:
                if room not in hf.keys():
                    hf = create_room_group(hf, room, room_stats[room].keys())
                roomgrp = hf[room]
                for a_id in room_stats[room]:
                    agent= roomgrp[a_id]
                    payoffs = agent['payoffs']
                    newpayoffsize = payoffs.shape[0]+1
                    payoffs.resize((newpayoffsize,))
                    payoffs[newpayoffsize-1]=room_stats[room][a_id]['payoff']
                    bids= agent['bids']
                    newbidsize = bids.shape[0]+1
                    bids.resize((newbidsize,))
                    bids[newbidsize-1]=room_stats[room][a_id]['bid']


    def visualize_bids_payoffs(self, room_stats, epoch):
        ''' Plot bids and payoffs of agents by room from data collected in room_bids_payoffs.h5 '''
        self.save_bids_payoffs_by_room(room_stats, epoch)
        filename = os.path.join(self.quantitative_dir,'room_bids_payoffs.h5')
        x_val = np.array(list(range(epoch+1)))
        with h5py.File(filename, 'r') as hf:
            for room in room_stats:
                plt.figure()
                roomgrp = hf[room]
                for a_id in room_stats[room]:
                    agent= roomgrp[a_id]
                    payoffs = agent['payoffs']
                    try:
                        payoff_vals= np.array(payoffs)
                        plt.plot(np.array(list(range(len(payoff_vals)))), payoff_vals, label='Agent '+a_id)
                    except:
                        print('Room 1 payoffs is not recorded')
                plt.title('Payoffs for Room ' + room)
                plt.xlabel('Epochs')
                plt.ylabel('Bids')
                plt.legend()
                plt.savefig(os.path.join(self.quantitative_dir,'_roombasedpayoffs_room{}.png'.format(room)))
                plt.close()

            for room in room_stats:
                plt.figure()
                roomgrp = hf[room]
                for a_id in room_stats[room]:
                    agent= roomgrp[a_id]
                    bids = agent['bids']
                    try:
                        bid_vals = np.array(bids)
                        plt.plot(np.array(list(range(len(bid_vals)))), bid_vals, label='Agent '+a_id)
                    except:
                        print('Room 1 bids not recorded')
                plt.title('Bids for Room ' + room)
                plt.xlabel('Epochs')
                plt.ylabel('Bids')
                plt.legend()
                plt.savefig(os.path.join(self.quantitative_dir,'_roombasedbids_room{}.png'.format(room)))
                plt.close()


def log_string(ordered_dict):
    s = ''
    for i, (k, v) in enumerate(ordered_dict.items()):
        delim = '' if i == 0 else ' | '
        if is_float(v):
            s += delim + '{}: {:.5f}'.format(k, v)
        else:
            s += delim + '{}: {}'.format(k, v)
    return s


def format_log_string(list_of_rows):
    length = max(len(s) for s in list_of_rows)
    outside_border = '#'*length
    inside_border = '*'*length
    s = '\n'.join([outside_border]+list_of_rows+[outside_border])
    return s


def renderfn(env, scale):
    frame = env.render(mode='rgb_array')

    if frame is not None:
        h, w, c = frame.shape
        frame = cv2.resize(frame, dsize=(int(w*scale), int(h*scale)), interpolation=cv2.INTER_CUBIC)
        return frame


def env_manager_switch(env_name, env_registry):
    envtype = env_registry.get_env_type(env_name)
    env_manager = dict(
        gym = GymEnvManager,
        mg = MinigridEnvManager,
        tab = TabularEnvManager,
        comp = ComputationEnvManager,
        vcomp = VisualComputationEnvManager,
        toh = HanoiEnvManager,
    )
    return env_manager[envtype]