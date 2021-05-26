"""
J. Lee, D. Niyato, Y. L. Guan, and D. I. Kim, “Learning to Schedule Joint Radar-Communication Requests for Optimal Information Freshness,” in 2020 IEEE Intelligent Vehicles Symposium (IV), 2021

Logger class for use with the KerasRL DQN solver.
"""
from __future__ import division
from rl.callbacks import Callback
import warnings
import timeit
import json
from tempfile import mkdtemp
import numpy as np

from keras import __version__ as KERAS_VERSION
from keras.callbacks import Callback as KerasCallback, CallbackList as KerasCallbackList
from keras.utils.generic_utils import Progbar



class Logger(Callback):
    def __init__(self, filepath, environment, interval=None, weights_filepath=None):
        self.filepath = filepath
        self.weights_filepath = weights_filepath
        self.interval = interval

        # Some algorithms compute multiple episodes at once since they are multi-threaded.
        # We therefore use a dict that maps from episode to metrics array.
        self.metrics = {}
        self.starts = {}
        self.data = {}
        self.actions = []
        self._set_env(env=environment)
        self.best_return = -9999

    def on_train_begin(self, logs):
        """ Initialize model metrics before training """
        self.metrics_names = self.model.metrics_names
        self.best_return = -9999

    def on_train_end(self, logs):
        """ Save model at the end of training """
        self.save_data()

    def on_episode_begin(self, episode, logs):
        """ Initialize metrics at the beginning of each episode """
        assert episode not in self.metrics
        assert episode not in self.starts
        self.metrics[episode] = []
        self.starts[episode] = timeit.default_timer()

    def on_episode_end(self, episode, logs):
        """ Compute and print metrics at the end of each episode """
        duration = timeit.default_timer() - self.starts[episode]

        metrics = self.metrics[episode]
        if np.isnan(metrics).all():
            mean_metrics = np.array([np.nan for _ in self.metrics_names])
        else:
            mean_metrics = np.nanmean(metrics, axis=0)
        assert len(mean_metrics) == len(self.metrics_names)

        data = list(zip(self.metrics_names, mean_metrics))
        data += list(logs.items())
        data += [('episode', episode), ('duration', duration),
                 ('nb_unexpected_ev', self.env.episode_observation['unexpected_ev_counter']), ('mean_action', np.mean(self.actions)),
                 ('wrong_mode_actions', self.env.episode_observation['wrong_mode_actions']),
                 ('throughput', self.env.episode_observation['throughput'] / self.env.episode_observation['step_counter']),
                 ('avg_sim_reward', self.env.episode_observation['sim_reward']),
                 ('data_counter', self.env.episode_observation['data_counter']),
                 ('urgency_counter', self.env.episode_observation['urgency_counter']),
                 ('peak_age_counter', self.env.episode_observation['peak_age_counter']),
                 ('comm_counter', self.env.episode_observation['comm_counter']),
                 ('good_ch_comm', self.env.episode_observation['good_ch_comm']),
                 ('r_age', self.env.episode_observation['r_age']),
                 ('r_radar', self.env.episode_observation['r_radar']),
                 ('r_overflow', self.env.episode_observation['r_overflow']),
                 ('state_map', self.env.episode_observation['state_map']),
                 ('action_map', self.env.episode_observation['action_map']),
                 ('r_age1', self.env.episode_observation['r_age1']),
                 ('r_age2', self.env.episode_observation['r_age2']),
                 ('r_age3', self.env.episode_observation['r_age3']),
                 ('r_age4', self.env.episode_observation['r_age4']),
                 ('r_age5', self.env.episode_observation['r_age5']),
                 ('r_class_age', self.env.episode_observation['r_class_age']),
                 ('r_class_age1', self.env.episode_observation['r_class_age1']),        # additions for JRCwithAoI_c
                 ('r_class_age2', self.env.episode_observation['r_class_age2']),
                 ('r_class_age3', self.env.episode_observation['r_class_age3']),
                 ('r_class_age4', self.env.episode_observation['r_class_age4']),
                 ('r_class_age5', self.env.episode_observation['r_class_age5']),
                 ('state_age_map', self.env.episode_observation['state_age_map']),
                 ('action_age_map', self.env.episode_observation['action_age_map']),
                 ]
        for key, value in data:
            if key not in self.data:
                self.data[key] = []
            self.data[key].append(value)

        if self.interval is not None and episode % self.interval == 0:
            self.save_data()
        
        if self.weights_filepath is not None:
            if self.data['episode_reward'][-1] > self.best_return:
                self.best_return = self.data['episode_reward'][-1]
                self.model.save_weights(self.weights_filepath, overwrite=True)
        
        # Clean up.
        self.actions = []
        del self.metrics[episode]
        del self.starts[episode]

    def on_step_end(self, step, logs):
        """ Append metric at the end of each step """
        if 'metrics' in logs:
            self.metrics[logs['episode']].append(logs['metrics'])
        self.actions.append(logs['action'])

    def save_data(self):
        """ Save metrics in a json file """
        if len(self.data.keys()) == 0:
            return

        # Sort everything by episode.
        assert 'episode' in self.data
        sorted_indexes = np.argsort(self.data['episode'])
        sorted_data = {}
        for key, values in self.data.items():
            assert len(self.data[key]) == len(sorted_indexes)
            # We convert to np.array() and then to list to convert from np datatypes to native datatypes.
            # This is necessary because json.dump cannot handle np.float32, for example.
            sorted_data[key] = np.array([self.data[key][idx] for idx in sorted_indexes]).tolist()

        # Overwrite already open file. We can simply seek to the beginning since the file will
        # grow strictly monotonously.
        with open(self.filepath, 'w') as f:
            json.dump(sorted_data, f)