"""
J. Lee, D. Niyato, Y. L. Guan, and D. I. Kim, “Learning to Schedule Joint Radar-Communication Requests for Optimal Information Freshness,” in 2020 IEEE Intelligent Vehicles Symposium (IV), 2021

This program trains a DQN agent for the JRC-AoI environment.
A log is written, and the best weights for the Q-network are saved.

"""
import numpy as np
import gym
import argparse
import time
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # use id from $ nvidia-smi

import tensorflow as tf
import random as python_random

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution2D
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy, BoltzmannQPolicy
from rl.memory import SequentialMemory
from rl.callbacks import FileLogger, ModelIntervalCheckpoint

from JRCwithAOI_v3d import AV_Environment
from config_jrc_aoi_v0 import test_parameters, transition_probability, unexpected_ev_prob, state_space_size, action_space_size
from logger_aoi_v0c import Logger
from AV_Processor import AVProcessor
import json


TEST_ID = test_parameters['test_id']
NB_STEPS = test_parameters['nb_steps']
EPSILON_LINEAR_STEPS = test_parameters['nb_epsilon_linear']
TARGET_MODEL_UPDATE = test_parameters['target_model_update']
GAMMA = 0.99       # discount rate
# ALPHA = test_parameters['alpha']
ALPHA = 0.001                           # learning rate
DOUBLE_DQN = False

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['train', 'test'], default='train')
parser.add_argument('--obj', choices=['peak','avg'], default='peak')
parser.add_argument('--env-name', type=str, default='AV_JRC_AoI-v3d_gamma0.99')
parser.add_argument('--weights', type=str, default=None)
parser.add_argument('--w_radar', type=int, nargs='+', default=[0,10,5])
parser.add_argument('--w_ovf', type=float, default=0)
parser.add_argument('--pv', type=int, nargs='+', default=[1,2,1])
parser.add_argument('--data_gen', type=int, nargs='+', default=[2,3,1])
parser.add_argument('--rd_bad2bad', type=float, nargs='+', default=[0.1,0.2,0.1])
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--n_experiments', type=int, default=1)
parser.add_argument('--nn_size', type=int, nargs='+', default = [32, 32])
parser.add_argument('--policy', type=str, default = 'e-greedy')
parser.add_argument('--dueling', action="store_true")
parser.add_argument('--double', action="store_true")

args = parser.parse_args()
time = time.strftime("%d-%m-%Y_%H-%M-%S")


for data_gen in range(args.data_gen[0], args.data_gen[1], args.data_gen[2]):
    for w_radar in range(args.w_radar[0], args.w_radar[1], args.w_radar[2]):
        for pv in range(args.pv[0], args.pv[1], args.pv[2]):
            for rd_bad2bad in np.arange(args.rd_bad2bad[0],args.rd_bad2bad[1],args.rd_bad2bad[2]):
                for e in range(args.n_experiments):
                    """ Set random seeds 
                    https://keras.io/getting_started/faq/
                    """
                    seed = args.seed + e*10
                    # The below is necessary for starting Numpy and Python generated random numbers in a well-defined initial state.
                    np.random.seed(seed)
                    python_random.seed(seed)
                    
                    # The below set_seed() will make random number generation in the TensorFlow backend have a well-defined initial state.
                    # For further details, see: # https://www.tensorflow.org/api_docs/python/tf/random/set_seed
                    tf.set_random_seed(seed)
                    
                    env = AV_Environment(pv=pv/10, w_radar=w_radar, w_overflow = args.w_ovf, data_gen=data_gen, road_sw_bad_to_bad=rd_bad2bad, age_obj=args.obj)
                    env.seed(seed)
                    nb_actions = env.action_space.n
                    if args.policy == 'e-greedy':
                        policy = EpsGreedyQPolicy(eps=.1)
                    elif args.policy == 'boltz':
                        policy = BoltzmannQPolicy()
                    processor = AVProcessor(env)
                    memory = SequentialMemory(limit=50000, window_length=1)
                    model = Sequential()
                    model.add(Flatten(input_shape=(1,) + env.observation_space.shape)) # (1,) + (6,) to make a (1,6)
                    for size in args.nn_size:
                        model.add(Dense(size, activation='relu'))
                    model.add(Dense(nb_actions, activation='linear'))
                    
                    print(model.summary())
                    
                    dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=100,
                                   target_model_update=TARGET_MODEL_UPDATE, policy=policy, processor=processor,
                                   enable_double_dqn=args.double, gamma=GAMMA,enable_dueling_network=args.dueling)
                    dqn.compile(Adam(lr=ALPHA), metrics=['mae'])
                    processor.add_agent(dqn)
                    
                    print('********************* Start {}DQN - test-id: {} ***********************'.
                          format('DOUBLE-' if args.double else '', TEST_ID))
                    print('************************************************************************** \n '
                          '**************************** Simulation parameters*********************** \n'
                          '{} \n {} \n {} \n {} \n {} \n'.format(transition_probability, unexpected_ev_prob, state_space_size,
                                                              action_space_size, test_parameters)
                          + '*************************************************************************** \n')
                    
                    if args.mode == 'train':
                        # directory for folder in which to write experimental data
                        folder = './logs/dqn_{}_min_{}_rdbad_{}_wr{}_gen{}_nn{}_pol_{}_{}/{}/'.format(args.env_name, args.mode,rd_bad2bad, w_radar, data_gen, args.nn_size, args.policy, time, seed)
                        if not(os.path.exists(folder)):
                            os.makedirs(folder)
                        
                        # Write experimental parameters to a json file.
                        paramfile = folder + 'params.json'
                        json_data = {}
                        json_data['double_dqn'] = args.double
                        json_data['dueling'] = args.dueling
                        json_data['policy'] = args.policy
                        json_data['data_gen'] = data_gen/10
                        json_data['w_radar'] = w_radar
                        json_data['w_overflow'] = args.w_ovf
                        json_data['rd_bad2bad'] = rd_bad2bad
                        json_data['gamma'] = GAMMA
                        with open(paramfile,'w') as outfile:
                            json.dump(json_data, outfile)
                        
                        weights_filename = folder + 'weights.h5f'
                        best_weights_filename = folder + 'best_weights.h5f'
                        checkpoint_weights_filename = folder + 'weights_step{step}.h5f'
                        log_filename = folder + 'log.json'
                        
                        # Configure logging function
                        callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=40000)]
                        callbacks += [Logger(log_filename, environment=env, interval=100, weights_filepath=best_weights_filename)]
                        
                        # Train using DQN
                        dqn.fit(env, nb_steps=NB_STEPS, visualize=False, verbose=2, nb_max_episode_steps=None, callbacks=callbacks)
                        dqn.save_weights(weights_filename, overwrite=True)
                        dqn.test(env, nb_episodes=10, visualize=False)
                    elif args.mode == 'test':
                        weights_filename = args.weights.format(pv, w_radar, data_gen) + '/{}/best_weights.h5f'.format(seed)
                        dqn.load_weights(weights_filename)
                        log_filename = args.weights.format(pv, w_radar, data_gen) + '/{}/testlog.json'.format(seed)
                        callbacks = [Logger(log_filename, environment=env, interval=100)]
                        dqn.test(env, nb_episodes=500, visualize=False, verbose=2, callbacks=callbacks)
                    
                    print("****************************************"
                          " End of training {}-th " 
                          "****************************************".format(TEST_ID))
