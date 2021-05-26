"""
J. Lee, D. Niyato, Y. L. Guan, and D. I. Kim, “Learning to Schedule Joint Radar-Communication Requests for Optimal Information Freshness,” in 2020 IEEE Intelligent Vehicles Symposium (IV), 2021

This program runs the '1-Step Planner' or 'Round Robin' algorithms for the JRC-AoI environment.

"""

from __future__ import division
from JRCwithAOI_v3d import AV_Environment
from config_jrc_aoi_v0 import test_parameters
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as random
import random as python_random
import json
import time
import os
import argparse

"""
CHANGE LOG

15 June 20
- Amended log for 'wrong_mode_actions' to be based on what the env records

16 June 2020
- added function for alternative switch policy (alternate between radar and a chosen communication urgency level)
- logging extra features: data counter, peak age counter, urgency counter, communication counter, comm with good channel

3 July 2020
- for loop for testing across a range of parameters
- argparse input

20 July 2020
- corrected 'r_comm' to 'r_age'

27 July 2020
 - added env.seed(seed) after initialising the environment
 
3 August 2020
 - allows for different value of increments when testing across a range of data_gen, w_rad or pv
 
4 August 2020
- now runs with environment JRCwithAOI_v3a
- command line input option for reward that accounts for 'peak' or 'avg' age. Folder name now reflects this option.

18 August 2020
- json_data log now reset within for loop
 
1 September 2020
- changed for loop to while not done
 
29 Nov 2020
- added functionality for choosing the best instantaneous action
- create environment with choice of transition probabilities

7 Dec 20
- added argparse for w_ovf
"""

def alternative_switch_action(t, num_actions):
    """
    Alternates between communication '0' and a choice of communications actions.
    Cycles between the communication actions

    Parameters
    ----------
    t : time step
    num_actions : Tnumber of communication actions available

    Returns
    -------
    action num

    """
    if t % 2 == 1:
        return 0
    else:
        r = t % (num_actions*2)
        return int(r/2 + 1)

def alt_switch_action5(t, comm_action):
    """
    Alternates between communication '0' and communicating packets with urgency level 'comm_action'.

    Parameters
    ----------
    t : time step
    num_actions : Tnumber of communication actions available

    Returns
    -------
    action num

    """
    if t % 2 == 1:
        return 0
    else:
        return comm_action


parser = argparse.ArgumentParser()
# parser.add_argument('--env-name', type=str, default='AV_JRC_AoI-v0b')
# parser.add_argument('--env-name', type=str, default='AV_JRC_AoI-v3a')
parser.add_argument('--env-name', type=str, default='AV_JRC_AoI-v3d')
parser.add_argument('--obj', choices=['peak','avg'], default='avg')
parser.add_argument('--w_radar', type=int, nargs='+', default=[0,10,1])
parser.add_argument('--w_ovf', type=float, default=1)
parser.add_argument('--pv', type=int, nargs='+', default=[1,2,1])
parser.add_argument('--data_gen', type=int, nargs='+', default=[3,4,1])
parser.add_argument('--rd_bad2bad', type=float, nargs='+', default=[0.1,0.2,0.1])
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--n_experiments', type=int, default=1)
parser.add_argument('--mode', choices=['rotate','urg5','best'], default='rotate')

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
                    
                    json_data = {}
                    json_data['episode'] = []
                    json_data['episode_reward'] = []
                    json_data['nb_unexpected_ev'] = []
                    json_data['nb_episode_steps'] = []
                    json_data['mean_action'] = []
                    json_data['wrong_mode_actions'] = []
                    json_data['throughput'] = []
                    json_data['data_counter'] = []
                    json_data['urgency_counter'] = []
                    json_data['peak_age_counter'] = []
                    json_data['comm_counter'] = []
                    json_data['good_ch_comm'] = []
                    json_data['r_age'] = []
                    json_data['r_radar'] = []
                    json_data['r_overflow'] = []
                    
                    json_data['r_class_age'] = []
                    json_data['r_class_age1'] = []
                    json_data['r_class_age2'] = []
                    json_data['r_class_age3'] = []
                    json_data['r_class_age4'] = []
                    json_data['r_class_age5'] = []
                    
                    json_data['r_age1'] = []
                    json_data['r_age2'] = []
                    json_data['r_age3'] = []
                    json_data['r_age4'] = []
                    json_data['r_age5'] = []
                    
                    if args.mode == 'rotate':
                        folder = './logs/switch_{}_{}_rdbad_{}_pv{}_wr{}_gen{}_{}/{}/'.format(args.env_name, args.obj, rd_bad2bad, pv, w_radar, data_gen, time, seed)
                    elif args.mode == 'urg5':
                        folder = './logs/switch_urg5_{}_{}_rdbad_{}_pv{}_wr{}_gen{}_{}/{}/'.format(args.env_name, args.obj, rd_bad2bad, pv, w_radar, data_gen, time, seed)
                    elif args.mode == 'best':
                        folder = './logs/oracle2_{}_{}_rdbad_{}_pv{}_wr{}_gen{}_{}/{}/'.format(args.env_name, args.obj, rd_bad2bad, pv, w_radar, data_gen, time, seed)
                    if not(os.path.exists(folder)):
                        os.makedirs(folder)
                    logfile = folder + 'log.json'
                        
                    env = AV_Environment(pv=pv/10, w_radar=w_radar, w_overflow = args.w_ovf, data_gen=data_gen, road_sw_bad_to_bad=rd_bad2bad, age_obj=args.obj)
                    env.seed(seed)
                    
                    for e in range(1, int(test_parameters['nb_steps'] / 400) + 1):
                        plot_target = 0
                        cumulative_reward = 0
                        actions = []
                        num_comm_actions = env.action_space.n - 1
                        wrong_mode_actions = 0
                        steps = 0
                        
                        state = env.reset()
                        next_state = state.copy()
                        done = False
                        # for t in range(1000):
                        while not done:
                            state = next_state
                            if args.mode == 'rotate':
                                action = alternative_switch_action(steps, num_comm_actions)
                            elif args.mode == 'urg5':
                                action = alt_switch_action5(steps, 5)
                            elif args.mode == 'best':
                                action, rewards = env.get_best_im_ac(state)
                            next_state, reward, done, info = env.step(action)
                            cumulative_reward += reward
                            actions.append(action)
                            steps += 1
                            
                    
                        # Save data to json file
                        json_data['episode'].append(e)
                        json_data['episode_reward'].append(int(cumulative_reward))
                        json_data['nb_unexpected_ev'].append(env.episode_observation['unexpected_ev_counter'])
                        json_data['nb_episode_steps'].append(steps)
                        json_data['mean_action'].append(np.mean(actions))
                        json_data['wrong_mode_actions'].append(env.episode_observation['wrong_mode_actions'])
                        json_data['throughput'].append(env.episode_observation['throughput'] / 400)
                        
                        json_data['data_counter'].append(env.episode_observation['data_counter'])
                        json_data['urgency_counter'].append(env.episode_observation['urgency_counter'])
                        json_data['peak_age_counter'].append(env.episode_observation['peak_age_counter'])
                        json_data['comm_counter'].append(env.episode_observation['comm_counter'])
                        json_data['good_ch_comm'].append(env.episode_observation['good_ch_comm'])
                        json_data['r_age'].append(env.episode_observation['r_age'])
                        json_data['r_radar'].append(env.episode_observation['r_radar'])
                        json_data['r_overflow'].append(env.episode_observation['r_overflow'])
                        
                        json_data['r_class_age'].append(env.episode_observation['r_class_age'])
                        json_data['r_class_age1'].append(env.episode_observation['r_class_age1'])
                        json_data['r_class_age2'].append(env.episode_observation['r_class_age2'])
                        json_data['r_class_age3'].append(env.episode_observation['r_class_age3'])
                        json_data['r_class_age4'].append(env.episode_observation['r_class_age4'])
                        json_data['r_class_age5'].append(env.episode_observation['r_class_age5'])
                        json_data['r_age1'].append(env.episode_observation['r_age1'])
                        json_data['r_age2'].append(env.episode_observation['r_age2'])
                        json_data['r_age3'].append(env.episode_observation['r_age3'])
                        json_data['r_age4'].append(env.episode_observation['r_age4'])
                        json_data['r_age5'].append(env.episode_observation['r_age5'])
                             
                        print('Episode: {}, Total reward: {}, Steps: {}, Average reward: {}, Mean action: {}'
                              .format(e, cumulative_reward, steps, cumulative_reward / steps, np.mean(actions)))
    
                        
                    with open(logfile,'w') as outfile:
                        json.dump(json_data, outfile)
