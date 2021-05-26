"""
J. Lee, D. Niyato, Y. L. Guan, and D. I. Kim, “Learning to Schedule Joint Radar-Communication Requests for Optimal Information Freshness,” in 2020 IEEE Intelligent Vehicles Symposium (IV), 2021

Environment JRC-AoI MDP, in the style of OpenAI Gym
"""

import gym
import numpy as np
import random
from gym import spaces
from gym.utils import seeding
from config_jrc_aoi_v0 import transition_probability, unexpected_ev_prob, state_space_size, r_params



class AV_Environment(gym.Env):
    def __init__(self, pv=0.1, w_radar=5, w_overflow=0, data_gen=2, road_sw_bad_to_bad=0.1, age_obj = 'peak'):
        
        r_params['age_obj'] = age_obj
        r_params['w_overflow'] = w_overflow
        assert (r_params['age_obj'] == 'peak' or r_params['age_obj'] =='avg')
        self.opt_obj = r_params['age_obj']
        max_age =  r_params['A_max']
        urg_lvls = transition_probability['arrival_mean'].shape[0]
        high = np.concatenate(
            ((max_age+1)*np.ones(state_space_size['data_size']),        # age of each packet
             urg_lvls*np.ones(state_space_size['data_size']),           # urgency level of each packet
             np.ones(state_space_size['channel_size'] +                 # environmental conditions
                     state_space_size['road_size'] + 
                     state_space_size['weather_size'] + 
                     state_space_size['speed_size'] + 
                     state_space_size['object_size']),
             (max_age+1)*np.ones(state_space_size['num_classes']),      # age of each class
             )
            )
        self.observation_space = spaces.Box(
            low=0, high=high, shape=(state_space_size['data_size']*2 +
            state_space_size['channel_size'] +
            state_space_size['road_size'] +
            state_space_size['weather_size'] +
            state_space_size['speed_size'] +
            state_space_size['object_size'] +
            state_space_size['num_classes'],)
            )

        # choice of radar mode or which 2 in data queue to transmit
        self.action_space = spaces.Discrete(urg_lvls + 1)
        
        unexpected_ev_prob['occur_with_fast_speed'] = pv
        r_params['w_radar'] = w_radar
        transition_probability['arrival_mean'] = data_gen/10 * np.ones((5,))
        transition_probability['road_sw_bad_to_bad'] = road_sw_bad_to_bad
        
        self.seed(123)
        self.state = self.reset()
        # print(self.state)
    
    def action_decode(self, action):
        index = action - 1
        ac = np.unravel_index(index, (state_space_size['data_size'],state_space_size['data_size']))
        return ac
    
    def markov_transition(self, current_state, state_id):
        """
        Parameters
        ----------
        current_state : indicator state for one of the following variables: channel, road, weather, speed
        state_id : indicator of which state variable is being input as 'current_state'

        Returns
        -------
        current_state : indicator state at time t+1

        """
        if current_state > 1 or current_state < 0:
            raise Exception('Invalid current_state')
        if state_id > 5 or state_id < 1:
            raise Exception('state_id should not exceed 5 or below 1')

        markov_probability = 0.0
        if current_state == 1:
            if state_id == 1:
                markov_probability = transition_probability['channel_sw_bad_to_bad']
            if state_id == 2:
                markov_probability = transition_probability['road_sw_bad_to_bad']
            if state_id == 3:
                markov_probability = transition_probability['weather_sw_bad_to_bad']
            if state_id == 4:
                markov_probability = transition_probability['speed_sw_fast_to_fast']
            if state_id == 5:
                markov_probability = transition_probability['object_sw_moving_to_moving']
            if self.nprandom.uniform() < markov_probability:
                current_state = 1
            else:
                current_state = 0
        else:
            if state_id == 1:
                markov_probability = transition_probability['channel_sw_good_to_good']
            if state_id == 2:
                markov_probability = transition_probability['road_sw_good_to_good']
            if state_id == 3:
                markov_probability = transition_probability['weather_sw_good_to_good']
            if state_id == 4:
                markov_probability = transition_probability['speed_sw_slow_to_slow']
            if state_id == 5:
                markov_probability = transition_probability['object_sw_static_to_static']
            if self.nprandom.uniform() < markov_probability:
                current_state = 0
            else:
                current_state = 1

        return current_state
    
    
    def state_transition(self, state, action, transmitted_packets, new_packets):
        """
        The state transition function. 
        Called by:  'step()' function.
        Used by:    'markov_transition'
        
        Parameters
        ----------
        state : Markov state at time t
        action : action at time t

        Returns
        -------
        Markov state at time t+1

        """
        
        
        
        self.data_age[self.data_urg>0] = self.data_age[self.data_urg>0] + 1   # Increment age counter by 1
        self.class_age = self.class_age + 1        
                
        new_data_urg = np.concatenate([np.tile(i+1,new_packets[i]) for i in range(5)])
        self.data_age = np.concatenate((self.data_age[self.data_urg!=0], np.zeros(state_space_size['data_size'])[self.data_urg==0]))
        self.data_urg = np.concatenate((self.data_urg[self.data_urg!=0], new_data_urg, self.data_urg[self.data_urg==0]))[:state_space_size['data_size']]
        
        # assert self.data_age.shape == (11,)
        # assert self.data_urg.shape == (11,)
        
        # TODO transition for Markov chain
        self.channel = np.array([self.markov_transition(self.channel, 1)])
        self.road = np.array([self.markov_transition(self.road, 2)])
        self.weather = np.array([self.markov_transition(self.weather, 3)])
        if self.object == 1:
            if action == 0:  # if RADAR mode
                self.speed = np.array([0])
                # if the AV's speed is slowed down, the moving object's state should be changed as well
                self.object =  np.array([0]) if np.random.uniform() < 0.9 else  np.array([1])  # 90% it change to good state
            else:
                self.speed = np.array([self.markov_transition(self.speed, 4)])
                self.object = np.array([self.markov_transition(self.object, 5)])
        else:
            self.speed = np.array([self.markov_transition(self.speed, 4)])
            self.object = np.array([self.markov_transition(self.object, 5)])

        # Log statistics
        self.episode_observation['throughput'] += int(np.sum(transmitted_packets))
        self.episode_observation['data_counter'] += int(np.sum(new_packets))
        self.episode_observation['urgency_counter'] += int(np.sum(new_data_urg))
        return self._get_obs()

    def risk_assessment(self, state):
        
        nb_bad_bits = np.sum((self.road, self.weather, self.speed, self.object))
        
        if self.nprandom.uniform() < np.exp(nb_bad_bits)/np.exp(4):
            unexpected_event = 1
        else:
            unexpected_event = 0
        
        return unexpected_event

    def get_reward(self, state, action):
        """
        The environment generates a reward rt when the agent makes an action a(t) from state s(t)

        Parameters
        ----------
        state : state at time t
        action : action taken by the agent at time t

        Returns
        -------
        reward : reward at time t
        transmitted_packets: (11,) np array where 1s indicate transmission and 0 indicates no transmission

        """
        unexpected_ev_occurs = self.risk_assessment(state)
        channel_state = self.channel
        nb_bad_bits = np.sum((self.road, self.weather, self.speed, self.object))
        # nb_bad_bits = 0
        r_age, r_radar, r_overflow = 0, 0, 0
        r_age_by_channel = np.zeros((5))
        
        expired_data = np.ones_like(self.data_age) * (self.data_age > r_params['A_max'])
        if np.sum(expired_data) > 0:
            print('expired data')
        
        
        transmitted_packets = np.zeros_like(self.data_urg)
        self.episode_observation['state_map'][nb_bad_bits, self.channel] += 1
        # if max(self.class_age) > 90:
        #     print('max class age is above 90')
        self.episode_observation['state_age_map'][np.arange(state_space_size['num_classes']), np.minimum(np.ones_like(self.class_age)*100,self.class_age.astype(int))] += 1
        if action != 0:                  # COMMUNICATION Mode
            self.episode_observation['comm_counter'] += 1
            self.episode_observation['action_age_map'][action-1, min(100,self.class_age.astype(int)[action-1])] += 1
            if unexpected_ev_occurs == 0:   # No bad/unexpected events

                idx = np.argwhere(self.data_urg == action).squeeze(1)
                
                if channel_state == 0:      # Channel is good
                    transmitted_packets = transmitted_packets + np.sum([np.eye(state_space_size['data_size'])[idx[i]] for i in range(min(2,len(idx)))],axis=0)
                    if len(idx) != 0:
                        self.class_age[action-1] = self.data_age[idx[min(1,len(idx)-1)]]       # if data was transmitted, set age of channel to age of last transmitted packet
                    self.episode_observation['good_ch_comm'] += 1
                else:
                    transmitted_packets = transmitted_packets + np.eye(state_space_size['data_size'])[idx[0]] if len(idx) != 0 else transmitted_packets
                    if len(idx) != 0:
                        self.class_age[action-1] = self.data_age[idx[0]]       # if data was transmitted, set age of channel to age of last transmitted packet
            else:
                r_radar = -nb_bad_bits
                self.episode_observation['wrong_mode_actions'] += 1
        else:                               # RADAR Mode
            self.episode_observation['action_map'][nb_bad_bits, self.channel] += 1
            # if unexpected_ev_occurs == 0:
            #     r_radar = 0
            # else:
            #     r_radar = nb_bad_bits
        
        # Remove transmitted packets and expired packets from the data queue
        unexpired_data = (self.data_age <= r_params['A_max'])
        self.data_urg = (transmitted_packets==0) * (unexpired_data) * self.data_urg
        self.data_age = (transmitted_packets==0) * (unexpired_data) * self.data_age
        
        if r_params['age_obj'] == 'peak':
            # r_age = np.sum(- r_params['phi'] * expired_data)
            r_age = np.sum(-self.data_urg*((self.data_age > 0) & (self.data_age <= r_params['A_max'])) - r_params['phi'] * expired_data)
        elif r_params['age_obj'] == 'avg':
            r_age = -self.data_urg*self.data_age*(self.data_age <= r_params['A_max']) - r_params['phi'] * expired_data
            for i in range(5):
                r_age_by_channel[i]=np.sum((self.data_urg==(i+1)) * r_age)
            r_age = np.sum(r_age)
        r_class_age = np.sum(-(self.class_age+1) * np.array([1,2,3,4,5]))    # multiply class age by urgency level
            
        
        new_packets = self.nprandom.poisson(transition_probability['arrival_mean'])     # Num new packets arriving in data queue.    
        excess_packets = np.sum(self.data_urg > 0 - expired_data - transmitted_packets) + np.sum(new_packets) - state_space_size['data_size']
        excess_packets = excess_packets * (excess_packets > 0)
        r_overflow += - excess_packets
        
        # reward = r_params['w_age']*r_age + r_params['w_radar']*r_radar + r_params['w_overflow']*r_overflow
        reward = r_params['w_age']*r_class_age + r_params['w_radar']*r_radar + r_params['w_overflow']*r_overflow
        
        if type(reward) == type(np.array([1])):
            print('reward is np')
        
        # Log statistics
        if unexpected_ev_occurs == 1:
            self.episode_observation['unexpected_ev_counter'] += 1
        # eliminated = np.maximum(transmitted_packets,expired_data)
        self.episode_observation['peak_age_counter'] = self.episode_observation['peak_age_counter'] + int(np.sum((np.maximum(transmitted_packets,expired_data) * self.data_age)))
        self.episode_observation['r_age'] += int(r_age)
        self.episode_observation['r_radar'] += int(r_radar)
        self.episode_observation['r_overflow'] += int(r_overflow)
        for i in range(5):
            self.episode_observation['r_age'+str(i+1)] += int(r_age_by_channel[i])
            self.episode_observation['r_class_age'+str(i+1)] += int(self.class_age[i] * (i+1))
        self.episode_observation['r_class_age'] += int(r_class_age)
        # print(nb_bad_bits)
        return reward, transmitted_packets, new_packets
    
    def get_best_im_ac_1(self, state):
        """
        The environment generates a reward rt when the agent makes an action a(t) from state s(t)

        Parameters
        ----------
        state : state at time t
        action : action taken by the agent at time t

        Returns
        -------
        reward : reward at time t
        transmitted_packets: (11,) np array where 1s indicate transmission and 0 indicates no transmission

        """
        unexpected_ev_occurs = self.risk_assessment(state)
        channel_state = self.channel
        nb_bad_bits = np.sum((self.road, self.weather, self.speed, self.object))
        # nb_bad_bits = 0
        r_radar, r_class_age = 0, 0
        reward = []
        next_class_age = self.class_age.copy()
        transmitted_packets = np.zeros_like(self.data_urg)
        
        for action in range(6):
            next_class_age = self.class_age.copy()
            if action != 0:                  # COMMUNICATION Mode
                if unexpected_ev_occurs == 0:   # No bad/unexpected events
    
                    idx = np.argwhere(self.data_urg == action).squeeze(1)
                    
                    if channel_state == 0:      # Channel is good
                        transmitted_packets = transmitted_packets + np.sum([np.eye(state_space_size['data_size'])[idx[i]] for i in range(min(2,len(idx)))],axis=0)
                        if len(idx) != 0:
                            next_class_age[action-1] = self.data_age[idx[min(1,len(idx)-1)]]       # if data was transmitted, set age of channel to age of last transmitted packet
                    else:
                        transmitted_packets = transmitted_packets + np.eye(state_space_size['data_size'])[idx[0]] if len(idx) != 0 else transmitted_packets
                        if len(idx) != 0:
                            next_class_age[action-1] = self.data_age[idx[0]]       # if data was transmitted, set age of channel to age of last transmitted packet
                else: # if unexpected event occurs
                    r_radar = -nb_bad_bits
            
            next_class_age = next_class_age + 1
            r_class_age = np.sum(-next_class_age * np.array([1,2,3,4,5]))    # multiply class age by urgency level
            reward.append(r_params['w_age']*r_class_age + r_params['w_radar']*r_radar)
        
        
        max_rew = max(reward)
        actions = [i for i, j in enumerate(reward) if j == max_rew]
        action = random.choice(actions)
        
        return action, reward
    
    def get_best_im_ac(self, state):
        """
        The environment generates a reward rt when the agent makes an action a(t) from state s(t)

        Parameters
        ----------
        state : state at time t
        action : action taken by the agent at time t

        Returns
        -------
        reward : reward at time t
        transmitted_packets: (11,) np array where 1s indicate transmission and 0 indicates no transmission

        """
        unexpected_ev_occurs = self.risk_assessment(state)
        channel_state = self.channel
        nb_bad_bits = np.sum((self.road, self.weather, self.speed, self.object))
        # nb_bad_bits = 0
        r_radar, r_class_age = 0, 0
        reward = []
        next_class_age = self.class_age.copy()
        transmitted_packets = np.zeros_like(self.data_urg)
        expired_data = np.ones_like(self.data_age) * (self.data_age > r_params['A_max'])
        
        for action in range(6):
            next_class_age = self.class_age.copy()
            if action != 0:                  # COMMUNICATION Mode

                idx = np.argwhere(self.data_urg == action).squeeze(1)
                
                if channel_state == 0:      # Channel is good
                    transmitted_packets = transmitted_packets + np.sum([np.eye(state_space_size['data_size'])[idx[i]] for i in range(min(2,len(idx)))],axis=0)
                    if len(idx) != 0:
                        next_class_age[action-1] = self.data_age[idx[min(1,len(idx)-1)]]       # if data was transmitted, set age of channel to age of last transmitted packet
                else:
                    transmitted_packets = transmitted_packets + np.eye(state_space_size['data_size'])[idx[0]] if len(idx) != 0 else transmitted_packets
                    if len(idx) != 0:
                        next_class_age[action-1] = self.data_age[idx[0]]       # if data was transmitted, set age of channel to age of last transmitted packet
                
                r_radar = -np.exp(nb_bad_bits)/np.exp(4) * nb_bad_bits
                
            excess_packets = np.sum(self.data_urg > 0 - expired_data - transmitted_packets) + 1 - state_space_size['data_size']
            excess_packets = excess_packets * (excess_packets > 0)
            r_overflow = - excess_packets    
            
            next_class_age = next_class_age + 1
            r_class_age = np.sum(-next_class_age * np.array([1,2,3,4,5]))    # multiply class age by urgency level
            reward.append(r_params['w_age']*r_class_age + r_params['w_radar']*r_radar + r_params['w_overflow']*r_overflow)
        
        
        max_rew = max(reward)
        actions = [i for i, j in enumerate(reward) if j == max_rew]
        action = random.choice(actions)
        
        return action, reward
    
    def step(self, action):
        """
        The agent takes a step in the environment.
        Parameters
        ----------
        action : action the agent has decided to take at time t

        Returns
        -------
        next_state : state at time t+1
        reward : reward at time t
        done : indicator of wether the episode has ended

        """
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        state = np.copy(self.state)
        reward, transmitted_packets, new_packets = self.get_reward(state, action)
        next_state = self.state_transition(state, action, transmitted_packets, new_packets)
        self.episode_observation['step_counter'] += 1
        if self.episode_observation['step_counter'] == 400:
            done = True
            print("Peak age: ", self.episode_observation['peak_age_counter']/(self.episode_observation['throughput'] + 1e-7))
            print("Wrong mode actions: ", self.episode_observation['wrong_mode_actions'])
        else:
            done = False
        self.state = next_state

        return self._get_obs(), reward, done, {}

    def reset(self):
        """
        Reset the environment and return the initial state s0

        Returns
        -------
        state at time t=0

        """
        self.episode_observation = {
            'step_counter': 0,
            'unexpected_ev_counter': 0,
            'wrong_mode_actions': 0,
            'throughput': 0,
            'sim_reward': 0,
            'data_counter': 0,
            'urgency_counter': 0,
            'peak_age_counter': 0,
            'good_ch_comm': 0,
            'comm_counter': 0,
            'r_age': 0,
            'r_radar': 0,
            'r_class_age': 0,
            'action_map': np.zeros((5,2), dtype=int),  # num bad bits x channel. Counter for discrete category is incremented when comm action is taken
            'state_map': np.zeros((5,2), dtype=int),   # num bad bits x channel. Counter for discrete category is incremented when state is visited
            'action_age_map': np.zeros((5,r_params['A_max']+1), dtype=int),  # num classes x age. Counter for discrete category is incremented when comm action is taken
            'state_age_map': np.zeros((5,r_params['A_max']+1), dtype=int),   # num classes x age. Counter for discrete category is incremented when state is visited
            'r_overflow': 0,
        }
        for i in range(5):
            self.episode_observation['r_age'+str(i+1)] = 0
            self.episode_observation['r_class_age'+str(i+1)] = 0
        self.data_urg = np.zeros(state_space_size['data_size'], dtype=int)
        self.data_age = np.zeros(state_space_size['data_size'], dtype=int)
        
        self.class_age = np.zeros(state_space_size['num_classes'], dtype=int)
        
        new_packets = self.nprandom.poisson(transition_probability['arrival_mean'])
        new_data_urg = np.concatenate([np.tile(i+1,new_packets[i]) for i in range(5)])
        len_diff = state_space_size['data_size']-len(new_data_urg)
        if len_diff >= 0:
            self.data_urg = np.pad(new_data_urg, (0,len_diff),'constant',constant_values=0)
        else:
            self.data_urg = new_data_urg[:state_space_size['data_size']]
        
        # self.data_urg = self.nprandom.poisson(transition_probability['arrival_mean'])
        self.channel = np.random.randint(2,size=state_space_size['channel_size'])
        self.road = np.random.randint(2,size=state_space_size['road_size'])
        self.weather = np.random.randint(2,size=state_space_size['weather_size'])
        self.speed = np.random.randint(2,size=state_space_size['speed_size'])
        self.object = np.random.randint(2,size=state_space_size['object_size'])
        
        # Log statistics
        self.episode_observation['data_counter'] += int(np.sum(new_packets))
        self.episode_observation['urgency_counter'] += int(np.sum(new_data_urg))
        
        state = self._get_obs()
        # assert state.shape == (27,)
        
        
        return state
    
    def _get_obs(self):
        state = np.concatenate((self.data_age/10, self.data_urg/5, self.channel, self.road, self.weather, self.speed, self.object, self.class_age/10))
        # assert state.shape == (27,)
        return state
    
    def seed(self, seed=None):
        self.nprandom, seed = seeding.np_random(seed)
        return [seed]
    
    def seed_new(self, seed):
        np.random.seed = seed
    
    def is_terminated(self):
        ...

