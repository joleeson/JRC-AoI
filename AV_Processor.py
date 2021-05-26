from rl.core import Processor
from config import state_space_size, test_parameters
import numpy as np

observation_size = state_space_size['data_size'] * state_space_size['channel_size'] * state_space_size['road_size'] \
               * state_space_size['weather_size'] * state_space_size['speed_size'] * state_space_size['object_size']
simulation_steps = observation_size * 10


class AVProcessor(Processor):
    def __init__(self, env):
        self.env = env

    def add_agent(self, agent):
        self.agent = agent

    def process_step(self, observation, reward, done, info):
        if done and test_parameters['add_simulation']:
            r_array = []
            # s0 = [observation]
            s0 = self.agent.memory.get_recent_state(observation)
            for i in range(simulation_steps):
                # instead of calling forward(), we use greedy-policy to select action based on current q_values
                q_values = self.agent.compute_q_values(s0)
                a0 = np.argmax(q_values)
                s1 = self.env.state_transition(s0[0], a0)
                r0 = self.env.get_reward(s0[0], a0)
                # s0 = [s1]
                s0 = self.agent.memory.get_recent_state(s1)
                r_array .append(r0)
            ave_r = np.mean(r_array)
            self.env.episode_observation['sim_reward'] = ave_r
        # print('AV_Processor - observation: {}'.format(observation))
        return observation, reward, done, info