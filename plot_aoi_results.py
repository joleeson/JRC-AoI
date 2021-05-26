"""
J. Lee, D. Niyato, Y. L. Guan, and D. I. Kim, “Learning to Schedule Joint Radar-Communication Requests for Optimal Information Freshness,” in 2020 IEEE Intelligent Vehicles Symposium (IV), 2021

Plot results from the DQN training.
"""
from __future__ import division
import json
import matplotlib.pyplot as plt
import pandas
import numpy as np
import seaborn as sns
from matplotlib import rc
import os
import pandas as pd


def get_datasets(fpath, condition=None):
    unit = 0
    datasets = []
    for root, dir, files in os.walk(fpath):
        if 'log.json' in files:
            log_path = os.path.join(root,'log.json')
            
            experiment_data = pd.read_json(log_path)
            
            experiment_data.insert(
                len(experiment_data.columns),
                'Unit',
                unit
                )        
            
            experiment_data = experiment_data.ewm(span=50, adjust=False).mean()
            experiment_data.insert(
                len(experiment_data.columns),
                '',#'Condition'
                condition
                )
            
            datasets.append(experiment_data)
            unit += 1
            # if unit ==2:
            #     break
    
    if '.json' in fpath:
        experiment_data = pd.read_json(fpath)
        
        experiment_data = experiment_data.ewm(span=100, adjust=False).mean()
        experiment_data.insert(
                len(experiment_data.columns),
                '',#'Condition'
                condition
                )
        
        datasets.append(experiment_data)
    
    return datasets

def plot_vars(data, values=['episode_reward']):
    data_to_plot = data[['Episode'] + values]
    data_melted = pd.melt(data_to_plot, ['Episode'])
    data_melted.rename(columns={"variable": "",
                         },inplace=True)
    
    sns.set(style="whitegrid", font_scale=1.1)
    # myplot = sns.lineplot(x=data["episode"], y=data[value], hue=data[""]) #,data=data)
    myplot = sns.lineplot(x=data_melted['Episode'], y=data_melted['value'], hue=data_melted[""], markers = True)
    
    
    # axes = myplot.axes
    # axes.set_xlim(0,2500)
    # axes.set_ylim(0,10)
    # axes.set_xlim(0,100) # for 10 users?
    plt.legend(bbox_to_anchor=(-0, 1.05), loc='lower left', borderaxespad=0.,ncol=3)
    plt.show()
    
def plot_data(data, value=['episode_reward']):
    
    sns.set(style="whitegrid", font_scale=1.1)
    myplot = sns.lineplot(x=data["Episode"], y=data[value], hue=data[""]) #,data=data)    
    
    axes = myplot.axes
    axes.set_xlim(0,2500)
    # axes.set_ylim(-1000,-100)
    plt.legend(bbox_to_anchor=(-0, 1.05), loc='lower left', borderaxespad=0.,ncol=2)
    plt.show()

""" DQN """
logdir_dqn = (

    './logs/dqn_AV_JRC_AoI-v3d2_gamma0.99_min_train_rdbad_0.1_wr10_gen2_nn[64, 64]_pol_e-greedy_07-12-2020_21-15-56',
    )

logdir_qlearning = (

    './logs/q_learning3_envAV_JRC_AoI-v3d2_min_avg_rdbad_0.1_pv1_wr10_gen2_01-12-2020_13-38-47',
    )

logdir_switch = (

    './logs/switch_AV_JRC_AoI-v3d2_avg_rdbad_0.1_pv1_wr10_gen2_01-12-2020_13-40-41',
    )

logdir_oracle = (
    './logs/oracle_AV_JRC_AoI-v3d2_avg_rdbad_0.1_pv1_wr10_gen2_30-11-2020_16-39-16',
    )

logdir = logdir_switch + logdir_dqn + logdir_qlearninglogdir = logdir_switch + logdir_oracle + logdir_qlearning + logdir_dqn


legend = ['Round Robin','1-Step Planner','Q-learning','Dueling DDQN']


# EVALUATED_VALUE = ['episode_reward']
EVALUATED_VALUE = 'Total reward, $r$'
# EVALUATED_VALUE = 'Weighted age reward, $w_{age}r_{age}$'
# EVALUATED_VALUE = 'Weighted radar reward, $w_{rad}r_{rad}$'
# EVALUATED_VALUE = 'r_class_age'
# EVALUATED_VALUE = ['r_age', 'r_class_age']
# EVALUATED_VALUE = ['reward, $r$','$w_{age}r_{age}$', 'peak_age_counter']
# EVALUATED_VALUE = ['reward, $r$', '$w_{age}r_{age}$', '$w_{rad}r_{rad}$']
# EVALUATED_VALUE = ['throughput']
# EVALUATED_VALUE = ['wrong_mode_actions']
# EVALUATED_VALUE = 'peak_age_counter'
# EVALUATED_VALUE = 'good_ch_comm'
# EVALUATED_VALUE = 'good_ch_comm %'
# EVALUATED_VALUE = ['r_comm']
# EVALUATED_VALUE = ['r_radar']


""" Gather data from across random seed subfolders """
data = []
for d, legend_title in zip(logdir, legend):
    data += get_datasets(d, legend_title)
        
        
if isinstance(data, list):
    data = pd.concat(data, ignore_index=True)
# data.r_age = data.r_age * 0.002 #0.01
data.r_class_age = data.r_class_age * 0.002
data.r_radar = data.r_radar * 10

data.rename(columns={"episode_reward": "Total reward, $r$",
                      # "r_age": "$w_{age}r_{age}$",
                      "r_class_age": "Weighted age reward, $w_{age}r_{age}$",
                      "r_radar": "Weighted radar reward, $w_{rad}r_{rad}$",
                      "episode": "Episode",
                          },inplace=True)


if isinstance(EVALUATED_VALUE, list):
    values = EVALUATED_VALUE
    plot_vars(data, values=values)
else:
    values = EVALUATED_VALUE
    plot_data(data, value=values)

