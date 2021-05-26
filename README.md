# JRC-AoI
Code for the paper "Learning to Schedule Joint Radar-Communication Requests for Optimal Information Freshness" as published in IEEE Intelligent Vehicles Symposium 2021

## Getting started
Install the dependencies listed in 'requirements.txt'.

## Running DQN
The DQN training process, as well as the 1-step planning and round robin baseline algorithms, may be run from the command line. Examples are provided below.
DQN:
```
ython dqn_JRC_AoI_d.py --obj avg --rd_bad2b 0.1 0.2 0.1 --w_radar 9 10 1 --w_ovf 1 --data_gen 2 3 1 --nn_size 64 64 --double --dueling --n_experiments 5
```
1-Step Planner:
```
python test_JRC_AoI2b.py --obj avg --mode best --rd_bad2b 0.1 0.2 0.1 --w_radar 9 10 1 --data_gen 2 3 1 --n_experiments 5
```
Round Robin:
```
python test_JRC_AoI2b.py --obj avg --mode rotate --rd_bad2b 0.1 0.2 0.1 --w_radar 9 10 1 --data_gen 2 3 1 --n_experiments 5
```
