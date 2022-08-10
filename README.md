# JRC-AoI
[![DOI](https://zenodo.org/badge/370934498.svg)](https://zenodo.org/badge/latestdoi/370934498)

Supplementary material for the following papers:

<ol>
  <li>
  
  [J. Lee, D. Niyato, Y. L. Guan and D. I. Kim, "Learning to Schedule Joint Radar-Communication With Deep Multi-Agent Reinforcement Learning," in IEEE Transactions on Vehicular Technology, vol. 71, no. 1, pp. 406-422, Jan. 2022, doi: 10.1109/TVT.2021.3124810.](https://ieeexplore.ieee.org/abstract/document/9601214)
  </li>
  <li>
  
  [J. Lee, D. Niyato, Y. L. Guan and D. I. Kim, "Learning to Schedule Joint Radar-Communication Requests for Optimal Information Freshness," 2021 IEEE Intelligent Vehicles Symposium (IV), 2021, pp. 8-15, doi: 10.1109/IV48863.2021.9575131.](https://ieeexplore.ieee.org/abstract/document/9575131)

Also available on [Digital Repository of NTU](https://hdl.handle.net/10356/150718).
</li>
</ol> 

## Getting started
Install the dependencies listed in 'requirements.txt'.

## Running Experiments
The DQN training process, as well as the 1-step planning and round robin baseline algorithms, may be run from the command line. Examples are provided below.
DQN:
```
python dqn_JRC_AoI_d.py --obj avg --rd_bad2b 0.1 0.2 0.1 --w_radar 9 10 1 --w_ovf 1 --data_gen 2 3 1 --nn_size 64 64 --double --dueling --n_experiments 5
```
1-Step Planner:
```
python test_JRC_AoI2b.py --obj avg --mode best --rd_bad2b 0.1 0.2 0.1 --w_radar 9 10 1 --data_gen 2 3 1 --n_experiments 5
```
Round Robin:
```
python test_JRC_AoI2b.py --obj avg --mode rotate --rd_bad2b 0.1 0.2 0.1 --w_radar 9 10 1 --data_gen 2 3 1 --n_experiments 5
```
