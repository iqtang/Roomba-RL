# Roomba RL Simulation

This project implements a reinforcement learning (RL) agent to control a Roomba-like robot in a PyBullet simulated environment. The agent learns to maximize area coverage while avoiding collisions. The RL algorithm used is **PPO (Proximal Policy Optimization)** from `stable-baselines3`.


---
# Requirements
- Python 3.10.x
### Packages

- ```numpy```
- ```matplotlib```
- ```gymnasium```
- ```stable-baselines3```
- ```notebook```
- ```torch```
- ```cloudpickle```
- ```pybullet```

All packages are in ```requirements.txt```

If ```pybullet``` does not install with pip, create a new Conda environment and run:
```bash
conda install -c conda-forge pybullet
```


# Demo Code

### Clone the repository
```bash
git clone https://github.com/iqtang/Roomba-RL.git
```

### Demo Trained Model
- Run ```run_trained_agent.py``` to see trained model
- Currently running ```dynamic_ppo_roomba_final```, can edit ```model_path``` and ```norm_path```

### Demo Training
- Run ```training.py``` to train own models
- Edit RoombaEnv's ```world_type``` ("empty", "obstacle", "carousel") to choose layout
- After training is finished, evaluations are stored in ```roomba_training_data.npz```
- Run ```plotting.ipynb``` to visualize ```roomba_training_data.npz``` data