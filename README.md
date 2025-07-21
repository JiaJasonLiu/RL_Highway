# Reinforcement Learning Conquering the Highway Environment
# Environment

The [Highway Env](https://highway-env.farama.org/) is a Python-based simulation environment for studying and experimenting with decision-making in highway traffic scenarios. It is part of the Farama Foundation's ecosystem and is compatible with OpenAI Gym interfaces.


## Installation
Running the following command to install all the needed packages
The project was built using anaconda python
```
pip install gymnasium
pip install highway-env
pip install --upgrade sympy torch
```

## Policies

Two network policies were used to return the q values of all the available actions

### MLP (Multi-Layer Perceptron)
The kinematic state representation is shaped as a 2D array providing a structured observation of the surrounding vehicles and their relative positions to the agent.

### CNN (Convolutional Neural Network)
Suitable for processing grayscale observations.
Converts image-based data from the highway environment into features that can be used for decision-making.

# Training and Testing
The environment selected for training is 'highway-fast-v0', which accelerates the simulation's step() function, allowing for faster training.

**Training**: To train the agents, run the jupyter notebooks and wait for the agent to train.

**Testing**: To test the agents, call the agent's evaluate method and you will be able to see a pygame open up and the vehicle traversing the environment.
If the pygame doesn't run, try to install it
```
pip install pygame
```


## Metrics, Model Saving, and Loading
**Metrics Tracking**: 
Use TensorBoard for monitoring training metrics. Ensure compatible version of numpy.
```
pip install tensorboard
```

Saving model, metrics and hyperparameters are conditions set inside the agents params. 

**Naming**:
Models are saved with timestamps in the format %Y%m%d%H%M (e.g., 20241228135206 corresponds to 2024/12/28 13:52:06).
This naming convention ensures uniqueness and avoids spaces in filenames.

**Saving**:
The agent training data is saved in a folder named training_results.
The agent model is saved in its folder specified by its policy_save_models

**Loading Model**
To load the model, make sure the current agent is the save policy as the model to be loaded. Then copy the file name of the model and load and evaluate it.

# Agents

## Baselines
Implementation of Random and Human Agent in the Highway Environment for comparison.

## Q-Learning
A foundational algorithm for discrete-action environments.
Learn the action-value function Q(s,a) for policy derivation

## Rainbow Deep Q-Network (DQN)
An enhancement of Q-Learning using neural networks to approximate Q(s,a). Includes replay buffers and target networks for stabilization

## Proximal Policy Optimization (PPO)
A policy-gradient method for continuous and discrete action spaces.
Emphasizes stable policy updates by constraining step sizes.