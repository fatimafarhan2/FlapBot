# Reinforcement Learning: Flappy Bird & CartPole

A deep Q-Network (DQN) implementation for training RL agents on two classic environments: **Flappy Bird** and **CartPole** using the Gymnasium library with PyTorch.

## Project Overview

This project demonstrates the application of reinforcement learning algorithms to solve two different control tasks:
- **Flappy Bird**: A bird avoidance game where the agent learns to navigate obstacles
- **CartPole**: A classic control problem where the agent learns to balance a pole on a moving cart

The implementation features advanced DQN techniques including:
- **Experience Replay**: Stores and samples transitions for efficient learning
- **DQN**: Deep Q-Network algorithm for learning optimal policies
- **Double DQN**: Reduces overestimation of Q-values
- **Dueling DQN**: Separates value and advantage streams for improved Q-value estimation
- **Configurable Hyperparameters**: YAML-based configuration for easy experimentation

## Requirements

- Python 3.8+
- PyTorch
- Gymnasium
- Flappy Bird Gymnasium
- PyYAML
- NumPy
- Matplotlib

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd flappy-bird

# Create and activate virtual environment (optional but recommended)
conda create -n dqnenv python=3.11
conda activate dqnenv

# Install dependencies
pip install torch gymnasium flappy-bird-gymnasium pyyaml numpy matplotlib
```

## Project Structure

```
.
├── agent.py                    # Main training agent and RL loop
├── dqn.py                      # DQN neural network architecture
├── experience_replay.py         # Experience replay memory buffer
├── hyperparameters.yaml        # Environment-specific hyperparameters
└── README.md                   # This file
```

## Files Description

### `agent.py`
The main training loop that:
- Initializes the RL environment
- Manages the DQN agent lifecycle
- Handles training episodes and experience collection
- Logs training metrics and saves model checkpoints

### `dqn.py`
Defines the Deep Q-Network architecture with:
- Fully connected layers
- Optional Dueling DQN configuration (separate value and advantage streams)
- Q-value computation and prediction

### `experience_replay.py`
Implements a replay memory buffer for:
- Storing agent experiences (state, action, reward, next_state)
- Sampling mini-batches for training
- Efficient memory management with fixed-size deque

### `hyperparameters.yaml`
Configuration file with hyperparameter sets for:
- **cartpole1**: CartPole-v1 environment settings
- **flappybird2**: FlappyBird-v0 environment settings

Each set includes learning rate, epsilon decay, network architecture, and environment-specific parameters.

## Usage

### Training an Agent

```bash
# Train on CartPole
python agent.py cartpole1 --train

# Train on Flappy Bird
python agent.py flappybird2 --train
```

The script will:
1. Load the specified hyperparameter configuration
2. Initialize the Gymnasium environment
3. Train the DQN agent over multiple episodes
4. Save the trained model to the `runs/` directory
5. Generate training plots showing reward progression

### Running a Trained Agent

To run a trained agent (using a saved model):

```bash
# Run trained Flappy Bird agent
python agent.py flappybird2

# Run trained CartPole agent
python agent.py cartpole1
```

The script will load the previously trained model from the `runs/` directory and run the agent in the environment (without training).

### Adding New Environments

1. Add a new hyperparameter set to `hyperparameters.yaml`:
```yaml
new_env:
  env_id: 'YourEnv-v0'
  replay_memory_size: 100000
  mini_batch_size: 32
  epsilon_init: 1.0
  epsilon_decay: 0.9995
  epsilon_min: 0.05
  network_sync_rate: 10
  learning_rate_a: 0.001
  discount_factor_g: 0.99
  fc1_nodes: 256
```

2. Train using: `python agent.py new_env`

## DQN Techniques Explained

### Deep Q-Network (DQN)
DQN uses a neural network to approximate the Q-function, which maps states to action values. Instead of storing a table of all state-action pairs (impractical for large state spaces), the network learns to predict Q-values directly. The agent uses an **epsilon-greedy** policy: it explores randomly with probability epsilon and exploits the best-known action with probability (1-epsilon).

### Double DQN
Standard DQN tends to overestimate Q-values because it uses the same network to both select and evaluate actions. Double DQN addresses this by using two networks:
- **Policy Network**: Selects the best action
- **Target Network**: Evaluates the Q-value of that action

This decoupling reduces overestimation bias and leads to more stable learning.

### Dueling DQN
Dueling DQN decomposes the Q-value into two components:
- **State Value (V)**: How good the current state is, regardless of action
- **Action Advantage (A)**: How much better one action is compared to others

The final Q-value is computed as: **Q = V + (A - mean(A))**

This architecture helps the agent learn state values more effectively, especially when multiple actions have similar values.

### Experience Replay
Instead of training on experiences sequentially (which causes high correlation), experience replay stores transitions in a memory buffer and samples random mini-batches for training. This breaks temporal correlations, improves sample efficiency, and leads to more stable learning.

## Key Hyperparameters

- **epsilon_init**: Initial exploration rate
- **epsilon_decay**: Rate at which exploration decreases per episode
- **epsilon_min**: Minimum exploration rate (exploration floor)
- **learning_rate_a**: Adam optimizer learning rate for the DQN
- **discount_factor_g**: Gamma value for discounting future rewards
- **replay_memory_size**: Maximum size of the experience replay buffer
- **mini_batch_size**: Batch size for training updates
- **network_sync_rate**: Frequency of synchronizing target network with policy network
- **fc1_nodes**: Hidden layer size in the neural network

## Model Checkpoints

Trained models are saved in the `runs/` directory with names like:
- `cartpole1.pt`
- `flappybird2.pt`

## Libraries Used

- **Gymnasium**: RL environment toolkit (successor to OpenAI Gym)
- **Flappy Bird Gymnasium**: Flappy Bird environment wrapper for Gymnasium
- **PyTorch**: Deep learning framework for neural networks
- **PyYAML**: YAML configuration file parsing
- **NumPy**: Numerical computing
- **Matplotlib**: Plotting and visualization

## Notes

- The agent uses CPU by default (`device='cpu'`) for compatibility
- Training progress is visualized through Matplotlib plots (saved as images)
- The experience replay buffer uses a deque for efficient fixed-size memory management

## License

[Add your license here]

## Author

[Your name/organization]
