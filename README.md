# Snake Game with Reinforcement Learning

This project implements a Snake game environment with a Deep Q-Network (DQN) agent that learns to play the game through reinforcement learning.

## Project Structure

- `snake_env.py`: The Snake game environment (implements a Gym-like interface)
- `snake_renderer.py`: Pygame-based visualization of the Snake game
- `dqn_agent.py`: Implementation of the DQN agent
- `train.py`: Script to train and evaluate the agent
- `requirements.txt`: Project dependencies

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd snake
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Agent

Run the training script:
```bash
python train.py
```

This will:
1. Train the DQN agent for a specified number of episodes
2. Save model checkpoints in the `checkpoints` directory
3. Plot and save training metrics
4. Evaluate the trained agent

### Configuration

You can modify the hyperparameters in `train.py`:

- `GRID_SIZE`: Size of the game grid
- `EPISODES`: Number of episodes for training
- `BATCH_SIZE`: Mini-batch size for training
- `TARGET_UPDATE_FREQ`: Frequency of target network updates
- `RENDER_DURING_TRAINING`: Enable/disable visualization during training
- `RENDER_TRAINED_AGENT`: Enable/disable evaluation with visualization

### Playing the Game Manually

You can also play the game manually:

```python
from snake_env import SnakeGameEnv
from snake_renderer import SnakeRenderer

env = SnakeGameEnv()
renderer = SnakeRenderer(env)
renderer.manual_play()
```

## Reinforcement Learning Approach

The project uses Deep Q-Learning with the following features:
- Experience replay to break correlations in sequential data
- Target network to stabilize learning
- Epsilon-greedy exploration strategy
- Convolutional neural network to process the game state

## License

[Specify license information here] 