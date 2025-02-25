# Snake Game with Reinforcement Learning

This project implements a Snake game environment with a Deep Q-Network (DQN) agent that learns to play the game through reinforcement learning.

## Project Structure

- `snake_env.py`: Snake game environment (implements a Gym-like interface)
- `snake_renderer.py`: Pygame-based visualization of the Snake game
- `dqn_agent.py`: Implementation of the DQN agent
- `train.py`: Script to train and evaluate the agent
- `play.py`: Script to play the game manually
- `visualize_agent.py`: Script to visualize the trained agent
- `requirements.txt`: Project dependencies

## Installation

1. Clone the repository:
```bash
git clone https://github.com/PopSoda2002/SnakeDQN.git
cd SnakeDQN
```

2. Install required dependencies:
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
3. Evaluate the trained agent

### Visualizing the Trained Agent

Use the visualization script to see the trained agent in action:
```bash
python visualize_agent.py --model_path checkpoints/best_model.weights.h5 --fps 5 --episodes 3
```

Parameters:
- `--model_path`: Path to the model weights file
- `--fps`: Game frame rate
- `--episodes`: Number of episodes to run

### Playing the Game Manually

You can also play the game manually:
```bash
python play.py
```

Use the arrow keys to control the snake's direction.

### Configuration

You can modify the hyperparameters in `train.py`:

- `GRID_SIZE`: Size of the game grid
- `EPISODES`: Number of episodes for training
- `BATCH_SIZE`: Mini-batch size for training
- `TARGET_UPDATE_FREQ`: Frequency of target network updates
- `RENDER_DURING_TRAINING`: Enable/disable visualization during training
- `RENDER_TRAINED_AGENT`: Enable/disable evaluation with visualization

## Reinforcement Learning Approach

The project uses Deep Q-Learning with the following features:
- Experience replay to break correlations in sequential data
- Target network to stabilize learning
- Epsilon-greedy exploration strategy
- Convolutional neural network to process the game state

## Training Tips

Training the agent may require several thousand episodes to achieve good performance. During training:

1. Regularly check the best model in the `checkpoints` directory
2. You can interrupt training at any time and use `visualize_agent.py` to see the current agent's performance
3. If the model is learning too slowly, try adjusting these parameters:
   - Lower `epsilon_decay` for slower exploration decay
   - Increase `learning_rate` for faster learning
   - Modify the neural network architecture

## Notes

- This project requires Python 3.6+
- Ensure TensorFlow and Pygame are correctly installed
- Model filenames must now end with `.weights.h5` for compatibility with Keras 3.x

## License

MIT License

## GitHub Repository

https://github.com/PopSoda2002/SnakeDQN 