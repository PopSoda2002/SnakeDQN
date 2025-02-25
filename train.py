import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from collections import deque
import time
from tqdm import tqdm

from snake_env import SnakeGameEnv, Direction
from dqn_agent import DQNAgent
from snake_renderer import SnakeRenderer

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Hyperparameters
GRID_SIZE = 10
EPISODES = 10000
BATCH_SIZE = 64
TARGET_UPDATE_FREQ = 100
SAVE_FREQ = 500
LOG_FREQ = 100
TRAIN_START = 1000
MAX_STEPS = 2000
RENDER_DURING_TRAINING = False
RENDER_EVERY = 500
RENDER_TRAINED_AGENT = True

# Create the directory for model checkpoints if it doesn't exist
if not os.path.exists('checkpoints'):
    os.makedirs('checkpoints')

def train_agent():
    # Initialize environment and agent
    env = SnakeGameEnv(grid_size=GRID_SIZE, max_step_limit=MAX_STEPS)
    state_shape = (GRID_SIZE, GRID_SIZE, 3)  # 3 channels: snake body, snake head, food
    action_size = 4  # UP, RIGHT, DOWN, LEFT
    
    agent = DQNAgent(
        state_shape=state_shape,
        action_size=action_size,
        learning_rate=0.0005,
        discount_factor=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995
    )
    
    # For rendering during training (optional)
    if RENDER_DURING_TRAINING:
        renderer = SnakeRenderer(env, fps=30)
    
    # Tracking metrics
    scores = []
    avg_scores = deque(maxlen=100)
    max_score = 0
    episode_durations = []
    
    # Training loop
    for episode in tqdm(range(EPISODES), desc="Training"):
        state = env.reset()
        done = False
        episode_reward = 0
        steps = 0
        
        # Render the start of selected episodes
        if RENDER_DURING_TRAINING and episode % RENDER_EVERY == 0:
            renderer.render()
            time.sleep(0.5)
        
        # Episode loop
        while not done:
            # Select action
            action = agent.select_action(state)
            
            # Take action
            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            steps += 1
            
            # Store experience
            agent.remember(state, action, reward, next_state, done)
            
            # Move to next state
            state = next_state
            
            # Render the episode (if enabled)
            if RENDER_DURING_TRAINING and episode % RENDER_EVERY == 0:
                renderer.render()
                time.sleep(0.05)
            
            # Train the agent
            if len(agent.memory) > TRAIN_START:
                agent.train(BATCH_SIZE)
            
            # Update target model periodically
            if steps % TARGET_UPDATE_FREQ == 0 and len(agent.memory) > TRAIN_START:
                agent.update_target_model()
        
        # Record metrics
        score = info['score']
        scores.append(score)
        avg_scores.append(score)
        episode_durations.append(steps)
        
        # Save best model
        if score > max_score:
            max_score = score
            agent.save('checkpoints/best_model.h5')
            print(f"\nNew best score: {max_score} (Episode {episode+1})")
        
        # Save model periodically
        if (episode + 1) % SAVE_FREQ == 0:
            agent.save(f'checkpoints/model_ep{episode+1}.h5')
        
        # Logging
        if (episode + 1) % LOG_FREQ == 0:
            avg_score = np.mean(list(avg_scores))
            print(f"\nEpisode {episode+1}/{EPISODES} - Avg Score: {avg_score:.2f}, Epsilon: {agent.epsilon:.4f}")
    
    # Save final model
    agent.save('checkpoints/final_model.h5')
    
    # Close renderer if used
    if RENDER_DURING_TRAINING:
        renderer.close()
    
    return agent, scores, episode_durations

def plot_training_results(scores, durations):
    """Plot the training progress."""
    plt.figure(figsize=(15, 5))
    
    # Plot scores
    plt.subplot(1, 2, 1)
    plt.plot(scores)
    plt.title('Score per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    
    # Plot running average
    window_size = min(100, len(scores))
    running_avg = np.convolve(scores, np.ones(window_size)/window_size, mode='valid')
    plt.plot(range(window_size-1, len(scores)), running_avg, 'r-')
    plt.legend(['Score', f'Average ({window_size} episodes)'])
    
    # Plot episode durations
    plt.subplot(1, 2, 2)
    plt.plot(durations)
    plt.title('Steps per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    
    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.show()

def evaluate_agent(agent, episodes=5):
    """Evaluate the trained agent with visualization."""
    env = SnakeGameEnv(grid_size=GRID_SIZE)
    renderer = SnakeRenderer(env, fps=10)  # Slower speed for better visualization
    
    scores = []
    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            # Use greedy policy (no exploration)
            agent.epsilon = 0
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            state = next_state
            
            # Render the game
            renderer.render()
            
            if done:
                scores.append(info['score'])
                print(f"Episode {episode+1} - Score: {info['score']}, Total Reward: {total_reward:.2f}")
                time.sleep(1)  # Pause briefly at the end of the episode
    
    renderer.close()
    print(f"Average Score: {np.mean(scores):.2f}")

if __name__ == "__main__":
    # Train the agent
    trained_agent, score_history, duration_history = train_agent()
    
    # Plot training results
    plot_training_results(score_history, duration_history)
    
    # Evaluate the trained agent
    if RENDER_TRAINED_AGENT:
        evaluate_agent(trained_agent) 