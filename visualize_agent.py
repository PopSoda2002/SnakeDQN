"""
Visualize a trained Snake agent
"""

import argparse
import numpy as np
import pygame
import time

from snake_env import SnakeGameEnv
from dqn_agent import DQNAgent
from snake_renderer import SnakeRenderer

def parse_arguments():
    parser = argparse.ArgumentParser(description='Visualize a trained Snake agent')
    parser.add_argument('--model_path', type=str, default='checkpoints/best_model.h5', 
                        help='Path to the trained model weights')
    parser.add_argument('--grid_size', type=int, default=10, 
                        help='Size of the game grid')
    parser.add_argument('--cell_size', type=int, default=40, 
                        help='Size of each cell in pixels')
    parser.add_argument('--fps', type=int, default=8, 
                        help='Frames per second (game speed)')
    parser.add_argument('--episodes', type=int, default=5, 
                        help='Number of episodes to run')
    return parser.parse_args()

def visualize_agent(model_path, grid_size=10, cell_size=40, fps=10, episodes=5):
    # Set up the environment
    env = SnakeGameEnv(grid_size=grid_size)
    
    # Set up the agent
    state_shape = (grid_size, grid_size, 3)
    action_size = 4
    agent = DQNAgent(state_shape=state_shape, action_size=action_size)
    
    # Load the trained weights
    try:
        agent.load(model_path)
        print(f"Successfully loaded model from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Set up the renderer
    renderer = SnakeRenderer(env, cell_size=cell_size, fps=fps)
    
    # Track statistics
    scores = []
    steps_list = []
    
    # Run episodes
    for episode in range(episodes):
        state = env.reset()
        done = False
        steps = 0
        
        # Set epsilon to 0 for evaluation (no exploration)
        agent.epsilon = 0
        
        while not done:
            # Check for exit events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    renderer.close()
                    return
            
            # Get the action from the agent
            action = agent.select_action(state)
            
            # Take the action
            next_state, reward, done, info = env.step(action)
            steps += 1
            
            # Update the state
            state = next_state
            
            # Render the game
            renderer.render()
            
            # Sleep to make it viewable
            time.sleep(1/fps)
            
            # Check if episode is done
            if done:
                scores.append(info['score'])
                steps_list.append(steps)
                print(f"Episode {episode+1} - Score: {info['score']}, Steps: {steps}")
                
                # Display game over message
                font = pygame.font.SysFont(None, 72)
                game_over_text = font.render('Game Over', True, renderer.WHITE)
                score_text = font.render(f'Score: {info["score"]}', True, renderer.WHITE)
                renderer.screen.blit(game_over_text, (renderer.width // 2 - 150, renderer.height // 2 - 50))
                renderer.screen.blit(score_text, (renderer.width // 2 - 100, renderer.height // 2 + 20))
                pygame.display.flip()
                time.sleep(2)
    
    # Print summary statistics
    if scores:
        print("\nResults Summary:")
        print(f"Average Score: {np.mean(scores):.2f}")
        print(f"Max Score: {np.max(scores)}")
        print(f"Average Steps: {np.mean(steps_list):.2f}")
    
    # Close the renderer
    renderer.close()

if __name__ == "__main__":
    args = parse_arguments()
    visualize_agent(
        model_path=args.model_path,
        grid_size=args.grid_size,
        cell_size=args.cell_size,
        fps=args.fps,
        episodes=args.episodes
    ) 