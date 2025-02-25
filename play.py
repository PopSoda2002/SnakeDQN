"""
Simple script to play the Snake game manually.
This is useful for testing the environment before training an agent.
"""

from snake_env import SnakeGameEnv
from snake_renderer import SnakeRenderer

def main():
    # Create the environment and renderer
    env = SnakeGameEnv(grid_size=10)  # You can adjust the grid size
    renderer = SnakeRenderer(env, cell_size=30, fps=8)  # Adjust cell size and fps as needed
    
    print("Starting manual play mode...")
    print("Use arrow keys to control the snake.")
    print("Press Q or close the window to quit.")
    
    # Start manual play
    renderer.manual_play()

if __name__ == "__main__":
    main() 