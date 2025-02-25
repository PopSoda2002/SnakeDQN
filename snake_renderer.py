import pygame
import time
from snake_env import SnakeGameEnv, Direction

class SnakeRenderer:
    def __init__(self, env, cell_size=30, fps=10):
        self.env = env
        self.cell_size = cell_size
        self.fps = fps
        self.width = env.grid_size * cell_size
        self.height = env.grid_size * cell_size
        
        # Colors
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.GREEN = (0, 255, 0)
        self.RED = (255, 0, 0)
        self.BLUE = (0, 0, 255)
        self.GRAY = (200, 200, 200)
        
        # Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Snake RL')
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 36)
        
    def render(self, mode='human'):
        """Render the current state of the environment."""
        self.screen.fill(self.BLACK)
        
        # Draw grid lines
        for i in range(self.env.grid_size + 1):
            pygame.draw.line(self.screen, self.GRAY, (0, i * self.cell_size), 
                             (self.width, i * self.cell_size), 1)
            pygame.draw.line(self.screen, self.GRAY, (i * self.cell_size, 0), 
                             (i * self.cell_size, self.height), 1)
        
        # Draw snake
        for i, segment in enumerate(self.env.snake):
            color = self.BLUE if i == 0 else self.GREEN  # Head is blue, body is green
            pygame.draw.rect(self.screen, color, 
                             (segment[1] * self.cell_size, segment[0] * self.cell_size, 
                              self.cell_size, self.cell_size))
        
        # Draw food
        pygame.draw.rect(self.screen, self.RED, 
                         (self.env.food[1] * self.cell_size, self.env.food[0] * self.cell_size, 
                          self.cell_size, self.cell_size))
        
        # Draw score
        score_text = self.font.render(f'Score: {self.env.score}', True, self.WHITE)
        self.screen.blit(score_text, (5, 5))
        
        pygame.display.flip()
        self.clock.tick(self.fps)
    
    def close(self):
        """Close the renderer."""
        pygame.quit()
        
    def manual_play(self):
        """Allow manual play of the game."""
        done = False
        state = self.env.reset()
        self.render()
        
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                    break
                
                if event.type == pygame.KEYDOWN:
                    action = None
                    if event.key == pygame.K_UP:
                        action = Direction.UP.value
                    elif event.key == pygame.K_RIGHT:
                        action = Direction.RIGHT.value
                    elif event.key == pygame.K_DOWN:
                        action = Direction.DOWN.value
                    elif event.key == pygame.K_LEFT:
                        action = Direction.LEFT.value
                    
                    if action is not None:
                        state, reward, done, info = self.env.step(action)
                        self.render()
                        if done:
                            print(f"Game Over! Score: {info['score']}")
                            time.sleep(2)
                            break
        
        self.close()

    def agent_play(self, agent, episodes=1, show_game_over=True):
        """Let an agent play the game."""
        for episode in range(episodes):
            state = self.env.reset()
            done = False
            total_reward = 0
            
            while not done:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.close()
                        return
                
                action = agent.select_action(state)
                next_state, reward, done, info = self.env.step(action)
                total_reward += reward
                state = next_state
                
                self.render()
                
                if done:
                    if show_game_over:
                        font = pygame.font.SysFont(None, 72)
                        game_over_text = font.render('Game Over', True, self.WHITE)
                        score_text = font.render(f'Score: {info["score"]}', True, self.WHITE)
                        self.screen.blit(game_over_text, (self.width // 2 - 150, self.height // 2 - 50))
                        self.screen.blit(score_text, (self.width // 2 - 100, self.height // 2 + 20))
                        pygame.display.flip()
                        time.sleep(2)
                    break
            
            print(f"Episode {episode+1} - Score: {self.env.score}, Total Reward: {total_reward:.2f}")
        
        self.close() 