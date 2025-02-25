import numpy as np
import random
from enum import Enum

class Direction(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

class SnakeGameEnv:
    def __init__(self, grid_size=10, max_step_limit=100):
        self.grid_size = grid_size
        self.max_step_limit = max_step_limit
        self.reset()
        
    def reset(self):
        """Reset the game state."""
        self.snake = [(self.grid_size // 2, self.grid_size // 2)]
        self.direction = Direction.UP
        self.food = self._place_food()
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.prev_distance_to_food = self._get_distance_to_food()
        return self._get_state()
    
    def _place_food(self):
        """Place food in a random position that is not occupied by the snake."""
        while True:
            food = (random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1))
            if food not in self.snake:
                return food
    
    def _get_state(self):
        """Return the current state representation for the agent."""
        # Create an empty grid
        state = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.float32)
        
        # Mark snake body
        for segment in self.snake:
            state[segment[0], segment[1], 0] = 1
        
        # Mark snake head
        head_x, head_y = self.snake[0]
        state[head_x, head_y, 1] = 1
        
        # Mark food
        state[self.food[0], self.food[1], 2] = 1
        
        return state
    
    def _is_collision(self, position):
        """Check if the position collides with the wall or snake body."""
        x, y = position
        # Check wall collision
        if x < 0 or x >= self.grid_size or y < 0 or y >= self.grid_size:
            return True
        
        # Check self collision (excluding the tail if the snake is moving without eating)
        if position in self.snake[:-1]:
            return True
        
        return False
    
    def _get_distance_to_food(self):
        """Calculate Manhattan distance from snake head to food."""
        head_x, head_y = self.snake[0]
        food_x, food_y = self.food
        return abs(head_x - food_x) + abs(head_y - food_y)
    
    def step(self, action):
        """
        Take a step in the environment.
        Action is an integer: 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT
        Returns: next_state, reward, done, info
        """
        if self.game_over:
            return self._get_state(), 0, True, {"score": self.score}
        
        # Update direction based on action
        # We prevent 180-degree turns
        if action == Direction.UP.value and self.direction != Direction.DOWN:
            self.direction = Direction.UP
        elif action == Direction.RIGHT.value and self.direction != Direction.LEFT:
            self.direction = Direction.RIGHT
        elif action == Direction.DOWN.value and self.direction != Direction.UP:
            self.direction = Direction.DOWN
        elif action == Direction.LEFT.value and self.direction != Direction.RIGHT:
            self.direction = Direction.LEFT
        
        # Move the snake
        head_x, head_y = self.snake[0]
        if self.direction == Direction.UP:
            new_head = (head_x - 1, head_y)
        elif self.direction == Direction.RIGHT:
            new_head = (head_x, head_y + 1)
        elif self.direction == Direction.DOWN:
            new_head = (head_x + 1, head_y)
        else:  # Direction.LEFT
            new_head = (head_x, head_y - 1)
        
        # Check for collision
        self.steps += 1
        
        # Calculate initial reward (will be modified based on outcome)
        reward = 0
        
        # Check if game is over
        if self._is_collision(new_head):
            self.game_over = True
            # Higher penalty for collision the longer the snake
            reward = -10 - self.score * 0.5  
            return self._get_state(), reward, True, {"score": self.score}
        
        # Check for timeout
        if self.steps >= self.max_step_limit:
            self.game_over = True
            reward = -1  # Mild penalty for timeout
            return self._get_state(), reward, True, {"score": self.score}
        
        # Move the snake
        self.snake.insert(0, new_head)
        
        # Check if food is eaten
        if new_head == self.food:
            self.score += 1
            # Reward for eating food (higher reward for longer snakes to encourage growth)
            reward = 10 + self.score * 0.5
            self.food = self._place_food()
            # Don't remove the tail to make the snake grow
            self.prev_distance_to_food = self._get_distance_to_food()
        else:
            self.snake.pop()
            
            # Calculate new distance to food
            current_distance = self._get_distance_to_food()
            
            # Reward for moving closer to food, penalty for moving away
            if current_distance < self.prev_distance_to_food:
                reward = 0.1  # Small positive reward for moving closer
            elif current_distance > self.prev_distance_to_food:
                reward = -0.1  # Small negative reward for moving away
            else:
                reward = -0.05  # Slightly negative reward for not changing distance
                
            self.prev_distance_to_food = current_distance
            
            # Additional penalty for circling/looping behavior
            if self.steps > self.grid_size * 3 and self.score == 0:
                reward -= 0.01 * (self.steps / self.grid_size)  # Increasing penalty for not finding food
        
        state = self._get_state()
        done = self.game_over
        
        return state, reward, done, {"score": self.score}
    
    def render_ascii(self):
        """Render the game state in ASCII for debugging."""
        grid = [['□' for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        
        for segment in self.snake[1:]:
            grid[segment[0]][segment[1]] = '■'
        
        head_x, head_y = self.snake[0]
        grid[head_x][head_y] = '●'
        
        grid[self.food[0]][self.food[1]] = '★'
        
        for row in grid:
            print(' '.join(row))
        print(f"Score: {self.score}") 