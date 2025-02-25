import tensorflow as tf
import numpy as np
import random
from collections import deque
from snake_env import Direction

class DQNAgent:
    def __init__(self, state_shape, action_size, learning_rate=0.001, 
                 discount_factor=0.95, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.state_shape = state_shape  # Shape of the state observation
        self.action_size = action_size  # Number of possible actions
        self.memory = deque(maxlen=10000)  # Experience replay buffer
        self.learning_rate = learning_rate  # Learning rate for the optimizer
        self.gamma = discount_factor  # Discount factor for future rewards
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_min = epsilon_min  # Minimum exploration rate
        self.epsilon_decay = epsilon_decay  # Decay rate for exploration
        
        # Create main and target networks
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
    
    def _build_model(self):
        """Build a CNN model for the DQN."""
        model = tf.keras.Sequential([
            # Convolutional layers to process the grid
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', 
                                  input_shape=self.state_shape, padding='same'),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.Flatten(),
            
            # Fully connected layers
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model
    
    def update_target_model(self):
        """Update the target model with weights from the main model."""
        self.target_model.set_weights(self.model.get_weights())
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory."""
        self.memory.append((state, action, reward, next_state, done))
    
    def select_action(self, state):
        """Select an action using epsilon-greedy policy."""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        # Reshape state for model input
        state = np.expand_dims(state, axis=0)
        q_values = self.model.predict(state, verbose=0)[0]
        return np.argmax(q_values)
    
    def train(self, batch_size=64):
        """Train the model using experience replay."""
        if len(self.memory) < batch_size:
            return
        
        # Sample mini-batch from memory
        minibatch = random.sample(self.memory, batch_size)
        
        # Extract data from mini-batch
        states = np.zeros((batch_size,) + self.state_shape)
        next_states = np.zeros((batch_size,) + self.state_shape)
        actions, rewards, dones = [], [], []
        
        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            states[i] = state
            next_states[i] = next_state
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
        
        # Predict Q-values for current and next states
        targets = self.model.predict(states, verbose=0)
        next_q_values = self.target_model.predict(next_states, verbose=0)
        
        # Update target Q-values with Bellman equation
        for i in range(batch_size):
            if dones[i]:
                targets[i, actions[i]] = rewards[i]
            else:
                targets[i, actions[i]] = rewards[i] + self.gamma * np.max(next_q_values[i])
        
        # Train the model
        self.model.fit(states, targets, epochs=1, verbose=0, batch_size=batch_size)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save(self, filename):
        """Save the model weights."""
        # Make sure filename ends with .weights.h5 as required by Keras 3.x
        if not filename.endswith('.weights.h5'):
            # Replace .h5 with .weights.h5 if it exists
            if filename.endswith('.h5'):
                filename = filename.replace('.h5', '.weights.h5')
            else:
                # Otherwise just append .weights.h5
                filename = filename + '.weights.h5'
        self.model.save_weights(filename)
    
    def load(self, filename):
        """Load model weights."""
        # Make sure filename ends with .weights.h5 as required by Keras 3.x
        if not filename.endswith('.weights.h5'):
            # Replace .h5 with .weights.h5 if it exists
            if filename.endswith('.h5'):
                filename = filename.replace('.h5', '.weights.h5')
            else:
                # Otherwise just append .weights.h5
                filename = filename + '.weights.h5'
        self.model.load_weights(filename) 