"""
T-Maze Environment

Adapted from belief-rnn-main project.
Maintains original logic while implementing unified interface.
"""

import torch
import random
import numpy as np
from math import ceil

from .base_env import BaseEnvironment


# Observation space (from original project)
OBSERVATIONS = torch.eye(4)
O_UP, O_DOWN, O_CORRIDOR, O_CROSSROAD = OBSERVATIONS

# Action space (from original project)
ACTIONS = (0, 1, 2, 3)
A_RIGHT, A_UP, A_LEFT, A_DOWN = ACTIONS


class TMazeEnvironment(BaseEnvironment):
    """
    A T-Maze environment (from original belief-rnn project).
    
    ```
                    |?| -> L + 1
        |0|.|.|.|.|.|L|
                    |?| -> L + 2
    ```
    """

    def __init__(self, length=10, stochasticity=0.0, irrelevant_features=0, bayes=False):
        self.gamma = 0.98
        self.observation_size = 4 + irrelevant_features
        self.action_size = 4
        self.belief_type = "exact"
        
        super().__init__(self.observation_size, self.action_size)
        
        self.length = int(length)
        self.stochasticity = float(stochasticity)
        self.irrelevant_features = irrelevant_features
        self.bayes = bayes
        
        # Initialize transition and observation models (from original)
        self.T = self._transition_model()
        self.O = self._observation_model()
        
        # States
        self.position = None
        self.goal_up = None
        self.last_position = None
        self.done = False
        self.step_count = 0
        self.max_steps = 2 * length + 10
        
        # Belief state tracking
        self.belief = None
        
    def reset(self):
        """Reset environment to initial state"""
        self.position = 0
        self.last_position = -1
        self.goal_up = random.choice([True, False])
        self.done = False
        self.step_count = 0
        
        # Return initial observation
        observation = self._get_observation()
        
        # Initialize belief if using Bayes
        if self.bayes:
            self._init_belief(observation)
        else:
            self.belief = None
            
        return observation
    
    def step(self, action):
        """Take one step in the environment"""
        if self.done:
            return self._get_observation(), 0.0, True, {}
            
        self._check_action(action)
        self.step_count += 1
        
        # Store last position
        self.last_position = self.position
        
        # Execute transition (from original logic)
        self._transition(action)
        observation = self._get_observation()
        reward = self._reward(action)
        done = self._terminal()
        
        # Update belief if using Bayes
        if self.bayes:
            self._update_belief(action, observation)
        
        # Check if max steps reached
        if self.step_count >= self.max_steps:
            done = True
            
        self.done = done
        info = {
            'position': self.position,
            'goal_up': self.goal_up,
            'belief': self.belief.copy() if self.belief is not None else None
        }
        
        return observation, reward, self.done, info
    
    def _get_observation(self):
        """Get current observation (from original logic)"""
        if self.position == 0:
            # Show goal hint at the beginning
            if self.goal_up:
                obs = O_UP
            else:
                obs = O_DOWN
        elif 0 < self.position < self.length:
            # In corridor
            obs = O_CORRIDOR
        elif self.length <= self.position <= self.length + 2:
            # At T-junction or terminal
            obs = O_CROSSROAD
        else:
            raise ValueError("Unexpected state")
            
        # Convert to numpy for compatibility
        obs_np = obs.numpy()
            
        # Add irrelevant features if specified
        if self.irrelevant_features > 0:
            irrelevant = np.random.random(self.irrelevant_features)
            obs_np = np.concatenate([obs_np, irrelevant])
            
        return obs_np
    
    def get_observation_space(self):
        """Get observation space size"""
        return self.observation_size
    
    def get_action_space(self):
        """Get action space size"""
        return self.action_size
    
    def horizon(self):
        """Expected horizon for this environment"""
        return self.max_steps
    
    def exploration(self):
        """Get random exploration action"""
        return random.choice(ACTIONS)
    
    def update_belief(self, observation, action):
        """Update belief state (for belief tracking experiments)"""
        # This is a simplified belief update
        # In practice, this would use the observation model
        pass
    
    def _check_action(self, action):
        """Check if action is valid (from original)"""
        if action < 0 or self.action_size <= action:
            size = self.action_size
            raise ValueError(f"The action should be in range [0, {size}[")
    
    def _terminal(self, last=False):
        """Returns True if the current state is terminal (from original)"""
        position = self.last_position if last else self.position
        return self.length + 1 <= position <= self.length + 2
    
    def _transition(self, action):
        """Transitions to a new state using the action (from original)"""
        # Terminal states are never updated
        if self._terminal():
            return
        
        # Apply stochasticity
        if random.random() < self.stochasticity:
            action = random.choice(ACTIONS)
        
        # Transition from the corridor
        if 0 < self.position < self.length:
            if action == A_RIGHT:
                self.position += 1
            elif action == A_LEFT:
                self.position -= 1
        
        # Transition from the first cell
        elif self.position == 0:
            if action == A_RIGHT:
                self.position += 1
        
        # Transition from the crossroad
        elif self.position == self.length:
            if action == A_LEFT:
                self.position = self.length - 1
            elif action == A_UP:
                self.position = self.length + 1
            elif action == A_DOWN:
                self.position = self.length + 2
        
        # Unexpected
        else:
            raise ValueError("Unexpected state")
    
    def _reward(self, action):
        """Returns the reward resulting from the action (from original)"""
        # If the previous state was terminal, then the reward is zero.
        if self._terminal(last=True):
            return 0.0
        
        # Otherwise, if the agent hasn't moved, it has bumped onto a wall.
        elif self.last_position == self.position:
            return -0.1
        
        # Otherwise, if it still is in the corridor or at the crossroad, the reward is 0.0
        elif 0 <= self.position <= self.length:
            return 0.0
        
        # Finally, if it reaches a terminal state
        elif self.length + 1 <= self.position <= self.length + 2:
            if self.goal_up and self.position == self.length + 1:
                return 4.0
            elif not self.goal_up and self.position == self.length + 2:
                return 4.0
            else:
                return -0.1
        
        # Unexpected
        else:
            raise ValueError("Unexpected state")
    
    def _observation_model(self):
        """Returns the observation model O (from original)"""
        O = {}
        
        for o in range(len(OBSERVATIONS)):
            O[o] = torch.zeros(2 * (self.length + 3))
        
        for i in range(2 * (self.length + 3)):
            goal_up = 1 - int(i / (self.length + 3))
            position = i % (self.length + 3)
            
            if 0 < position < self.length:
                O[O_CORRIDOR.argmax().item()][i] = 1.0
            elif position == 0:
                if goal_up:
                    O[O_UP.argmax().item()][i] = 1.0
                else:
                    O[O_DOWN.argmax().item()][i] = 1.0
            elif position == self.length:
                O[O_CROSSROAD.argmax().item()][i] = 1.0
            elif self.length + 1 <= position <= self.length + 2:
                O[O_CROSSROAD.argmax().item()][i] = 1.0
        
        return O
    
    def _transition_model(self):
        """Returns the transition model T (from original)"""
        T = {}
        
        for ACTION in ACTIONS:
            T[ACTION] = torch.zeros(
                2 * (self.length + 3),
                2 * (self.length + 3),
            )
        
        for i in range(2 * (self.length + 3)):
            position = i % (self.length + 3)
            
            if 0 < position < self.length:
                T[A_RIGHT][i - 1, i] = self.stochasticity / 4
                T[A_RIGHT][i, i] = 2 * self.stochasticity / 4
                T[A_RIGHT][i + 1, i] = 1 - 3 * self.stochasticity / 4
                
                T[A_UP][i - 1, i] = self.stochasticity / 4
                T[A_UP][i, i] = 1 - 2 * self.stochasticity / 4
                T[A_UP][i + 1, i] = self.stochasticity / 4
                
                T[A_LEFT][i - 1, i] = 1 - 3 * self.stochasticity / 4
                T[A_LEFT][i, i] = 2 * self.stochasticity / 4
                T[A_LEFT][i + 1, i] = self.stochasticity / 4
                
                T[A_DOWN][i - 1, i] = self.stochasticity / 4
                T[A_DOWN][i, i] = 1 - 2 * self.stochasticity / 4
                T[A_DOWN][i + 1, i] = self.stochasticity / 4
                
            elif position == 0:
                T[A_RIGHT][i + 1, i] = 1 - 3 * self.stochasticity / 4
                T[A_RIGHT][i, i] = 3 * self.stochasticity / 4
                
                for A_OTHER in (A_UP, A_LEFT, A_DOWN):
                    T[A_OTHER][i, i] = 1 - 3 * self.stochasticity / 4
                    T[A_OTHER][i + 1, i] = self.stochasticity / 4
                    
            elif position == self.length:
                T[A_RIGHT][i - 1, i] = self.stochasticity / 4
                T[A_RIGHT][i, i] = 1 - 3 * self.stochasticity / 4
                T[A_RIGHT][i + 1, i] = self.stochasticity / 4
                T[A_RIGHT][i + 2, i] = self.stochasticity / 4
                
                T[A_UP][i - 1, i] = self.stochasticity / 4
                T[A_UP][i, i] = self.stochasticity / 4
                T[A_UP][i + 1, i] = 1 - 3 * self.stochasticity / 4
                T[A_UP][i + 2, i] = self.stochasticity / 4
                
                T[A_LEFT][i - 1, i] = 1 - 3 * self.stochasticity / 4
                T[A_LEFT][i, i] = self.stochasticity / 4
                T[A_LEFT][i + 1, i] = self.stochasticity / 4
                T[A_LEFT][i + 2, i] = self.stochasticity / 4
                
                T[A_DOWN][i - 1, i] = self.stochasticity / 4
                T[A_DOWN][i, i] = self.stochasticity / 4
                T[A_DOWN][i + 1, i] = self.stochasticity / 4
                T[A_DOWN][i + 2, i] = 1 - 3 * self.stochasticity / 4
                
            elif self.length + 1 <= position <= self.length + 2:
                for ACTION in ACTIONS:
                    T[ACTION][i, i] = 1.0
        
        return T
    
    def _init_belief(self, observation):
        """Initialises the belief b_0 (from original)"""
        self.belief = torch.zeros(2 * (self.length + 3))
        self.belief[0] = 0.5
        self.belief[self.length + 3] = 0.5
        
        O = self.O[observation.argmax().item()]
        self.belief = O * self.belief
        self.belief /= self.belief.sum()
    
    def _update_belief(self, action, observation):
        """Updates the belief (from original)"""
        T = self.T[action]
        O = self.O[observation.argmax().item()]
        self.belief = O * (T @ self.belief)
        self.belief /= self.belief.sum()
    
    def get_belief(self):
        """Returns the current belief (from original)"""
        return (self.belief.clone(),) if self.belief is not None else None

    def render(self):
        """Render the current state"""
        print(f"Position: {self.position}/{self.length}, Goal Up: {self.goal_up}, Done: {self.done}")
        if self.belief is not None:
            print(f"Belief: {self.belief}")


class IrrelevantWrapper:
    """
    Wrapper to add irrelevant features to observations
    (from original project)
    """
    
    def __init__(self, environment, state_size=2):
        self.environment = environment
        self.state_size = state_size
        self.observation_size = environment.observation_size + state_size
        self.action_size = environment.action_size
        
    def reset(self):
        obs = self.environment.reset()
        irrelevant = np.random.random(self.state_size)
        return np.concatenate([obs, irrelevant])
    
    def step(self, action):
        obs, reward, done, info = self.environment.step(action)
        irrelevant = np.random.random(self.state_size)
        obs = np.concatenate([obs, irrelevant])
        return obs, reward, done, info
    
    def get_observation_space(self):
        return self.observation_size
    
    def get_action_space(self):
        return self.action_size
    
    def __getattr__(self, name):
        """Delegate other attributes to wrapped environment"""
        return getattr(self.environment, name)
