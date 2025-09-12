"""
Base Environment Class for Unified RL Framework

Provides common interface for all environments.
"""

import numpy as np
from abc import ABC, abstractmethod


class BaseEnvironment(ABC):
    """
    Abstract base class for all environments in the unified framework.
    """
    
    def __init__(self, observation_size, action_size):
        self.observation_size = observation_size
        self.action_size = action_size
        
    @abstractmethod
    def reset(self):
        """
        Reset environment to initial state.
        
        Returns:
            observation: Initial observation
        """
        pass
    
    @abstractmethod
    def step(self, action):
        """
        Take one step in the environment.
        
        Args:
            action: Action to take
            
        Returns:
            observation: Next observation
            reward: Reward received
            done: Whether episode is finished
            info: Additional information
        """
        pass
    
    @abstractmethod
    def get_observation_space(self):
        """Get observation space size."""
        pass
    
    @abstractmethod
    def get_action_space(self):
        """Get action space size."""
        pass
    
    def render(self):
        """Render the environment (optional)."""
        pass
