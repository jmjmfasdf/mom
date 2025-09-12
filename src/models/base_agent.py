"""
Base Agent Class for Unified RL Framework

Provides common interface for both GRU and DRQN agents.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class BaseAgent(ABC):
    """
    Abstract base class for all RL agents in the unified framework.
    """
    
    def __init__(self, observation_size, action_size, hidden_size=128, num_layers=1):
        self.observation_size = observation_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
    @abstractmethod
    def get_action(self, observation, hidden_state=None):
        """
        Get action from observation.
        
        Args:
            observation: Current observation
            hidden_state: Hidden state (if applicable)
            
        Returns:
            action: Selected action
            hidden_state: Updated hidden state
        """
        pass
    
    @abstractmethod
    def train_step(self, batch_data):
        """
        Perform one training step.
        
        Args:
            batch_data: Training batch data
            
        Returns:
            loss: Training loss
        """
        pass
    
    @abstractmethod
    def reset_hidden(self, batch_size=1):
        """
        Reset hidden state.
        
        Args:
            batch_size: Batch size
            
        Returns:
            hidden_state: Initial hidden state
        """
        pass
    
    @abstractmethod
    def save_model(self, path):
        """Save model to path."""
        pass
    
    @abstractmethod
    def load_model(self, path):
        """Load model from path."""
        pass
