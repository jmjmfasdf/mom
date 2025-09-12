"""
Unified RL Framework - Models Module

This module provides unified interfaces for both GRU and DRQN models
while preserving the original logic from both projects.
"""

from .gru_model import GRUAgent
from .drqn_model import DRQNAgent

__all__ = ['GRUAgent', 'DRQNAgent']
