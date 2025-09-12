"""
Unified RL Framework - Environments Module

This module provides unified interfaces for both T-maze and MDP task environments
while preserving the original logic from both projects.
"""

from .tmaze_env import TMazeEnvironment
from .mdp_env import MDPEnvironment, TaskGenerator

__all__ = ['TMazeEnvironment', 'MDPEnvironment', 'TaskGenerator']
