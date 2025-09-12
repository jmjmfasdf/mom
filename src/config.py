"""
Configuration System for Unified RL Framework

Manages hyperparameters and settings for different model-environment combinations.
"""

import argparse
from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class ModelConfig:
    """Configuration for models"""
    hidden_size: int = 128
    num_layers: int = 2
    learning_rate: float = 1e-3
    batch_size: int = 32


@dataclass
class GRUConfig(ModelConfig):
    """GRU-specific configuration"""
    hidden_size: int = 128
    num_layers: int = 1
    learning_rate: float = 1e-3
    batch_size: int = 128
    clip_grad: float = 0.25


@dataclass
class DRQNConfig(ModelConfig):
    """DRQN-specific configuration"""
    hidden_size: int = 32
    num_layers: int = 2
    learning_rate: float = 1e-3
    batch_size: int = 32
    epsilon: float = 0.2
    gamma: float = 0.98
    buffer_capacity: int = 8192
    target_update_freq: int = 10
    cell_type: str = 'gru'


@dataclass
class EnvironmentConfig:
    """Configuration for environments"""
    name: str = ""


@dataclass
class TMazeConfig(EnvironmentConfig):
    """T-Maze specific configuration"""
    name: str = "tmaze"
    length: int = 20
    stochasticity: float = 0.0
    irrelevant_features: int = 0


@dataclass
class MDPConfig(EnvironmentConfig):
    """Two-step task specific configuration"""
    name: str = "mdp"
    s1_duration: int = 1  # Stage 1 stimulus duration (fixed to 1 for 5-slot structure)
    s2_duration: int = 1  # Stage 2 stimulus duration (fixed to 1 for 5-slot structure)


@dataclass
class TrainingConfig:
    """Training configuration"""
    num_episodes: int = 1000
    eval_period: int = 100
    eval_every: int = 50  # Evaluation frequency (default: every 50 episodes)
    num_rollouts: int = 50
    save_period: int = 1000
    save_every: int = 0  # Model saving frequency (0=only at end, >0=every N episodes)
    log_dir: str = "./logs"
    save_dir: str = "./models"
    seed: int = 42
    cuda: bool = False


class ConfigManager:
    """Manages configurations for different combinations"""
    
    def __init__(self):
        self.model_configs = {
            'gru': GRUConfig(),
            'drqn': DRQNConfig()
        }
        
        self.env_configs = {
            'tmaze': TMazeConfig(),
            'mdp': MDPConfig()
        }
        
        self.training_config = TrainingConfig()
        
    def get_model_config(self, model_name: str) -> ModelConfig:
        """Get model configuration"""
        if model_name not in self.model_configs:
            raise ValueError(f"Unknown model: {model_name}")
        return self.model_configs[model_name]
    
    def get_env_config(self, env_name: str) -> EnvironmentConfig:
        """Get environment configuration"""
        if env_name not in self.env_configs:
            raise ValueError(f"Unknown environment: {env_name}")
        return self.env_configs[env_name]
    
    def get_training_config(self) -> TrainingConfig:
        """Get training configuration"""
        return self.training_config
    
    def update_from_args(self, args):
        """Update configurations from command line arguments"""
        # Update model config
        model_config = self.get_model_config(args.model)
        if hasattr(args, 'hidden_size') and args.hidden_size is not None:
            model_config.hidden_size = args.hidden_size
        if hasattr(args, 'num_layers') and args.num_layers is not None:
            model_config.num_layers = args.num_layers
        if hasattr(args, 'learning_rate') and args.learning_rate is not None:
            model_config.learning_rate = args.learning_rate
        if hasattr(args, 'batch_size') and args.batch_size is not None:
            model_config.batch_size = args.batch_size
            
        # DRQN specific
        if args.model == 'drqn':
            if hasattr(args, 'epsilon') and args.epsilon is not None:
                model_config.epsilon = args.epsilon
            if hasattr(args, 'gamma') and args.gamma is not None:
                model_config.gamma = args.gamma
            if hasattr(args, 'buffer_capacity') and args.buffer_capacity is not None:
                model_config.buffer_capacity = args.buffer_capacity
            if hasattr(args, 'cell_type') and args.cell_type is not None:
                model_config.cell_type = args.cell_type
                
        # Update environment config
        env_config = self.get_env_config(args.environment)
        if args.environment == 'tmaze':
            if hasattr(args, 'length') and args.length is not None:
                env_config.length = args.length
            if hasattr(args, 'stochasticity') and args.stochasticity is not None:
                env_config.stochasticity = args.stochasticity
            if hasattr(args, 'irrelevant') and args.irrelevant is not None:
                env_config.irrelevant_features = args.irrelevant
        elif args.environment == 'mdp':
            if hasattr(args, 'task') and args.task is not None:
                env_config.task_name = args.task
                
        # Update training config
        if hasattr(args, 'num_episodes') and args.num_episodes is not None:
            self.training_config.num_episodes = args.num_episodes
        if hasattr(args, 'eval_period') and args.eval_period is not None:
            self.training_config.eval_period = args.eval_period
        if hasattr(args, 'eval_every') and args.eval_every is not None:
            self.training_config.eval_every = args.eval_every
        if hasattr(args, 'save_every') and args.save_every is not None:
            self.training_config.save_every = args.save_every
        if hasattr(args, 'num_rollouts') and args.num_rollouts is not None:
            self.training_config.num_rollouts = args.num_rollouts
        if hasattr(args, 'seed') and args.seed is not None:
            self.training_config.seed = args.seed
        if hasattr(args, 'cuda') and args.cuda is not None:
            self.training_config.cuda = args.cuda


def create_parser():
    """Create argument parser for the unified framework"""
    parser = argparse.ArgumentParser(
        description='Unified RL Framework - Train GRU or DRQN on T-maze or MDP tasks'
    )
    
    # Model selection
    parser.add_argument('--model', type=str, choices=['gru', 'drqn'], required=True,
                       help='Model type to use (gru or drqn)')
    
    # Environment selection
    parser.add_argument('--environment', type=str, choices=['tmaze', 'mdp'], required=True,
                       help='Environment type to use (tmaze or mdp)')
    
    # Model hyperparameters
    parser.add_argument('--hidden-size', type=int, help='Hidden layer size')
    parser.add_argument('--num-layers', type=int, help='Number of RNN layers')
    parser.add_argument('--learning-rate', type=float, help='Learning rate')
    parser.add_argument('--batch-size', type=int, help='Batch size')
    
    # DRQN specific
    parser.add_argument('--epsilon', type=float, help='Epsilon for epsilon-greedy')
    parser.add_argument('--gamma', type=float, help='Discount factor')
    parser.add_argument('--buffer-capacity', type=int, help='Replay buffer capacity')
    parser.add_argument('--cell-type', type=str, choices=['gru', 'lstm'], help='RNN cell type for DRQN')
    
    # T-maze specific
    parser.add_argument('--length', type=int, help='T-maze corridor length')
    parser.add_argument('--stochasticity', type=float, help='T-maze transition stochasticity')
    parser.add_argument('--irrelevant', type=int, help='Number of irrelevant features')
    
    # Two-step task parameters
    parser.add_argument('--s1-duration', type=int, help='Stage 1 stimulus duration (default: 3)')
    parser.add_argument('--s2-duration', type=int, help='Stage 2 stimulus duration (default: 3)')
    parser.add_argument('--trans-prob', type=float, help='Transition probability for two-step task (default: 0.8)')
    parser.add_argument('--reward-prob', type=float, help='Reward probability for two-step task (default: 0.8)')
    
    # Training parameters
    parser.add_argument('--num-episodes', type=int, help='Number of training episodes')
    parser.add_argument('--eval-period', type=int, help='Evaluation period')
    parser.add_argument('--eval-every', type=int, default=50, help='Evaluation frequency (default: 50)')
    parser.add_argument('--save-every', type=int, default=0, help='Model saving frequency (0=only at end, >0=every N episodes)')
    parser.add_argument('--num-rollouts', type=int, help='Number of evaluation rollouts')
    parser.add_argument('--seed', type=int, help='Random seed')
    parser.add_argument('--cuda', action='store_true', help='Use CUDA if available')
    
    # Logging and saving
    parser.add_argument('--log-dir', type=str, default='./logs', help='Log directory')
    parser.add_argument('--save-dir', type=str, default='./models', help='Model save directory')
    parser.add_argument('--name', type=str, help='Experiment name')
    
    # Evaluation mode
    parser.add_argument('--preload', type=str, help='Path to model checkpoint for evaluation mode (e.g., logs/test_save/test_save_final.pt)')
    parser.add_argument('--eval-num-episodes', type=int, default=100, help='Number of episodes for evaluation mode (default: 100)')
    
    return parser


def get_default_config(model_name: str, env_name: str) -> Dict[str, Any]:
    """Get default configuration for model-environment combination"""
    config_manager = ConfigManager()
    
    model_config = config_manager.get_model_config(model_name)
    env_config = config_manager.get_env_config(env_name)
    training_config = config_manager.get_training_config()
    
    return {
        'model': model_config,
        'environment': env_config,
        'training': training_config
    }
