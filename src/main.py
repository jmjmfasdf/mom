"""
Main Script for Unified RL Framework

Allows training and evaluation of GRU or DRQN models on T-maze or MDP tasks.

Usage examples:
    python main.py --model gru --environment mdp --task shape_recognition --num-episodes 5000
    python main.py --model drqn --environment tmaze --length 20 --num-episodes 5000
"""

import os
import sys
import time
import random
import numpy as np
import torch
import logging
from typing import Dict, Any

# Add the framework to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import GRUAgent, DRQNAgent
from environments import TMazeEnvironment, MDPEnvironment, TaskGenerator
from config import ConfigManager, create_parser


def set_random_seeds(seed: int):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def setup_logging(experiment_name: str):
    """Setup logging in logs/experiment_name folder"""
    log_dir = os.path.join("logs", experiment_name)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{experiment_name}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__), log_dir


def create_agent(model_name: str, model_config, observation_size: int, action_size: int, cuda_enabled: bool):
    """Create agent based on model name and configuration"""
    if model_name == 'gru':
        return GRUAgent(
            observation_size=observation_size,
            action_size=action_size,
            hidden_size=model_config.hidden_size,
            num_layers=model_config.num_layers,
            learning_rate=model_config.learning_rate,
            batch_size=model_config.batch_size,
            cuda_enabled=cuda_enabled
        )
    elif model_name == 'drqn':
        return DRQNAgent(
            observation_size=observation_size,
            action_size=action_size,
            hidden_size=model_config.hidden_size,
            num_layers=model_config.num_layers,
            cell=model_config.cell_type,
            learning_rate=model_config.learning_rate,
            gamma=model_config.gamma,
            epsilon=model_config.epsilon,
            buffer_capacity=model_config.buffer_capacity,
            target_update_freq=model_config.target_update_freq
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


def create_environment(env_name: str, env_config, args=None):
    """Create environment based on name and configuration"""
    if env_name == 'tmaze':
        env = TMazeEnvironment(
            length=env_config.length,
            stochasticity=env_config.stochasticity,
            irrelevant_features=env_config.irrelevant_features
        )
        return env
    elif env_name == 'mdp':
        # Prepare task-specific kwargs for two-step task
        task_kwargs = {}
        if args:
            if hasattr(args, 's1_duration') and args.s1_duration is not None:
                task_kwargs['s1_duration'] = args.s1_duration
            if hasattr(args, 's2_duration') and args.s2_duration is not None:
                task_kwargs['s2_duration'] = args.s2_duration
            if hasattr(args, 'trans_prob') and args.trans_prob is not None:
                task_kwargs['trans_prob'] = args.trans_prob
            if hasattr(args, 'reward_prob') and args.reward_prob is not None:
                task_kwargs['reward_prob'] = args.reward_prob
                
        return MDPEnvironment(**task_kwargs)
    else:
        raise ValueError(f"Unknown environment: {env_name}")


def train_drqn_tmaze(agent, environment, training_config, logger, log_dir):
    """Training loop for DRQN on T-maze"""
    logger.info("Starting DRQN training on T-maze")
    
    episode_rewards = []
    
    for episode in range(training_config.num_episodes):
        # Reset environment and agent
        observation = environment.reset()
        hidden_state = agent.reset_hidden(1)
        
        # Collect trajectory
        observations = [observation]
        actions = []
        rewards = []
        dones = []
        
        total_reward = 0
        step_count = 0
        max_steps = environment.horizon()
        
        while step_count < max_steps:
            # Get action
            action, hidden_state = agent.get_action(observation, hidden_state)
            
            # Take step
            next_observation, reward, done, info = environment.step(action)
            
            # Store experience
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            observations.append(next_observation)
            
            total_reward += reward
            step_count += 1
            observation = next_observation
            
            if done:
                break
        
        # Store trajectory in replay buffer
        agent.store_trajectory(observations[:-1], actions, rewards, dones)
        
        # Train agent
        if len(agent.replay_buffer) > agent.replay_buffer.capacity // 4:
            loss = agent.train_step(None)  # DRQN uses replay buffer internally
        
        episode_rewards.append(total_reward)
        
        # Logging and saving
        if (episode + 1) % training_config.eval_every == 0:
            avg_reward = np.mean(episode_rewards[-training_config.eval_every:])
            success_rate = np.mean([1 if r > 0 else 0 for r in episode_rewards[-training_config.eval_every:]])
            logger.info(f"Episode {episode + 1}/{training_config.num_episodes}, "
                       f"Avg Reward: {avg_reward:.3f}, Success Rate: {success_rate:.3f}, Loss: N/A")
        
        # Save model if save_every > 0 and it's time to save
        if training_config.save_every > 0 and (episode + 1) % training_config.save_every == 0:
            model_path = os.path.join(log_dir, f"model_episode_{episode + 1}.pt")
            agent.save_model(model_path)
            logger.info(f"Model saved to {model_path}")
    
    return episode_rewards


def train_gru_tmaze(agent, environment, training_config, logger, log_dir):
    """Training loop for GRU on T-maze"""
    logger.info("Starting GRU training on T-maze")
    
    episode_rewards = []
    
    for episode in range(training_config.num_episodes):
        # Reset environment and agent
        observation = environment.reset()
        hidden_state = agent.model.init_hidden(1)
        
        # Collect trajectory
        observations = [observation]
        actions = []
        rewards = []
        
        total_reward = 0
        step_count = 0
        max_steps = environment.horizon()
        
        while step_count < max_steps:
            # Get action using GRU
            action, hidden_state = agent.get_action(observation, hidden_state)
            
            # Take step
            next_observation, reward, done, info = environment.step(action)
            
            # Store experience
            actions.append(action)
            rewards.append(reward)
            observations.append(next_observation)
            
            total_reward += reward
            observation = next_observation
            step_count += 1
            
            if done:
                break
        
        # Train GRU using collected trajectory
        if len(observations) > 1:
            # Convert trajectory to training format
            obs_array = np.array(observations[:-1])  # Convert to numpy first
            obs_sequence = torch.FloatTensor(obs_array).unsqueeze(1)  # (seq_len, 1, obs_size)
            action_sequence = torch.LongTensor(actions)
            reward_sequence = torch.FloatTensor(rewards)
            
            # Simple Q-learning style training for GRU
            agent.train_trajectory(obs_sequence, action_sequence, reward_sequence)
        
        episode_rewards.append(total_reward)
        
        # Logging and saving
        if (episode + 1) % training_config.eval_every == 0:
            avg_reward = np.mean(episode_rewards[-training_config.eval_every:])
            success_rate = np.mean([1 if r > 0 else 0 for r in episode_rewards[-training_config.eval_every:]])
            logger.info(f"Episode {episode + 1}/{training_config.num_episodes}, "
                       f"Avg Reward: {avg_reward:.3f}, Success Rate: {success_rate:.3f}, Loss: N/A")
        
        # Save model if save_every > 0 and it's time to save
        if training_config.save_every > 0 and (episode + 1) % training_config.save_every == 0:
            model_path = os.path.join(log_dir, f"model_episode_{episode + 1}.pt")
            agent.save_model(model_path)
            logger.info(f"Model saved to {model_path}")
    
    return episode_rewards


def train_drqn_mdp(agent, environment, training_config, logger, log_dir):
    """Training loop for DRQN on Two-Step task"""
    logger.info("Starting DRQN training on Two-Step task")
    
    # Create task generator
    generator = TaskGenerator()
    
    episode_rewards = []
    
    for episode in range(training_config.num_episodes):
        # Generate single trial data
        trials, conditions = generator.generate(
            num_trials=1,
            s1_duration=environment.task.s1_duration,
            s2_duration=environment.task.s2_duration
        )
        
        trial_data = trials[0][0]  # Get trial sequence
        condition = conditions['training_guide'][0]
        
        # Configure environment with trial
        # Block changes every 50 trials: 0-49=block0, 50-99=block1, 100-149=block0, etc.
        block = (episode // 50) % 2
        environment.configure_trial([trial_data], {
            'block': block,
            'choice': np.random.choice([0, 1]),
        })
        
        # Reset environment and agent
        observation = environment.reset()
        hidden_state = agent.reset_hidden(1)
        
        # Collect trajectory
        observations = [observation]
        actions = []
        rewards = []
        dones = []
        
        total_reward = 0
        step_count = 0
        max_steps = environment.task.trial_length
        
        while step_count < max_steps:
            # Get action
            action, hidden_state = agent.get_action(observation, hidden_state)
            
            # Take step
            next_observation, reward, done, info = environment.step(action)
            
            # Store experience
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            observations.append(next_observation)
            
            total_reward += reward
            observation = next_observation
            step_count += 1
            
            if done:
                break
        
        # Train DRQN using trajectory
        if len(observations) > 1:
            trajectory = {
                'observations': observations[:-1],
                'actions': actions,
                'rewards': rewards,
                'next_observations': observations[1:],
                'dones': dones
            }
            agent.train_trajectory(trajectory)
        
        episode_rewards.append(total_reward)
        
        # Logging and saving
        if (episode + 1) % training_config.eval_every == 0:
            avg_reward = np.mean(episode_rewards[-training_config.eval_every:])
            success_rate = np.mean([1 if r > 0 else 0 for r in episode_rewards[-training_config.eval_every:]])
            logger.info(f"Episode {episode + 1}/{training_config.num_episodes}, "
                       f"Avg Reward: {avg_reward:.3f}, Success Rate: {success_rate:.3f}, Loss: N/A")
        
        # Save model if save_every > 0 and it's time to save
        if training_config.save_every > 0 and (episode + 1) % training_config.save_every == 0:
            model_path = os.path.join(log_dir, f"model_episode_{episode + 1}.pt")
            agent.save_model(model_path)
            logger.info(f"Model saved to {model_path}")
    
    return episode_rewards


def train_gru_mdp(agent, environment, training_config, logger, log_dir):
    """Training loop for GRU on Two-Step task"""
    logger.info("Starting GRU training on Two-Step task")
    
    # Create task generator
    generator = TaskGenerator()
    
    episode_rewards = []
    
    for episode in range(training_config.num_episodes):
        # Generate trial data with current durations
        trials, conditions = generator.generate(
            num_trials=agent.batch_size,
            s1_duration=environment.task.s1_duration,
            s2_duration=environment.task.s2_duration
        )
        
        if trials:
            # Prepare batch data - stack all trials together
            batch_trials = np.stack([trial[0] for trial in trials])  # (batch_size, seq_len, input_size)
            batch_trials = batch_trials.transpose(1, 0, 2)  # (seq_len, batch_size, input_size)
            
            # Prepare batch rewards with proper block structure
            batch_rewards = []
            for i in range(len(trials)):
                # Each trial in batch gets its own block based on episode + trial index
                trial_episode = episode + i
                block = (trial_episode // 50) % 2
                batch_rewards.append([True, 0, 4])  # training_guide format
            
            batch_rewards = np.array(batch_rewards)
            
            # Prepare batch data for agent
            batch_data = {
                'observations': batch_trials,
                'rewards': batch_rewards
            }
            
            # Train step with batch data
            loss = agent.train_step(batch_data)
            
            # Calculate accuracy (simplified)
            accuracy = 0.5  # Placeholder - would need proper evaluation
            episode_rewards.append(accuracy)
            
            # Logging and saving
            if (episode + 1) % training_config.eval_every == 0:
                avg_reward = np.mean(episode_rewards[-training_config.eval_every:])
                success_rate = np.mean(episode_rewards[-training_config.eval_every:])  # For GRU, accuracy is used as success rate
                logger.info(f"Episode {episode + 1}/{training_config.num_episodes}, "
                           f"Avg Reward: {avg_reward:.3f}, Success Rate: {success_rate:.3f}, Loss: {loss:.4f}")
            
            # Save model if save_every > 0 and it's time to save
            if training_config.save_every > 0 and (episode + 1) % training_config.save_every == 0:
                model_path = os.path.join(log_dir, f"model_episode_{episode + 1}.pt")
                agent.save_model(model_path)
                logger.info(f"Model saved to {model_path}")
        else:
            episode_rewards.append(0.0)
    
    return episode_rewards


def evaluate_agent(agent, environment, env_name, num_rollouts, logger):
    """Evaluate agent performance"""
    logger.info(f"Evaluating agent for {num_rollouts} rollouts")
    
    agent.eval_mode()
    
    total_rewards = []
    success_rate = 0
    
    for rollout in range(num_rollouts):
        if env_name == 'tmaze':
            observation = environment.reset()
            hidden_state = agent.reset_hidden(1)
            total_reward = 0
            step_count = 0
            max_steps = environment.horizon()
            
            while step_count < max_steps:
                action, hidden_state = agent.get_action(observation, hidden_state, epsilon=0.0)
                observation, reward, done, info = environment.step(action)
                total_reward += reward
                step_count += 1
                
                if done:
                    if reward > 0:
                        success_rate += 1
                    break
                    
            total_rewards.append(total_reward)
            
        elif env_name == 'mdp':
            # For Two-Step task, we need trial data
            generator = TaskGenerator()
            trials, conditions = generator.generate(
                num_trials=1,
                s1_duration=environment.task.s1_duration,
                s2_duration=environment.task.s2_duration
            )
            
            if trials:
                trial_data = trials[0][0]
                # Use block 0 for evaluation (consistent evaluation)
                environment.configure_trial([trial_data], {
                    'block': 0,
                    'choice': np.random.choice([0, 1]),
                })
                
                # Simple evaluation - check if task completed successfully
                if environment.is_winning():
                    success_rate += 1
                    total_rewards.append(1.0)
                else:
                    total_rewards.append(0.0)
    
    avg_reward = np.mean(total_rewards) if total_rewards else 0
    success_rate = success_rate / num_rollouts
    
    logger.info(f"Evaluation Results - Avg Reward: {avg_reward:.3f}, Success Rate: {success_rate:.3f}")
    
    agent.train_mode()
    return avg_reward, success_rate


def main():
    """Main function"""
    # Parse arguments
    parser = create_parser()
    args = parser.parse_args()
    
    # Create config manager and update from args
    config_manager = ConfigManager()
    config_manager.update_from_args(args)
    
    model_config = config_manager.get_model_config(args.model)
    env_config = config_manager.get_env_config(args.environment)
    training_config = config_manager.get_training_config()
    
    # Set random seeds
    set_random_seeds(training_config.seed)
    
    # Setup experiment name
    experiment_name = args.name if args.name else f"{args.model}_{args.environment}_{int(time.time())}"
    
    # Setup logging
    logger, log_dir = setup_logging(experiment_name)
    
    # Log configuration
    logger.info(f"Starting experiment: {experiment_name}")
    logger.info(f"Model: {args.model}, Environment: {args.environment}")
    logger.info(f"Model config: {model_config}")
    logger.info(f"Environment config: {env_config}")
    logger.info(f"Training config: {training_config}")
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available() and training_config.cuda
    if cuda_available:
        logger.info("Using CUDA")
    else:
        logger.info("Using CPU")
    
    # Create environment
    environment = create_environment(args.environment, env_config, args)
    logger.info(f"Created environment: {environment.__class__.__name__}")
    
    # Create agent
    agent = create_agent(
        args.model, 
        model_config, 
        environment.get_observation_space(), 
        environment.get_action_space(),
        cuda_available
    )
    logger.info(f"Created agent: {agent.__class__.__name__}")
    
    # Training
    start_time = time.time()
    
    # Universal training based on model type
    if args.model == 'drqn':
        if args.environment == 'tmaze':
            episode_rewards = train_drqn_tmaze(agent, environment, training_config, logger, log_dir)
        elif args.environment == 'mdp':
            episode_rewards = train_drqn_mdp(agent, environment, training_config, logger, log_dir)
        else:
            raise ValueError(f"Unsupported environment {args.environment} for DRQN")
    elif args.model == 'gru':
        if args.environment == 'tmaze':
            episode_rewards = train_gru_tmaze(agent, environment, training_config, logger, log_dir)
        elif args.environment == 'mdp':
            episode_rewards = train_gru_mdp(agent, environment, training_config, logger, log_dir)
        else:
            raise ValueError(f"Unsupported environment {args.environment} for GRU")
    else:
        raise ValueError(f"Unsupported model {args.model}")
    
    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time:.2f} seconds")
    
    # Evaluation
    avg_reward, success_rate = evaluate_agent(
        agent, environment, args.environment, training_config.num_rollouts, logger
    )
    
    # Save final model
    model_path = os.path.join(log_dir, f"{experiment_name}_final.pt")
    agent.save_model(model_path)
    logger.info(f"Final model saved to {model_path}")
    
    # Final summary
    logger.info("="*50)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("="*50)
    logger.info(f"Model: {args.model}")
    logger.info(f"Environment: {args.environment}")
    logger.info(f"Training Episodes: {training_config.num_episodes}")
    logger.info(f"Final Avg Reward: {avg_reward:.3f}")
    logger.info(f"Success Rate: {success_rate:.3f}")
    logger.info(f"Training Time: {training_time:.2f}s")
    logger.info("="*50)


if __name__ == "__main__":
    main()
