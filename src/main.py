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
from dataclasses import asdict
import yaml

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


def hidden_to_numpy(hidden_state):
    """Extract a 1D numpy activation vector from various hidden state types."""
    if isinstance(hidden_state, tuple):
        if len(hidden_state) == 1:
            h = hidden_state[0]
        else:
            h = hidden_state[0]
    else:
        h = hidden_state

    if isinstance(h, torch.Tensor):
        # h shape: (num_layers, batch, hidden_size)
        return h[-1, 0, :].detach().cpu().numpy()
    return np.array(h)


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


def create_agent(model_name: str, model_config, observation_size: int, action_size: int, cuda_enabled: bool, sequence_length: int = 20):
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
            target_update_freq=model_config.target_update_freq,
            sequence_length=sequence_length,
            cuda_enabled=cuda_enabled
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


def save_run_config(config_path: str,
                    experiment_name: str,
                    model_name: str,
                    env_name: str,
                    model_config,
                    env_config,
                    training_config,
                    sequence_length: int):
    """Persist a YAML config capturing model/env/training args for later eval."""
    try:
        cfg = {
            'name': experiment_name,
            'model': model_name,
            'environment': env_name,
            'sequence_length': int(sequence_length),
            'model_config': asdict(model_config) if hasattr(model_config, '__dict__') else dict(model_config),
            'env_config': asdict(env_config) if hasattr(env_config, '__dict__') else dict(env_config),
            'training_config': asdict(training_config) if hasattr(training_config, '__dict__') else dict(training_config),
        }
        with open(config_path, 'w') as f:
            yaml.safe_dump(cfg, f)
    except Exception as e:
        logging.getLogger(__name__).warning(f"Failed to save config.yaml: {e}")


def create_environment(env_name: str, env_config, args=None):
    """Create environment based on name and configuration"""
    if env_name == 'tmaze':
        env = TMazeEnvironment(
            length=env_config.length,
            stochasticity=env_config.stochasticity,
            irrelevant_features=env_config.irrelevant_features,
            obs_mode=getattr(env_config, 'obs_mode', 'type')
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
            if hasattr(args, 'mdp_reward_boundary') and args.mdp_reward_boundary is not None:
                task_kwargs['reward_boundary'] = args.mdp_reward_boundary
            if hasattr(args, 'mdp_permute_actions') and args.mdp_permute_actions is not None:
                task_kwargs['permute_actions'] = args.mdp_permute_actions
                
        return MDPEnvironment(**task_kwargs)
    else:
        raise ValueError(f"Unknown environment: {env_name}")


def train_drqn_tmaze(agent, environment, training_config, logger, log_dir):
    """Training loop for DRQN on T-maze"""
    logger.info("Starting DRQN training on T-maze")
    
    episode_rewards = []
    training_episodes = []
    
    for episode in range(training_config.num_episodes):
        # Reset environment and agent
        observation = environment.reset()
        hidden_state = agent.reset_hidden(1)
        
        # Collect trajectory
        observations = [observation]
        actions = []
        rewards = []
        dones = []

        # For logging trajectory and activations
        episode_states = []
        episode_actions = []
        episode_rewards_list = []
        episode_dones = []
        episode_activations = []
        
        total_reward = 0
        step_count = 0
        max_steps = environment.horizon()
        
        while step_count < max_steps:
            # Record current state
            episode_states.append(np.asarray(observation, dtype=float))

            # Get action
            action, hidden_state = agent.get_action(observation, hidden_state)

            # Record activation after processing observation
            episode_activations.append(hidden_to_numpy(hidden_state))
            
            # Take step
            next_observation, reward, done, info = environment.step(action)
            
            # Store experience
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            observations.append(next_observation)

            # Store per-step info for logging
            episode_actions.append(int(action))
            episode_rewards_list.append(float(reward))
            episode_dones.append(bool(done))
            
            total_reward += reward
            step_count += 1
            observation = next_observation
            
            if done:
                break

        # Store per-episode trajectory and activations for analysis
        training_episodes.append({
            'states': episode_states,
            'actions': episode_actions,
            'rewards': episode_rewards_list,
            'dones': episode_dones,
            'activations': episode_activations,
            'total_reward': float(total_reward),
        })
        
        # Store trajectory in replay buffer
        agent.store_trajectory(observations[:-1], actions, rewards, dones)
        
        # Train agent
        loss = 0.0
        if len(agent.replay_buffer) > 100:  # Start learning much earlier
            loss = agent.train_step(None)  # DRQN uses replay buffer internally
        
        episode_rewards.append(total_reward)
        
        # Logging and saving
        if (episode + 1) % training_config.eval_every == 0:
            # Evaluate current policy with greedy actions
            eval_avg_reward, eval_success_rate = evaluate_agent(
                agent, environment, 'tmaze', training_config.num_rollouts, logger
            )
            logger.info(
                f"Episode {episode + 1}/{training_config.num_episodes}, "
                f"Eval Avg Reward: {eval_avg_reward:.3f}, Eval Success Rate: {eval_success_rate:.3f}, "
                f"Train Loss: {loss:.4f}"
            )
        
        # Save model if save_every > 0 and it's time to save
        if training_config.save_every > 0 and (episode + 1) % training_config.save_every == 0:
            model_path = os.path.join(log_dir, f"model_episode_{episode + 1}.pt")
            agent.save_model(model_path)
            logger.info(f"Model saved to {model_path}")

    # Save training trajectories and activations (T-maze)
    if training_episodes:
        ts = int(time.time())
        out_path = os.path.join(log_dir, f"train_episodes_tmaze_{ts}.npz")
        meta = {
            'env': 'tmaze',
            'model': 'drqn',
            'seed': int(getattr(training_config, 'seed', 0)),
            'timestamp': ts,
            'source': 'train',
        }
        np.savez(out_path,
                 episodes=np.array(training_episodes, dtype=object),
                 meta=meta)
        logger.info(f"Training trajectories and activations saved to {out_path}")

    return episode_rewards


def train_gru_tmaze(agent, environment, training_config, logger, log_dir):
    """Training loop for GRU on T-maze"""
    logger.info("Starting GRU training on T-maze")
    
    episode_rewards = []
    training_episodes = []
    
    for episode in range(training_config.num_episodes):
        # Reset environment and agent
        observation = environment.reset()
        hidden_state = agent.model.init_hidden(1)
        
        # Collect trajectory
        observations = [observation]
        actions = []
        rewards = []

        # For logging trajectory and activations
        episode_states = []
        episode_actions = []
        episode_rewards_list = []
        episode_dones = []
        episode_activations = []
        
        total_reward = 0
        step_count = 0
        max_steps = environment.horizon()
        
        while step_count < max_steps:
            # Record current observation
            episode_states.append(np.asarray(observation, dtype=float))

            # Get action using GRU
            action, hidden_state = agent.get_action(observation, hidden_state, env_name='tmaze')

            # Record activation after processing observation
            episode_activations.append(hidden_to_numpy(hidden_state))
            
            # Take step
            next_observation, reward, done, info = environment.step(action)
            
            # Store experience
            actions.append(action)
            rewards.append(reward)
            observations.append(next_observation)

            # Store per-step info for logging
            episode_actions.append(int(action))
            episode_rewards_list.append(float(reward))
            episode_dones.append(bool(done))
            
            total_reward += reward
            observation = next_observation
            step_count += 1
            
            if done:
                break

        # Store per-episode trajectory and activations for analysis
        training_episodes.append({
            'states': episode_states,
            'actions': episode_actions,
            'rewards': episode_rewards_list,
            'dones': episode_dones,
            'activations': episode_activations,
            'total_reward': float(total_reward),
        })
        
        # Train GRU using collected trajectory
        loss = 0.0
        if len(observations) > 1:
            # Convert trajectory to training format for GRU
            obs_array = np.array(observations[:-1])  # Convert to numpy first
            obs_sequence = torch.FloatTensor(obs_array).unsqueeze(1)  # (seq_len, 1, obs_size)
            action_sequence = torch.LongTensor(actions)
            reward_sequence = torch.FloatTensor(rewards)
            
            # Simple Q-learning style training for GRU
            loss = agent.train_trajectory(obs_sequence, action_sequence, reward_sequence)
        
        episode_rewards.append(total_reward)
        
        # Logging and saving
        if (episode + 1) % training_config.eval_every == 0:
            # Evaluate current policy with greedy actions
            eval_avg_reward, eval_success_rate = evaluate_agent(
                agent, environment, 'tmaze', training_config.num_rollouts, logger
            )
            logger.info(
                f"Episode {episode + 1}/{training_config.num_episodes}, "
                f"Eval Avg Reward: {eval_avg_reward:.3f}, Eval Success Rate: {eval_success_rate:.3f}, "
                f"Train Loss: {loss:.4f}"
            )
        
        # Save model if save_every > 0 and it's time to save
        if training_config.save_every > 0 and (episode + 1) % training_config.save_every == 0:
            model_path = os.path.join(log_dir, f"model_episode_{episode + 1}.pt")
            agent.save_model(model_path)
            logger.info(f"Model saved to {model_path}")

    # Save training trajectories and activations (T-maze)
    if training_episodes:
        ts = int(time.time())
        out_path = os.path.join(log_dir, f"train_episodes_tmaze_{ts}.npz")
        meta = {
            'env': 'tmaze',
            'model': 'gru',
            'seed': int(getattr(training_config, 'seed', 0)),
            'timestamp': ts,
            'source': 'train',
        }
        np.savez(out_path,
                 episodes=np.array(training_episodes, dtype=object),
                 meta=meta)
        logger.info(f"Training trajectories and activations saved to {out_path}")

    return episode_rewards


def train_drqn_mdp(agent, environment, training_config, logger, log_dir):
    """Training loop for DRQN on Two-Step task (event-driven)."""
    logger.info("Starting DRQN training on Two-Step task")

    episode_rewards = []
    training_episodes = []
    block_size = int(getattr(training_config, 'mdp_block_size', 0) or 0)
    num_blocks = int(getattr(training_config, 'mdp_num_blocks', 0) or 0)

    for episode in range(training_config.num_episodes):
        # Reset environment for a new trial
        observation = environment.reset()

        # Reset agent hidden state
        hidden_state = agent.reset_hidden(1)

        # Collect trajectory
        observations = [observation]
        actions = []
        rewards = []
        dones = []

        # For logging trajectory and activations
        episode_states = []
        episode_actions = []
        episode_rewards_list = []
        episode_dones = []
        episode_activations = []

        total_reward = 0.0
        done = False
        while not done:
            # Record current observation
            episode_states.append(np.asarray(observation, dtype=float))

            action, hidden_state = agent.get_action(observation, hidden_state)
            # Record activation after processing observation
            episode_activations.append(hidden_to_numpy(hidden_state))

            next_observation, reward, done, info = environment.step(action)

            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            observations.append(next_observation)

            # Store per-step info for logging
            episode_actions.append(int(action))
            episode_rewards_list.append(float(reward))
            episode_dones.append(bool(done))

            total_reward += reward
            observation = next_observation

        # Store per-episode trajectory and activations for analysis
        block_index = None
        if block_size > 0:
            block_index = (episode // block_size) + 1
        training_episodes.append({
            'states': episode_states,
            'actions': episode_actions,
            'rewards': episode_rewards_list,
            'dones': episode_dones,
            'activations': episode_activations,
            'total_reward': float(total_reward),
            'block': block_index,
        })

        # Train DRQN using trajectory
        loss = 0.0
        if len(observations) > 1:
            trajectory = {
                'observations': observations[:-1],
                'actions': actions,
                'rewards': rewards,
                'next_observations': observations[1:],
                'dones': dones
            }
            loss = agent.train_trajectory(trajectory)

        episode_rewards.append(total_reward)

        # Logging and saving
        if (episode + 1) % training_config.eval_every == 0:
            eval_avg_reward, eval_success_rate = evaluate_agent(
                agent, environment, 'mdp', training_config.num_rollouts, logger
            )
            logger.info(
                f"Episode {episode + 1}/{training_config.num_episodes}, "
                f"Eval Avg Reward: {eval_avg_reward:.3f}, Eval Success Rate: {eval_success_rate:.3f}, "
                f"Train Loss: {loss:.4f}"
            )

        if training_config.save_every > 0 and (episode + 1) % training_config.save_every == 0:
            model_path = os.path.join(log_dir, f"model_episode_{episode + 1}.pt")
            agent.save_model(model_path)
            logger.info(f"Model saved to {model_path}")
        if block_size > 0 and (episode + 1) % block_size == 0:
            suffix = f"/{num_blocks}" if num_blocks > 0 else ""
            logger.info(f"Completed MDP block {episode // block_size + 1}{suffix}")

    # Save training trajectories and activations (MDP)
    if training_episodes:
        ts = int(time.time())
        out_path = os.path.join(log_dir, f"train_episodes_mdp_{ts}.npz")
        meta = {
            'env': 'mdp',
            'model': 'drqn',
            'seed': int(getattr(training_config, 'seed', 0)),
            'timestamp': ts,
            'source': 'train',
        }
        np.savez(out_path,
                 episodes=np.array(training_episodes, dtype=object),
                 meta=meta)
        logger.info(f"Training trajectories and activations saved to {out_path}")

    return episode_rewards


def train_gru_mdp(agent, environment, training_config, logger, log_dir):
    """On-policy RL training loop for GRU on Two-Step task (REINFORCE)."""
    logger.info("Starting GRU RL training on Two-Step task")
    
    episode_rewards = []
    training_episodes = []
    block_size = int(getattr(training_config, 'mdp_block_size', 0) or 0)
    num_blocks = int(getattr(training_config, 'mdp_num_blocks', 0) or 0)

    for episode in range(training_config.num_episodes):
        # Reset environment for a new trial
        observation = environment.reset()

        # Reset agent hidden state
        hidden_state = agent.model.init_hidden(1)
        log_probs = []
        total_reward = 0.0

        # For logging trajectory and activations
        states = []
        actions = []
        activations = []
        rewards = []

        # Act at both phases until done
        done = False
        while not done:
            states.append(np.asarray(observation, dtype=float))

            action, log_prob, hidden_state = agent.policy_action(observation, hidden_state, deterministic=False)
            log_probs.append(log_prob)
            actions.append(int(action))
            activations.append(hidden_to_numpy(hidden_state))

            observation, reward, done, info = environment.step(action)
            total_reward += reward
            rewards.append(float(reward))

        # Policy gradient update
        loss = agent.reinforce_update(log_probs, total_reward)
        episode_rewards.append(total_reward)

        # Store per-episode trajectory and activations for analysis
        block_index = None
        if block_size > 0:
            block_index = (episode // block_size) + 1
        training_episodes.append({
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'activations': activations,
            'total_reward': float(total_reward),
            'block': block_index,
        })

        # Evaluate periodically
        if (episode + 1) % training_config.eval_every == 0:
            eval_avg_reward, eval_success_rate = evaluate_agent(
                agent, environment, 'mdp', training_config.num_rollouts, logger
            )
            logger.info(
                f"Episode {episode + 1}/{training_config.num_episodes}, "
                f"Eval Avg Reward: {eval_avg_reward:.3f}, Eval Success Rate: {eval_success_rate:.3f}, "
                f"Train Loss: {loss:.4f}"
            )

        # Save model periodically
        if training_config.save_every > 0 and (episode + 1) % training_config.save_every == 0:
            model_path = os.path.join(log_dir, f"model_episode_{episode + 1}.pt")
            agent.save_model(model_path)
            logger.info(f"Model saved to {model_path}")
        if block_size > 0 and (episode + 1) % block_size == 0:
            suffix = f"/{num_blocks}" if num_blocks > 0 else ""
            logger.info(f"Completed MDP block {episode // block_size + 1}{suffix}")

    # Save training trajectories and activations (MDP)
    if training_episodes:
        ts = int(time.time())
        out_path = os.path.join(log_dir, f"train_episodes_mdp_{ts}.npz")
        meta = {
            'env': 'mdp',
            'model': 'gru',
            'seed': int(getattr(training_config, 'seed', 0)),
            'timestamp': ts,
            'source': 'train',
        }
        np.savez(out_path,
                 episodes=np.array(training_episodes, dtype=object),
                 meta=meta)
        logger.info(f"Training trajectories and activations saved to {out_path}")

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
                # GRU needs env_name to map to 4 actions; DRQN ignores this arg
                if hasattr(agent, 'model'):
                    action, hidden_state = agent.get_action(observation, hidden_state, epsilon=0.0, env_name='tmaze')
                else:
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
            # Event-driven two-step: act until done
            observation = environment.reset()
            hidden_state = agent.reset_hidden(1)
            total_reward = 0.0

            done = False
            while not done:
                if hasattr(agent, 'model'):
                    action, hidden_state = agent.get_action(observation, hidden_state, epsilon=0.0, env_name='mdp')
                else:
                    action, hidden_state = agent.get_action(observation, hidden_state, epsilon=0.0)
                observation, reward, done, info = environment.step(action)
                total_reward += reward

            if environment.is_winning():
                success_rate += 1
            total_rewards.append(total_reward)
    
    avg_reward = np.mean(total_rewards) if total_rewards else 0
    success_rate = success_rate / num_rollouts
    
    logger.info(f"Evaluation Results - Avg Reward: {avg_reward:.3f}, Success Rate: {success_rate:.3f}")
    
    agent.train_mode()
    return avg_reward, success_rate


def run_evaluation_mode(agent, environment, env_name, num_episodes, logger, log_dir):
    """Run evaluation mode with trajectory and activation saving"""
    logger.info(f"Starting evaluation mode for {num_episodes} episodes")
    
    agent.eval_mode()
    
    all_trajectories = []
    all_activations = []
    episode_rewards = []
    success_count = 0
    
    for episode in range(num_episodes):
        episode_trajectory = []
        episode_activations = []
        
        if env_name == 'tmaze':
            observation = environment.reset()
            hidden_state = agent.reset_hidden(1)
            total_reward = 0
            step_count = 0
            max_steps = environment.horizon()
            
            while step_count < max_steps:
                # Get action and hidden state
                action, hidden_state = agent.get_action(observation, hidden_state, epsilon=0.0, env_name='tmaze')
                
                # Store trajectory (s, a)
                episode_trajectory.append({
                    'state': observation.copy(),
                    'action': action
                })
                
                # Store activation (hidden state)
                if hasattr(agent, 'model') and hasattr(agent.model, 'rnn'):
                    # For RNN models, store hidden state
                    if isinstance(hidden_state, tuple):
                        episode_activations.append(hidden_state[0].detach().cpu().numpy())
                    else:
                        episode_activations.append(hidden_state.detach().cpu().numpy())
                
                # Take step
                next_observation, reward, done, info = environment.step(action)
                total_reward += reward
                step_count += 1
                observation = next_observation
                
                if done:
                    if reward > 0:
                        success_count += 1
                    break
            
            # Add final reward to trajectory
            if episode_trajectory:
                episode_trajectory[-1]['reward'] = total_reward
                
        elif env_name == 'mdp':
            # Event-driven two-step
            observation = environment.reset()
            hidden_state = agent.reset_hidden(1)
            total_reward = 0

            done = False
            while not done:
                # Choose action greedily
                action, hidden_state = agent.get_action(observation, hidden_state, epsilon=0.0, env_name='mdp')

                # Store trajectory (s, a)
                episode_trajectory.append({
                    'state': observation.copy(),
                    'action': action
                })

                # Store activation
                if hasattr(agent, 'model') and hasattr(agent.model, 'rnn'):
                    if isinstance(hidden_state, tuple):
                        episode_activations.append(hidden_state[0].detach().cpu().numpy())
                    else:
                        episode_activations.append(hidden_state.detach().cpu().numpy())

                observation, reward, done, info = environment.step(action)
                total_reward += reward

            # Add final reward to all steps
            for step_data in episode_trajectory:
                step_data['reward'] = total_reward

            # Check if winning
            is_winning = environment.is_winning()
            if is_winning:
                success_count += 1
        
        all_trajectories.append(episode_trajectory)
        all_activations.append(episode_activations)
        episode_rewards.append(total_reward)
        
        # Log progress
        if (episode + 1) % 10 == 0:
            logger.info(f"Evaluation Episode {episode + 1}/{num_episodes}, "
                       f"Avg Reward: {np.mean(episode_rewards):.3f}, "
                       f"Success Rate: {success_count/(episode+1):.3f}")
    
    # Calculate final metrics
    avg_reward = np.mean(episode_rewards)
    success_rate = success_count / num_episodes
    
    logger.info(f"Evaluation Complete - Avg Reward: {avg_reward:.3f}, Success Rate: {success_rate:.3f}")
    
    # Save trajectories
    trajectory_file = os.path.join(log_dir, f"trajectories_{env_name}.npz")
    np.savez(trajectory_file, trajectories=all_trajectories)
    logger.info(f"Trajectories saved to {trajectory_file}")
    
    # Save activations
    activation_file = os.path.join(log_dir, f"activations_{env_name}.npz")
    np.savez(activation_file, activations=all_activations)
    logger.info(f"Activations saved to {activation_file}")
    
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
    
    # Check if this is evaluation mode
    if args.preload:
        # Evaluation mode
        experiment_name = f"eval_{args.model}_{args.environment}_{int(time.time())}"
        logger, log_dir = setup_logging(experiment_name)
        
        logger.info(f"Starting evaluation mode: {experiment_name}")
        logger.info(f"Preloading model from: {args.preload}")
        logger.info(f"Model: {args.model}, Environment: {args.environment}")
        logger.info(f"Evaluation episodes: {args.eval_num_episodes}")
        
        # Check CUDA availability
        cuda_available = torch.cuda.is_available() and training_config.cuda
        if cuda_available:
            logger.info("Using CUDA")
        else:
            logger.info("Using CPU")
        
        # Create environment
        environment = create_environment(args.environment, env_config, args)
        logger.info(f"Created environment: {environment.__class__.__name__}")
        
        # Calculate sequence length based on environment
        if args.environment == 'tmaze':
            # T-maze: start(0) + corridor(1~length) + junction(length+1) + goal(length+2)
            sequence_length = args.length + 3
        elif args.environment == 'mdp':
            # MDP: s1 + a1 + s2 + a2 + reward = 5 steps
            sequence_length = 5
        else:
            sequence_length = 20  # default
        
        # Create agent
        agent = create_agent(
            args.model, 
            model_config, 
            environment.get_observation_space(), 
            environment.get_action_space(),
            cuda_available,
            sequence_length
        )
        logger.info(f"Created agent: {agent.__class__.__name__}")
        
        # Load model - prepend logs/ if not already present
        model_path = args.preload
        if not model_path.startswith('logs/'):
            model_path = os.path.join('logs', model_path)
        
        agent.load_model(model_path)
        logger.info(f"Model loaded from {model_path}")
        
        # Run evaluation
        avg_reward, success_rate = run_evaluation_mode(
            agent, environment, args.environment, args.eval_num_episodes, logger, log_dir
        )
        
        # Final summary
        logger.info("="*50)
        logger.info("EVALUATION SUMMARY")
        logger.info("="*50)
        logger.info(f"Model: {args.model}")
        logger.info(f"Environment: {args.environment}")
        logger.info(f"Evaluation Episodes: {args.eval_num_episodes}")
        logger.info(f"Avg Reward: {avg_reward:.3f}")
        logger.info(f"Success Rate: {success_rate:.3f}")
        logger.info("="*50)
        
        return
    
    # Training mode
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
    
    # Calculate sequence length based on environment
    if args.environment == 'tmaze':
        # T-maze: start(0) + corridor(1~length) + junction(length+1) + goal(length+2)
        sequence_length = args.length + 3
    elif args.environment == 'mdp':
        # MDP: s1 + a1 + s2 + a2 + reward = 5 steps
        sequence_length = 5
    else:
        sequence_length = 20  # default
    
    # Create agent
    agent = create_agent(
        args.model, 
        model_config, 
        environment.get_observation_space(), 
        environment.get_action_space(),
        cuda_available,
        sequence_length
    )
    logger.info(f"Created agent: {agent.__class__.__name__}")

    # Persist configuration for later evaluation
    config_yaml_path = os.path.join(log_dir, 'config.yaml')
    # Convert env_config to a plain dict, and include dynamic mdp params if any
    try:
        env_cfg_dict = asdict(env_config)
    except Exception:
        env_cfg_dict = dict(env_config)
    if args.environment == 'mdp':
        # Include transition and reward probabilities actually used
        if hasattr(args, 'trans_prob') and args.trans_prob is not None:
            env_cfg_dict['trans_prob'] = args.trans_prob
        elif hasattr(environment, 'task') and hasattr(environment.task, 'trans_prob'):
            env_cfg_dict['trans_prob'] = float(environment.task.trans_prob)
        if hasattr(args, 'reward_prob') and args.reward_prob is not None:
            env_cfg_dict['reward_prob'] = args.reward_prob
        elif hasattr(environment, 'task') and hasattr(environment.task, 'reward_prob'):
            env_cfg_dict['reward_prob'] = float(environment.task.reward_prob)
        if hasattr(args, 'mdp_reward_boundary') and args.mdp_reward_boundary is not None:
            env_cfg_dict['reward_boundary'] = args.mdp_reward_boundary
        elif hasattr(environment, 'task') and hasattr(environment.task, 'reward_boundary'):
            env_cfg_dict['reward_boundary'] = str(environment.task.reward_boundary)
        if hasattr(args, 'mdp_permute_actions') and args.mdp_permute_actions is not None:
            env_cfg_dict['permute_actions'] = bool(args.mdp_permute_actions)
        elif hasattr(environment, 'task') and hasattr(environment.task, 'permute_actions'):
            env_cfg_dict['permute_actions'] = bool(environment.task.permute_actions)

    save_run_config(
        config_yaml_path,
        experiment_name,
        args.model,
        args.environment,
        model_config,
        env_cfg_dict,
        training_config,
        sequence_length,
    )
    
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
    final_model_path = os.path.join(log_dir, f"{experiment_name}_final.pt")
    agent.save_model(final_model_path)
    logger.info(f"Final model saved to {final_model_path}")

    # Also save a standard filename for evaluation script compatibility
    std_model_path = os.path.join(log_dir, 'model.pt')
    try:
        agent.save_model(std_model_path)
        logger.info(f"Evaluation checkpoint saved to {std_model_path}")
    except Exception as e:
        logger.warning(f"Failed to save standard model checkpoint: {e}")
    
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
