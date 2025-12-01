"""
DRQN-based RL Agent

Adapted from belief-rnn-main project.
Maintains original logic while implementing unified interface.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from random import random
from collections import namedtuple, deque

from .base_agent import BaseAgent


# RNN implementations from original belief-rnn project
class GRU(nn.Module):
    """Gated Recurrent Unit from original project"""
    num_states = 1

    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        self.rnn = nn.GRU(input_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x, h=None):
        if h is not None:
            h, = h
        x, h = self.rnn(x, h)
        x = self.linear(x)
        return x, (h,)


class LSTM(nn.Module):
    """Long Short Term Memory from original project"""
    num_states = 2

    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        self.rnn = nn.LSTM(input_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x, h=None):
        x, h = self.rnn(x, h)
        x = self.linear(x)
        return x, h


RNNS = {'lstm': LSTM, 'gru': GRU}


# Memory classes from original project
Trajectory = namedtuple('Trajectory', ['observations', 'actions', 'rewards', 'dones'])


class ReplayBuffer:
    """
    Replay buffer for DRQN (from original project)
    """

    def __init__(self, capacity=8192):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def store(self, trajectory):
        self.buffer.append(trajectory)

    def sample(self, batch_size, sequence_length):
        # Sample random trajectories
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        trajectories = [self.buffer[i] for i in indices]
        
        # Pre-allocate arrays for better performance
        max_seq_len = min(sequence_length, max(len(t.observations) for t in trajectories))
        batch_observations = np.zeros((batch_size, max_seq_len, trajectories[0].observations[0].shape[0]), dtype=np.float32)
        batch_actions = np.zeros((batch_size, max_seq_len), dtype=np.int64)
        batch_rewards = np.zeros((batch_size, max_seq_len), dtype=np.float32)
        batch_dones = np.zeros((batch_size, max_seq_len), dtype=bool)
        batch_lengths = np.zeros(batch_size, dtype=np.int32)
        
        for i, trajectory in enumerate(trajectories):
            # Sample random sequence from trajectory
            if len(trajectory.observations) > sequence_length:
                start_idx = np.random.randint(0, len(trajectory.observations) - sequence_length)
                end_idx = start_idx + sequence_length
            else:
                start_idx = 0
                end_idx = len(trajectory.observations)
            
            seq_len = end_idx - start_idx
            batch_lengths[i] = seq_len
            
            # Convert to numpy arrays efficiently
            obs_array = np.array(trajectory.observations[start_idx:end_idx], dtype=np.float32)
            batch_observations[i, :seq_len] = obs_array
            batch_actions[i, :seq_len] = np.array(trajectory.actions[start_idx:end_idx], dtype=np.int64)
            batch_rewards[i, :seq_len] = np.array(trajectory.rewards[start_idx:end_idx], dtype=np.float32)
            batch_dones[i, :seq_len] = np.array(trajectory.dones[start_idx:end_idx], dtype=bool)
        
        return {
            'observations': batch_observations,
            'actions': batch_actions,
            'rewards': batch_rewards,
            'dones': batch_dones,
            'lengths': batch_lengths
        }

    def __len__(self):
        return len(self.buffer)


class DRQNAgent(BaseAgent):
    """
    Deep Recurrent Q-Network reinforcement learning agent (from original project)
    """

    def __init__(self, observation_size, action_size, hidden_size=32, num_layers=2,
                 cell='gru', learning_rate=1e-3, gamma=0.98, epsilon=0.2,
                 buffer_capacity=8192, target_update_freq=10, sequence_length=20, cuda_enabled=False):
        super().__init__(observation_size, action_size, hidden_size, num_layers)
        
        self.cell = cell
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.target_update_freq = target_update_freq
        self.sequence_length = sequence_length
        self.cuda_enabled = cuda_enabled
        self.step_count = 0
        
        # Initialize Q network and target Q network
        input_size = action_size + observation_size
        self.Q = RNNS[cell](
            input_size=input_size,
            output_size=action_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
        )
        self.Q_tar = RNNS[cell](
            input_size=input_size,
            output_size=action_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
        )
        
        # Move to CUDA if enabled
        if cuda_enabled and torch.cuda.is_available():
            self.Q = self.Q.cuda()
            self.Q_tar = self.Q_tar.cuda()
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        
        # Initialize optimizers
        self.optimizer = torch.optim.Adam(self.Q.parameters(), lr=learning_rate)
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        
        # Initialize hidden states
        self.hidden_state = None
        self.target_hidden_state = None
        
        # Copy parameters to target network
        self.update_target_network()

    def update_target_network(self):
        """Copy parameters from main network to target network"""
        self.Q_tar.load_state_dict(self.Q.state_dict())

    def get_action(self, observation, hidden_state=None, epsilon=None):
        """
        Get action using epsilon-greedy policy with DRQN
        """
        if epsilon is None:
            epsilon = self.epsilon
            
        if hidden_state is None:
            hidden_state = self.reset_hidden(1)
            
        # Epsilon-greedy action selection
        if random() < epsilon:
            action = np.random.randint(self.action_size)
            new_hidden = hidden_state  # keep hidden state unchanged on random action
        else:
            with torch.no_grad():
                # Prepare input (observation + previous action)
                prev_action = torch.zeros(self.action_size, device=self.device)
                if hasattr(self, 'last_action') and self.last_action is not None:
                    prev_action[self.last_action] = 1.0
                    
                obs_tensor = torch.FloatTensor(observation).to(self.device)
                input_tensor = torch.cat([prev_action, obs_tensor]).unsqueeze(0).unsqueeze(0)
                
                # Forward pass
                q_values, new_hidden = self.Q(input_tensor, hidden_state)
                action = torch.argmax(q_values, dim=-1).item()
                
        self.last_action = action
        # Return the updated hidden state so callers can propagate it
        return action, new_hidden

    def train_step(self, batch_data):
        """
        Perform one training step using DRQN algorithm
        """
        if len(self.replay_buffer) < 100:  # Wait for enough samples
            return 0.0
            
        # Sample batch from replay buffer
        batch = self.replay_buffer.sample(batch_size=32, sequence_length=self.sequence_length)
        
        # Convert to tensors efficiently (batch processing)
        obs_tensor = torch.FloatTensor(batch['observations']).to(self.device)  # (batch_size, seq_len, obs_dim)
        action_tensor = torch.LongTensor(batch['actions']).to(self.device)     # (batch_size, seq_len)
        reward_tensor = torch.FloatTensor(batch['rewards']).to(self.device)    # (batch_size, seq_len)
        done_tensor = torch.BoolTensor(batch['dones']).to(self.device)         # (batch_size, seq_len)
        lengths = batch['lengths']  # (batch_size,)
        
        batch_size, max_seq_len = obs_tensor.shape[:2]
        total_loss = 0.0
        valid_samples = 0
        
        for i in range(batch_size):
            seq_len = lengths[i]
            if seq_len < 2:
                continue
                
            # Extract sequence for this sample
            obs_seq = obs_tensor[i, :seq_len]  # (seq_len, obs_dim)
            action_seq = action_tensor[i, :seq_len]  # (seq_len,)
            reward_seq = reward_tensor[i, :seq_len]  # (seq_len,)
            done_seq = done_tensor[i, :seq_len]  # (seq_len,)
                
            # Prepare input sequences efficiently
            input_seq = torch.zeros(seq_len, self.action_size + obs_seq.shape[1], device=self.device)
            for t in range(seq_len):
                if t > 0:
                    input_seq[t, action_seq[t-1]] = 1.0  # One-hot previous action
                input_seq[t, self.action_size:] = obs_seq[t]  # Current observation
                
            input_seq = input_seq.unsqueeze(1)  # (seq_len, 1, input_size)
            
            # Forward pass through Q network
            q_values, _ = self.Q(input_seq, None)
            q_values = q_values.squeeze(1)  # (seq_len, action_size)
            
            # Forward pass through target network
            with torch.no_grad():
                target_q_values, _ = self.Q_tar(input_seq, None)
                target_q_values = target_q_values.squeeze(1)
                
            # Calculate targets
            targets = q_values.clone()
            for t in range(seq_len - 1):
                action = action_seq[t]
                reward = reward_seq[t]
                done = done_seq[t]
                
                if done:
                    target_value = reward
                else:
                    target_value = reward + self.gamma * torch.max(target_q_values[t + 1])
                    
                targets[t, action] = target_value
                
            # Calculate loss
            loss = F.mse_loss(q_values[:-1], targets[:-1])
            total_loss += loss.item()
            valid_samples += 1
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.Q.parameters(), 1.0)
            self.optimizer.step()
        
        # Update target network periodically
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.update_target_network()
            
        return total_loss / valid_samples if valid_samples > 0 else 0.0

    def store_trajectory(self, observations, actions, rewards, dones):
        """Store trajectory in replay buffer"""
        trajectory = Trajectory(observations, actions, rewards, dones)
        self.replay_buffer.store(trajectory)

    def reset_hidden(self, batch_size=1):
        """Reset hidden state"""
        if self.cell == 'gru':
            hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=self.device)
            return (hidden,)
        elif self.cell == 'lstm':
            hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=self.device)
            cell = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=self.device)
            return (hidden, cell)

    def save_model(self, path):
        """Save model state dict"""
        torch.save({
            'Q_state_dict': self.Q.state_dict(),
            'Q_tar_state_dict': self.Q_tar.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

    def load_model(self, path):
        """Load model state dict"""
        checkpoint = torch.load(path)
        self.Q.load_state_dict(checkpoint['Q_state_dict'])
        self.Q_tar.load_state_dict(checkpoint['Q_tar_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def eval_mode(self):
        """Set model to evaluation mode"""
        self.Q.eval()
        self.Q_tar.eval()

    def train_mode(self):
        """Set model to training mode"""
        self.Q.train()
        self.Q_tar.train()
    
    def train_trajectory(self, trajectory):
        """
        Train DRQN using trajectory data for Two-Step MDP
        
        Args:
            trajectory: dict with 'observations', 'actions', 'rewards', 'next_observations', 'dones'
        """
        # Store trajectory in replay buffer
        self.store_trajectory(
            trajectory['observations'],
            trajectory['actions'], 
            trajectory['rewards'],
            trajectory['dones']
        )
        
        # Perform training step if enough data
        if len(self.replay_buffer) >= 100:
            return self.train_step({})
        else:
            return 0.0
