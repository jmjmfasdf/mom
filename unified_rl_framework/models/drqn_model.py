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
        trajectories = np.random.choice(self.buffer, batch_size, replace=False)
        
        batch_observations = []
        batch_actions = []
        batch_rewards = []
        batch_dones = []
        
        for trajectory in trajectories:
            # Sample random sequence from trajectory
            if len(trajectory.observations) > sequence_length:
                start_idx = np.random.randint(0, len(trajectory.observations) - sequence_length)
                end_idx = start_idx + sequence_length
            else:
                start_idx = 0
                end_idx = len(trajectory.observations)
                
            batch_observations.append(trajectory.observations[start_idx:end_idx])
            batch_actions.append(trajectory.actions[start_idx:end_idx])
            batch_rewards.append(trajectory.rewards[start_idx:end_idx])
            batch_dones.append(trajectory.dones[start_idx:end_idx])
        
        return {
            'observations': batch_observations,
            'actions': batch_actions,
            'rewards': batch_rewards,
            'dones': batch_dones
        }

    def __len__(self):
        return len(self.buffer)


class DRQNAgent(BaseAgent):
    """
    Deep Recurrent Q-Network reinforcement learning agent (from original project)
    """

    def __init__(self, observation_size, action_size, hidden_size=32, num_layers=2,
                 cell='gru', learning_rate=1e-3, gamma=0.98, epsilon=0.2,
                 buffer_capacity=8192, target_update_freq=10):
        super().__init__(observation_size, action_size, hidden_size, num_layers)
        
        self.cell = cell
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.target_update_freq = target_update_freq
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
        else:
            with torch.no_grad():
                # Prepare input (observation + previous action)
                prev_action = torch.zeros(self.action_size)
                if hasattr(self, 'last_action') and self.last_action is not None:
                    prev_action[self.last_action] = 1.0
                    
                obs_tensor = torch.FloatTensor(observation)
                input_tensor = torch.cat([prev_action, obs_tensor]).unsqueeze(0).unsqueeze(0)
                
                # Forward pass
                q_values, new_hidden = self.Q(input_tensor, hidden_state)
                action = torch.argmax(q_values, dim=-1).item()
                
        self.last_action = action
        return action, hidden_state

    def train_step(self, batch_data):
        """
        Perform one training step using DRQN algorithm
        """
        if len(self.replay_buffer) < 100:  # Wait for enough samples
            return 0.0
            
        # Sample batch from replay buffer
        batch = self.replay_buffer.sample(batch_size=32, sequence_length=20)
        
        total_loss = 0.0
        
        for i in range(len(batch['observations'])):
            obs_seq = torch.FloatTensor(batch['observations'][i])
            action_seq = torch.LongTensor(batch['actions'][i])
            reward_seq = torch.FloatTensor(batch['rewards'][i])
            done_seq = torch.BoolTensor(batch['dones'][i])
            
            seq_len = len(obs_seq)
            if seq_len < 2:
                continue
                
            # Prepare input sequences
            input_seq = []
            for t in range(seq_len):
                prev_action = torch.zeros(self.action_size)
                if t > 0:
                    prev_action[action_seq[t-1]] = 1.0
                input_t = torch.cat([prev_action, obs_seq[t]])
                input_seq.append(input_t)
                
            input_seq = torch.stack(input_seq).unsqueeze(1)  # (seq_len, 1, input_size)
            
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
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.Q.parameters(), 1.0)
            self.optimizer.step()
        
        # Update target network periodically
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.update_target_network()
            
        return total_loss / len(batch['observations']) if batch['observations'] else 0.0

    def store_trajectory(self, observations, actions, rewards, dones):
        """Store trajectory in replay buffer"""
        trajectory = Trajectory(observations, actions, rewards, dones)
        self.replay_buffer.store(trajectory)

    def reset_hidden(self, batch_size=1):
        """Reset hidden state"""
        if self.cell == 'gru':
            hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size)
            return (hidden,)
        elif self.cell == 'lstm':
            hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size)
            cell = torch.zeros(self.num_layers, batch_size, self.hidden_size)
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
