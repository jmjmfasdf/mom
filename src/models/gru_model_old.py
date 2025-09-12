"""
GRU-based RL Agent

Adapted from Sequence_learning-master project.
Maintains original logic while implementing unified interface.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from torch.autograd import Variable

from .base_agent import BaseAgent


class GRUNetwork(nn.Module):
    """
    GRU network model (from original Sequence_learning project)
    """

    def __init__(self, ninp, nhid, bsz, nlayers, lr, cuda_enabled=False):
        super(GRUNetwork, self).__init__()
        
        self.lr = lr
        self.rnn = nn.GRU(ninp, nhid, num_layers=nlayers)
        self.ninp = ninp
        self.nhid = nhid
        self.nlayers = nlayers
        self.batch_size = bsz
        self.cuda_enabled = cuda_enabled

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
                        
        self.decoder = nn.Linear(nhid, ninp)  # output_size == input_size
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.criterion = nn.MSELoss(reduction='mean')

        if self.cuda_enabled:
            self.init_hidden = self.init_hidden_gpu
            self.cuda()
        else:
            self.init_hidden = self.init_hidden_cpu

    def forward(self, input, hidden, bsz=None):
        """
        inputs shape: seq_len, batch, input_size
        outputs shape; seq_len, batch, input_size
        """
        bsz = self.batch_size if bsz is None else bsz
        output, hidden = self.rnn(input, hidden)
        hidden = self.relu(hidden)
        output = self.relu(output)
        
        # Reshape output for decoder: (seq_len, batch, hidden) -> (seq_len * batch, hidden)
        seq_len, batch_size, hidden_size = output.size()
        output = output.view(-1, hidden_size)
        output = self.decoder(output)
        output = self.sigmoid(output)
        
        # Reshape back to (seq_len, batch, input_size)
        output = output.view(seq_len, batch_size, -1)
        return output, hidden

    def init_hidden_cpu(self, bsz):
        weight = next(self.parameters()).data
        return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())
        
    def init_hidden_gpu(self, bsz):
        weight = next(self.parameters()).data
        return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_().cuda())


class Calculator:
    """Calculator class from original project for loss computation"""
    
    def __init__(self, batch_size=1, cuda_enabled=False):
        self.cuda_enabled = cuda_enabled
        self.batch_size = batch_size

    def get_output(self, model, trial_data, hidden, tmp_reward, bsz=None, tau=np.inf):
        """
        Calculate output tensor for specific input (simplified version)
        """
        if bsz is None:
            bsz = self.batch_size
            
        total_loss = 0
        criterion = model.criterion
        prediction_trial_length = trial_data.shape[0] - 1
        
        pred_trial = np.zeros((prediction_trial_length, bsz, model.ninp))
        
        # Convert to torch tensors and ensure 3D shape (seq_len, batch_size, input_size)
        if len(trial_data.shape) == 2:
            # If 2D, add batch dimension: (seq_len, input_size) -> (seq_len, 1, input_size)
            trial_data = np.expand_dims(trial_data, 1)
        
        # Initialize hidden state with correct size
        if hidden is None or hidden.size(1) != trial_data.shape[1]:
            if model.cuda_enabled:
                hidden = model.init_hidden_cuda(trial_data.shape[1])
            else:
                hidden = model.init_hidden_cpu(trial_data.shape[1])
        
        # Convert trial data to tensor once
        if self.cuda_enabled:
            trial_tensor = Variable(torch.FloatTensor(trial_data).cuda())
        else:
            trial_tensor = Variable(torch.FloatTensor(trial_data))
        
        # Process each timestep individually
        for i in range(prediction_trial_length):
            # Get input for this timestep
            inputs = trial_tensor[i:i+1]  # (1, batch_size, input_size)
            
            # Forward pass for this timestep
            output, hidden = model(inputs, hidden, bsz)
            
            # Calculate loss for this timestep if training is needed
            if tmp_reward[0][0] and tmp_reward[0][1] <= i < tmp_reward[0][2]:
                target = trial_tensor[i+1]  # (batch_size, input_size)
                target = target.unsqueeze(0)  # (1, batch_size, input_size)
                loss = criterion(output, target)
                total_loss += loss
            
            # Store prediction
            if self.cuda_enabled:
                pred_trial[i] = output.data.cpu().numpy()
            else:
                pred_trial[i] = output.data.numpy()
            
        return pred_trial, total_loss, hidden


class GRUAgent(BaseAgent):
    """
    GRU-based RL Agent implementing unified interface
    """
    
    def __init__(self, observation_size, action_size, hidden_size=128, num_layers=1, 
                 learning_rate=1e-3, batch_size=1, cuda_enabled=False):
        super().__init__(observation_size, action_size, hidden_size, num_layers)
        
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.cuda_enabled = cuda_enabled
        
        # Initialize GRU network
        self.model = GRUNetwork(
            ninp=observation_size,
            nhid=hidden_size,
            bsz=batch_size,
            nlayers=num_layers,
            lr=learning_rate,
            cuda_enabled=cuda_enabled
        )
        
        # Initialize calculator for training
        self.calculator = Calculator(batch_size, cuda_enabled)
        
        # Action selection network (maps hidden state to actions)
        self.action_net = nn.Linear(hidden_size, action_size)
        if cuda_enabled:
            self.action_net.cuda()
            
        self.hidden_state = None
        
    def get_action(self, observation, hidden_state=None, epsilon=0.1):
        """
        Get action from observation using epsilon-greedy policy
        """
        if hidden_state is None:
            hidden_state = self.model.init_hidden(1)
            
        # Convert observation to tensor
        if self.cuda_enabled:
            obs_tensor = Variable(torch.FloatTensor(observation).view(1, 1, -1).cuda())
        else:
            obs_tensor = Variable(torch.FloatTensor(observation).view(1, 1, -1))
            
        # Forward pass through GRU
        with torch.no_grad():
            gru_out, new_hidden = self.model.rnn(obs_tensor, hidden_state)
            action_logits = self.action_net(gru_out.view(1, -1))
            
        # Epsilon-greedy action selection
        if np.random.random() < epsilon:
            action = np.random.randint(self.action_size)
        else:
            action = torch.argmax(action_logits, dim=1).item()
            
        return action, new_hidden
    
    def train_step(self, batch_data):
        """
        Perform one training step using original GRU training logic
        """
        trial_data = batch_data['observations']  # shape: (seq_len, batch_size, obs_size)
        rewards = batch_data['rewards']  # training guide from original project
        
        if self.hidden_state is None:
            self.hidden_state = self.model.init_hidden(self.batch_size)
            
        # Use original calculator logic
        pred_trial, total_loss, self.hidden_state = self.calculator.get_output(
            self.model, trial_data, self.hidden_state, rewards, self.batch_size
        )
        
        # Backward pass
        if isinstance(total_loss, torch.Tensor) and total_loss.requires_grad and total_loss.item() > 0:
            self.model.optimizer.zero_grad()
            try:
                total_loss.backward(retain_graph=False)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)
                self.model.optimizer.step()
            except RuntimeError as e:
                if "backward through the graph a second time" in str(e):
                    # Skip backward pass if already done
                    pass
                else:
                    raise e
            
        return total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss
    
    def reset_hidden(self, batch_size=None):
        """Reset hidden state"""
        if batch_size is None:
            batch_size = self.batch_size
        self.hidden_state = self.model.init_hidden(batch_size)
        return self.hidden_state
    
    def save_model(self, path):
        """Save model state dict"""
        torch.save(self.model.state_dict(), path)
        
    def load_model(self, path):
        """Load model state dict"""
        self.model.load_state_dict(torch.load(path))
        
    def eval_mode(self):
        """Set model to evaluation mode"""
        self.model.eval()
        
    def train_mode(self):
        """Set model to training mode"""
        self.model.train()
    
    def train_trajectory(self, obs_sequence, action_sequence, reward_sequence):
        """
        Train GRU using trajectory data for T-maze (Q-learning style)
        
        Args:
            obs_sequence: (seq_len, 1, obs_size) tensor
            action_sequence: (seq_len,) tensor
            reward_sequence: (seq_len,) tensor
        """
        if self.cuda_enabled:
            obs_sequence = obs_sequence.cuda()
            action_sequence = action_sequence.cuda()
            reward_sequence = reward_sequence.cuda()
        
        # Forward pass through GRU
        hidden = self.model.init_hidden(1)
        gru_output, _ = self.model.rnn(obs_sequence, hidden)
        
        # Get Q-values for all timesteps
        q_values = self.action_net(gru_output.view(-1, self.model.nhid))  # (seq_len, action_size)
        
        # Get Q-values for taken actions
        action_q_values = q_values.gather(1, action_sequence.view(-1, 1)).squeeze()
        
        # Calculate target Q-values (simple reward-based targets)
        with torch.no_grad():
            # For T-maze, use cumulative discounted reward as target
            targets = torch.zeros_like(action_q_values)
            gamma = 0.98
            discounted_reward = 0
            
            # Backward pass to calculate discounted rewards
            for i in reversed(range(len(reward_sequence))):
                discounted_reward = reward_sequence[i] + gamma * discounted_reward
                targets[i] = discounted_reward
        
        # Compute loss and update
        loss = F.mse_loss(action_q_values, targets)
        
        self.model.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)
        self.model.optimizer.step()
        
        return loss.item()
