"""
GRU Model - Based on original Sequence_learning-master implementation
Adapted for 5-slot MDP environment with batch processing support
"""

import copy
import time
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

from .base_agent import BaseAgent


def cons_np2c(bsz, cuda=False):
    """Convert numpy to torch and vice versa with proper batching"""
    if cuda:
        def func(state_vector, require_g=True):
            return Variable(torch.Tensor(state_vector).type(torch.FloatTensor).view(1, bsz, -1).cuda(),
                            requires_grad=require_g)

        def func2(t_variable):
            return t_variable.data.cpu().numpy()
    else:
        def func(state_vector, require_g=True):
            return Variable(torch.Tensor(state_vector).type(torch.FloatTensor).view(1, bsz, -1),
                            requires_grad=require_g)

        def func2(t_variable):
            return t_variable.data.numpy()

    return func, func2


class GRUNetwork(nn.Module):
    """
    GRU network model - Original implementation
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
        self.criterion = nn.MSELoss(reduce=True)

        if self.cuda_enabled:
            torch.cuda.manual_seed_all(int(time.time()))
            self.init_hidden = self.init_hidden_gpu
            self.cuda()
        else:
            torch.manual_seed(int(time.time()))
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
        output = output.view(bsz, -1) 
        output = self.decoder(output)
        output = self.sigmoid(output)
        return output, hidden 
        
    def init_hidden_cpu(self, bsz=None):
        """Init the hidden cells' states before any trial."""
        bsz = self.batch_size if bsz is None else bsz
        weight = next(self.parameters()).data
        return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())

    def init_hidden_gpu(self, bsz=None):
        """Init the hidden cells' states before any trial."""
        bsz = self.batch_size if bsz is None else bsz
        weight = next(self.parameters()).data
        return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_().cuda())


class Calculator:
    """Calculator class from original project for loss computation"""
    
    def __init__(self, batch_size=1, cuda_enabled=False):
        self.cuda_enabled = cuda_enabled
        self.batch_size = batch_size
        self.np2t, self.t2np = cons_np2c(batch_size, self.cuda_enabled)

    def get_output(self, model, trial_data, hidden, tmp_reward, bsz=None, tau=np.inf):
        """
        Calculate output tensor for specific input (original implementation)
        """
        total_loss = 0
        criterion = model.criterion
        prediction_trial_length = trial_data.shape[0] - 1
        
        pred_trial = np.zeros((prediction_trial_length, model.batch_size, model.ninp))
        
        # Create reward guide for training segments
        reward_guide = np.tile(tmp_reward[:,0], (trial_data.shape[2],1))
        reward_guide = np.tile(reward_guide, (trial_data.shape[0],1,1)).transpose(0,2,1)
        for i in range(model.batch_size):
            reward_guide[:tmp_reward[i,1], i, :] = 0
            reward_guide[tmp_reward[i,2]:, i, :] = 0

        # Convert to torch Variable
        if self.cuda_enabled:
            nstep = Variable(torch.Tensor(np.array([0])).type(torch.FloatTensor).cuda(), requires_grad=False)
            reward_guide = Variable(torch.Tensor(reward_guide).type(torch.FloatTensor).cuda(), requires_grad=False)
        else:
            nstep = Variable(torch.Tensor(np.array([0])).type(torch.FloatTensor), requires_grad=False)
            reward_guide = Variable(torch.Tensor(reward_guide).type(torch.FloatTensor), requires_grad=False)
        
        # The time step when the hidden response is saved
        save_step = int(trial_data.shape[0]/2-1)
        
        for i in range(prediction_trial_length):
            inputs = self.np2t(trial_data[i, :, :])
            n_put = self.np2t(trial_data[i+1, :, :], require_g=False)
            output, hidden = model(inputs, hidden, bsz=bsz)
            
            if i == save_step:  # Save hidden state for next trial
                hidden_ = copy.deepcopy(Variable(hidden))
            
            n_put = n_put.reshape(bsz, -1)  # 1 by batch size by number of inputs
            
            if reward_guide[i].sum() == 0:
                loss = criterion(output*0.0, n_put*0.0)
            else:
                loss = criterion((output[reward_guide[i]==1]).reshape([1,-1])[0], 
                                (n_put[reward_guide[i]==1]).reshape([1,-1])[0])

            if not np.isnan(loss.item()):
                nstep += reward_guide[i].sum()
                total_loss = total_loss + loss*reward_guide[i].sum()
                
            raw_output = self.t2np(output)
            pred_trial[i, :, :] = np.around(raw_output)
        
        total_loss = total_loss/nstep if nstep != 0 else total_loss
        return pred_trial, total_loss, hidden_


class GRUAgent(BaseAgent):
    """
    GRU-based RL Agent implementing unified interface
    Based on original Sequence_learning-master implementation
    """
    
    def __init__(self, observation_size, action_size, config, cuda=False):
        super().__init__(observation_size, action_size, config)
        
        self.cuda = cuda
        self.batch_size = config.batch_size
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_layers
        self.learning_rate = config.learning_rate
        
        # Create GRU model
        self.model = GRUNetwork(
            ninp=observation_size,
            nhid=hidden_size,
            bsz=batch_size,
            nlayers=num_layers,
            lr=learning_rate,
            cuda_enabled=cuda
        )
        
        # Create calculator for loss computation
        self.calculator = Calculator(batch_size=batch_size, cuda_enabled=cuda)
        
        # Hidden state for each batch
        self.hidden_state = None
        
    def reset_hidden(self, batch_size=None):
        """Reset hidden state"""
        if batch_size is None:
            batch_size = self.batch_size
        self.hidden_state = self.model.init_hidden(batch_size)
        
    def get_action(self, observation, training=True):
        """Get action from observation (not used in GRU training)"""
        # GRU model doesn't use this method for action selection
        # It's trained on sequences, not individual actions
        return 0
        
    def train_step(self, batch_data):
        """Train step for GRU model"""
        trial_data = batch_data['observations']  # (seq_len, batch_size, input_size)
        rewards = batch_data['rewards']  # Training guide
        
        # Initialize hidden state if needed
        if self.hidden_state is None:
            self.hidden_state = self.model.init_hidden(self.batch_size)
            
        # Use original calculator logic
        pred_trial, total_loss, self.hidden_state = self.calculator.get_output(
            self.model, trial_data, self.hidden_state, rewards, self.batch_size
        )
        
        # Backward pass
        if isinstance(total_loss, torch.Tensor) and total_loss.requires_grad and total_loss.item() > 0:
            self.model.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)
            self.model.optimizer.step()
            
        return total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss
    
    def save_model(self, filepath):
        """Save model"""
        torch.save(self.model.state_dict(), filepath)
        
    def load_model(self, filepath):
        """Load model"""
        self.model.load_state_dict(torch.load(filepath))
