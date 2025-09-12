"""
MDP Task Environment

Adapted from Sequence_learning-master project.
Maintains original logic while implementing unified interface.
"""

import copy
import numpy as np
from .base_env import BaseEnvironment


class TwoStepTask:
    """
    Two-Step Task (from original generatorST.py, single trial version)
    Stage 1: Choice between A1/A2
    Stage 2: Outcome B1/B2 based on transition probabilities
    """
    
    def __init__(self, s1_duration=3, s2_duration=3, trans_prob=0.8, reward_prob=0.8, **kwargs):
        # Timing parameters (customizable)
        self.s1_duration = s1_duration  # Duration of stage 1 stimulus
        self.s2_duration = s2_duration  # Duration of stage 2 stimulus  
        self.choice_duration = 2  # Duration for choice period
        self.reward_duration = 2  # Duration for reward period
        
        # Calculate trial structure based on durations
        self.fixation_start = 0
        self.fixation_end = 2  # 2 timesteps for fixation
        self.s1_start = self.fixation_end
        self.s1_end = self.s1_start + self.s1_duration
        self.choice_start = self.s1_end  
        self.choice_end = self.choice_start + self.choice_duration
        self.s2_start = self.choice_end
        self.s2_end = self.s2_start + self.s2_duration
        self.reward_start = self.s2_end
        self.reward_end = self.reward_start + self.reward_duration
        
        self.trial_length = self.reward_end
        self.n_input = 10  # From original
        
        # Task parameters
        self.trans_prob = trans_prob  # A1->B1, A2->B2 probability
        self.reward_prob = reward_prob  # B1/B2 -> reward probability
        self.block_size = 50
        
        # Task state
        self.completed = None
        self.trial_end = None
        self.reward = None
        self.choice = None  # 0: A1, 1: A2
        self.stage2 = None  # 1: B1, 2: B2
        self.chosen = None
        self.common = None  # Whether transition was common
        
        self.time_step = -1
        self.block = None  # Current block (0: B1 high reward, 1: B2 high reward)
        
        # Action timing
        self.choice_period = list(range(self.choice_start, self.choice_end))  # When choice can be made
        
    def configure(self, trial_data, trial_settings):
        """Configure trial with generatorST.py logic"""
        self.states_pool = trial_data[0].tolist()
        
        # Get trial parameters from settings
        self.block = trial_settings.get('block', 0)
        self.choice = trial_settings.get('choice', np.random.choice([0, 1]))
        
        # Calculate stage 2 based on choice and transition probability
        trans_prob_actual = self.trans_prob if self.choice == 0 else (1 - self.trans_prob)
        self.stage2 = 1 if trans_prob_actual > np.random.rand() else 2
        
        # Calculate reward probability based on block and stage2
        if self.stage2 == 1:  # B1
            reward_prob_actual = self.reward_prob if self.block == 0 else (1 - self.reward_prob)
        else:  # B2
            reward_prob_actual = (1 - self.reward_prob) if self.block == 0 else self.reward_prob
            
        # Determine reward
        self.reward = 1 if reward_prob_actual > np.random.rand() else 0
        
        # Check if transition was common
        self.common = (self.choice == 0 and self.stage2 == 1) or (self.choice == 1 and self.stage2 == 2)
        
        self.trial_end = False
        self.time_step = -1
        self.completed = None
        self.chosen = False
        
    def step(self, action):
        """Execute one step with generatorST.py structure"""
        self.time_step += 1
        
        if self.time_step < len(self.states_pool):
            next_sensory_inputs = self.states_pool[self.time_step]
            
            # Check if it's choice time and choice hasn't been made
            if self.time_step in self.choice_period and not self.chosen:
                if action in (0, 1):  # Valid choice (A1 or A2)
                    self.chosen = True
                    # Update the states_pool to reflect the choice and outcome
                    self._update_states_with_outcome()
                else:
                    # Invalid choice - trial fails
                    self.reward = 0
                    
            # Check if trial is complete
            if self.time_step >= self.trial_length - 1:
                self.trial_end = True
                self.completed = True
                
            return self.trial_end, next_sensory_inputs
            
        return True, np.zeros(self.n_input)
    
    def _update_states_with_outcome(self):
        """Update states_pool with choice outcome (simplified version of generatorST.py logic)"""
        # This would update the states_pool to show:
        # - Choice made (time steps 5-6)
        # - Stage 2 outcome (time steps 7-9) 
        # - Reward (time steps 10-11)
        # For simplicity, we assume states_pool is already properly formatted
        pass
        
    def is_winning(self):
        """Check if won"""
        return 1 if self.reward else 0
        
    def is_completed(self):
        """Check if completed"""
        return 1 if self.completed else 0
        
    def extract_trial_abstract(self):
        """Extract trial info"""
        return {
            "choice": self.choice,
            "stage2": self.stage2,
            "reward": self.reward,
            "common": self.common,
            "block": self.block,
            "completed": self.completed,
        }




class MDPEnvironment(BaseEnvironment):
    """
    MDP Task Environment - Two-Step Task Only
    """
    
    def __init__(self, **kwargs):
        # Only support two-step task now
        self.task = TwoStepTask(**kwargs)
        self.observation_size = 10  # From generatorST.py
        self.action_size = 2  # 0: A1, 1: A2
            
        super().__init__(self.observation_size, self.action_size)
        
        # Current state
        self.current_observation = None
        self.done = False
        self.episode_reward = 0
        
    def reset(self):
        """Reset environment"""
        # This would typically be called with trial data from a generator
        # For now, return a dummy observation
        self.done = False
        self.episode_reward = 0
        self.current_observation = np.zeros(self.observation_size)
        return self.current_observation
        
    def configure_trial(self, trial_data, trial_settings):
        """Configure environment with trial data"""
        self.task.configure(trial_data, trial_settings)
        self.done = False
        self.episode_reward = 0
        
        # Get initial observation
        if len(self.task.states_pool) > 0:
            self.current_observation = np.array(self.task.states_pool[0])
        else:
            self.current_observation = np.zeros(self.observation_size)
            
        return self.current_observation
    
    def step(self, action):
        """Take step in environment"""
        if self.done:
            return self.current_observation, 0.0, True, {}
            
        # Execute step in task
        trial_end, next_observation = self.task.step(action)
        
        # Update state
        self.current_observation = np.array(next_observation)
        
        # Get reward
        reward = 0.0
        if self.task.reward is not None:
            reward = float(self.task.reward)
            self.episode_reward += reward
            
        # Check if done
        self.done = trial_end
        
        # Prepare info
        info = {
            'task_info': self.task.extract_trial_abstract() if hasattr(self.task, 'extract_trial_abstract') else {},
            'episode_reward': self.episode_reward
        }
        
        return self.current_observation, reward, self.done, info
    
    def get_observation_space(self):
        """Get observation space size"""
        return self.observation_size
    
    def get_action_space(self):
        """Get action space size"""
        return self.action_size
    
    def is_winning(self):
        """Check if current trial is winning"""
        return self.task.is_winning()
    
    def is_completed(self):
        """Check if current trial is completed"""
        return self.task.is_completed()
        
    def render(self):
        """Render current state"""
        print(f"Two-Step Task, Reward: {self.episode_reward}, Done: {self.done}")
        if hasattr(self.task, 'choice') and self.task.choice is not None:
            print(f"Choice made: {self.task.choice}, Stage2: {self.task.stage2}, Reward: {self.task.reward}")
            print(f"Common transition: {self.task.common}, Block: {self.task.block}")


# Task generator for two-step task
class TaskGenerator:
    """Task generator for two-step task"""
    
    def __init__(self):
        pass
        
    def generate(self, num_trials=1, s1_duration=3, s2_duration=3):
        """Generate trials for the two-step task"""
        trials = []
        conditions = []
        
        for _ in range(num_trials):
            # Calculate trial length based on durations
            trial_length = 2 + s1_duration + 2 + s2_duration + 2  # fixation + s1 + choice + s2 + reward
            
            # Generate dummy two-step trial with proper length
            trial_data = [np.random.random((trial_length, 10))]  # 10 inputs from original
            condition = {
                'block': np.random.choice([0, 1]),  # 0: B1 high reward, 1: B2 high reward
                'choice': np.random.choice([0, 1]),  # 0: A1, 1: A2
                'training_guide': [True, 0, trial_length-1]
            }
                
            trials.append(trial_data)
            conditions.append(condition)
            
        return trials, {'training_guide': [c['training_guide'] for c in conditions]}
