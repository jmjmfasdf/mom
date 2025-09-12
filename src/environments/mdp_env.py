"""
MDP Task Environment - 5-Slot Structure
s1 → a1 → s2 → a2 → reward

Adapted from Sequence_learning-master project.
Maintains original logic while implementing unified interface.
"""

import copy
import numpy as np
from .base_env import BaseEnvironment


class TwoStepTask:
    """
    Two-Step Task with 5-slot structure
    Slot 0: s1 (Stage 1 stimulus)
    Slot 1: a1 (Stage 1 action) 
    Slot 2: s2 (Stage 2 stimulus)
    Slot 3: a2 (Stage 2 action)
    Slot 4: reward
    """
    
    def __init__(self, s1_duration=1, s2_duration=1, trans_prob=0.8, reward_prob=0.8, **kwargs):
        # Fixed 5-slot structure (s1_duration and s2_duration kept for compatibility)
        self.trial_length = 5
        self.n_input = 10
        
        # Task parameters
        self.trans_prob = trans_prob  # A1->B1, A2->B2 probability
        self.reward_prob = reward_prob  # B1/B2 -> reward probability
        self.block_size = 50
        
        # Task state
        self.completed = None
        self.trial_end = None
        self.reward = None
        self.choice1 = None  # 0: A1, 1: A2 (Stage 1 choice)
        self.choice2 = None  # 0: A1, 1: A2 (Stage 2 choice)
        self.stage2 = None  # 1: B1, 2: B2
        self.common = None  # Whether transition was common
        
        self.time_step = -1
        self.block = None  # Current block (0: B1 high reward, 1: B2 high reward)
        
        # Action timing - only at slots 1 and 3
        self.action_slots = [1, 3]
        
        # Compatibility attributes
        self.s1_duration = s1_duration
        self.s2_duration = s2_duration
        self.choice = None  # For backward compatibility
        self.chosen = None
        
    def configure(self, trial_data, trial_settings):
        """Configure trial with new structure"""
        self.states_pool = trial_data[0].tolist()
        
        # Get trial parameters from settings
        self.block = trial_settings.get('block', 0)
        self.choice1 = trial_settings.get('choice1', trial_settings.get('choice', np.random.choice([0, 1])))
        
        # Calculate stage 2 based on choice1 and transition probability
        trans_prob_actual = self.trans_prob if self.choice1 == 0 else (1 - self.trans_prob)
        self.stage2 = 1 if trans_prob_actual > np.random.rand() else 2
        
        # Calculate reward probability based on block and stage2
        if self.stage2 == 1:  # B1
            reward_prob_actual = self.reward_prob if self.block == 0 else (1 - self.reward_prob)
        else:  # B2
            reward_prob_actual = (1 - self.reward_prob) if self.block == 0 else self.reward_prob
            
        # Determine reward
        self.reward = 1 if reward_prob_actual > np.random.rand() else 0
        
        # Check if transition was common
        self.common = (self.choice1 == 0 and self.stage2 == 1) or (self.choice1 == 1 and self.stage2 == 2)
        
        self.trial_end = False
        self.time_step = -1
        self.completed = None
        self.choice2 = None
        self.chosen = False
        
        # Backward compatibility
        self.choice = self.choice1
        
    def step(self, action):
        """Execute one step with 5-slot structure"""
        self.time_step += 1
        
        if self.time_step < len(self.states_pool):
            next_sensory_inputs = self.states_pool[self.time_step]
            
            # Check if it's action time
            if self.time_step in self.action_slots:
                if self.time_step == 1:  # Stage 1 action
                    if action in (0, 1):  # Valid choice (A1 or A2)
                        self.choice1 = action
                        self.chosen = True
                        # Backward compatibility
                        self.choice = self.choice1
                    else:
                        # Invalid choice - trial fails
                        self.reward = 0
                elif self.time_step == 3:  # Stage 2 action
                    if action in (0, 1):  # Valid choice (A1 or A2)
                        self.choice2 = action
                    else:
                        # Invalid choice - trial fails
                        self.reward = 0
                        
            # Check if trial is complete
            if self.time_step >= self.trial_length - 1:
                self.trial_end = True
                self.completed = True
                
            return self.trial_end, next_sensory_inputs
            
        return True, np.zeros(self.n_input)
    
    def is_winning(self):
        """Check if won"""
        return 1 if self.reward else 0
        
    def is_completed(self):
        """Check if completed"""
        return 1 if self.completed else 0
        
    def extract_trial_abstract(self):
        """Extract trial info"""
        return {
            "choice1": self.choice1,
            "choice2": self.choice2,
            "choice": self.choice,  # Backward compatibility
            "stage2": self.stage2,
            "reward": self.reward,
            "common": self.common,
            "block": self.block,
            "completed": self.completed,
        }


class MDPEnvironment(BaseEnvironment):
    """
    MDP Task Environment - 5-Slot Structure
    """
    
    def __init__(self, **kwargs):
        # Only support two-step task
        self.task = TwoStepTask(**kwargs)
        self.observation_size = 10
        self.action_size = 2  # 0: A1, 1: A2
            
        super().__init__(self.observation_size, self.action_size)
        
        # Current state
        self.current_observation = None
        self.done = False
        self.episode_reward = 0
        
    def reset(self):
        """Reset environment"""
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
        
        # Get reward (only at slot 4)
        reward = 0.0
        if self.task.reward is not None and self.task.time_step == 4:
            reward = float(self.task.reward)
            self.episode_reward += reward
            
        # Check if done
        self.done = trial_end
        
        # Prepare info
        info = {
            'task_info': self.task.extract_trial_abstract() if hasattr(self.task, 'extract_trial_abstract') else {},
            'episode_reward': self.episode_reward,
            'time_step': self.task.time_step
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
        if hasattr(self.task, 'choice1') and self.task.choice1 is not None:
            print(f"Choice1: {self.task.choice1}, Choice2: {self.task.choice2}")
            print(f"Stage2: {self.task.stage2}, Reward: {self.task.reward}")
            print(f"Common transition: {self.task.common}, Block: {self.task.block}")


# Task generator for two-step task
class TaskGenerator:
    """Task generator for two-step task"""
    
    def __init__(self):
        pass
        
    def generate(self, num_trials=1, s1_duration=1, s2_duration=1):
        """Generate trials for the two-step task"""
        trials = []
        conditions = []
        
        for _ in range(num_trials):
            # Fixed 5-slot trial length
            trial_length = 5
            
            # Generate dummy two-step trial with proper length
            trial_data = [np.random.random((trial_length, 10))]  # 10 inputs
            condition = {
                'block': np.random.choice([0, 1]),  # 0: B1 high reward, 1: B2 high reward
                'choice1': np.random.choice([0, 1]),  # 0: A1, 1: A2
                'choice': np.random.choice([0, 1]),  # Backward compatibility
                'training_guide': [True, 0, trial_length-1]
            }
                
            trials.append(trial_data)
            conditions.append(condition)
            
        return trials, {'training_guide': [c['training_guide'] for c in conditions]}