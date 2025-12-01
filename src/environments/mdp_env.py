"""
MDP Two-Step Task Environment (event-driven)
Stage 1: show s1 until agent picks a1
Stage 2: show s2 (B1/B2) until agent picks a2
Then deliver reward and terminate.

Transition parameter `trans_prob` is the probability of an undesired transition:
- If a1 selects the path toward B1, go to B2 with probability trans_prob (undesired),
  otherwise to B1 with probability 1 - trans_prob (desired). Symmetric for selecting B2.
"""

import copy
import numpy as np
from .base_env import BaseEnvironment


class TwoStepTask:
    """Event-driven Two-Step Task with minimal observations.

    Observation channels (n_input = 3): [s1, s2_B1, s2_B2]
    Phases:
      - phase = 0: show s1 (1,0,0) until agent picks a1 in {0,1}
      - phase = 1: show s2 one-hot depending on transition; wait for a2 in {0,1}
      - phase = 2: terminal; reward exposed via step() return
    """

    def __init__(self, s1_duration=1, s2_duration=1, trans_prob=0.2, reward_prob=0.8, **kwargs):
        # durations kept for compatibility but unused
        self.n_input = 3

        # Task parameters
        self.trans_prob = trans_prob  # probability of undesired transition
        self.reward_prob = reward_prob  # probability of reward in high-reward stage
        self.block_size = 50

        # Trial state
        self.block = 0
        self.phase = 0
        self.stage2 = None  # 1: B1, 2: B2
        self.choice1 = None
        self.choice2 = None
        self.reward = None
        self.completed = False
        self.common = None  # whether transition followed desired mapping
        self.choice = None  # backward-compat alias for choice1

    def reset(self):
        self.phase = 0
        self.stage2 = None
        self.choice1 = None
        self.choice2 = None
        self.reward = None
        self.completed = False
        self.common = None
        self.choice = None
        return np.array([1.0, 0.0, 0.0], dtype=float)  # s1

    def configure(self, trial_data, trial_settings):
        # Configure block; ignore any durations or precomputed sequences
        self.block = trial_settings.get('block', 0)
        return self.reset()

    def _observe(self):
        if self.phase == 0:
            return np.array([1.0, 0.0, 0.0], dtype=float)
        elif self.phase == 1:
            if self.stage2 == 1:
                return np.array([0.0, 1.0, 0.0], dtype=float)
            elif self.stage2 == 2:
                return np.array([0.0, 0.0, 1.0], dtype=float)
            else:
                return np.zeros(self.n_input, dtype=float)
        else:
            return np.zeros(self.n_input, dtype=float)

    def _sample_stage2(self, choice1):
        # Desired mapping: choice1==0 prefers B1, choice1==1 prefers B2
        undesired = np.random.rand() < self.trans_prob
        if choice1 == 0:
            self.common = not undesired
            return 2 if undesired else 1
        else:
            self.common = not undesired
            return 1 if undesired else 2

    def _sample_reward(self):
        # block 0: B1 high reward, block 1: B2 high reward
        if self.stage2 == 1:  # B1
            p = self.reward_prob if self.block == 0 else (1 - self.reward_prob)
        else:  # B2
            p = (1 - self.reward_prob) if self.block == 0 else self.reward_prob
        return 1 if np.random.rand() < p else 0

    def step(self, action):
        if self.phase == 0:
            # Expect a1 in {0,1}
            if action in (0, 1):
                self.choice1 = action
                self.choice = action
                self.stage2 = self._sample_stage2(self.choice1)
                self.phase = 1
                obs = self._observe()
                return obs, 0.0, False, {'phase': self.phase}
            else:
                # ignore invalid action, keep s1
                return self._observe(), 0.0, False, {'phase': self.phase}
        elif self.phase == 1:
            # Expect a2 in {0,1}
            if action in (0, 1):
                self.choice2 = action
                self.reward = self._sample_reward()
                self.phase = 2
                self.completed = True
                return self._observe(), float(self.reward), True, {'phase': self.phase}
            else:
                return self._observe(), 0.0, False, {'phase': self.phase}
        else:
            # Terminal
            return self._observe(), 0.0, True, {'phase': self.phase}
        
    
    
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
        self.observation_size = self.task.n_input
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
        self.current_observation = self.task.reset()
        return self.current_observation
        
    def configure_trial(self, trial_data, trial_settings):
        """Configure environment with trial data"""
        self.done = False
        self.episode_reward = 0
        self.current_observation = self.task.configure(trial_data, trial_settings)
        return self.current_observation
    
    def step(self, action):
        """Take step in environment"""
        if self.done:
            return self.current_observation, 0.0, True, {}
            
        # Execute step in task
        next_observation, reward, self.done, info = self.task.step(action)
        self.current_observation = np.array(next_observation)
        self.episode_reward += reward
        info.update({'episode_reward': self.episode_reward})
        return self.current_observation, reward, self.done, info
    
    def get_observation_space(self):
        """Get observation space size"""
        return self.observation_size
    
    def get_action_space(self):
        """Get action space size"""
        return self.action_size
    
    def is_winning(self):
        """Success if agent reached the high-reward stage, regardless of sampled reward.

        Block 0: target stage is B1 (stage2 == 1)
        Block 1: target stage is B2 (stage2 == 2)
        """
        if self.task.stage2 is None:
            return 0
        target_stage = 1 if self.task.block == 0 else 2
        return 1 if self.task.stage2 == target_stage else 0
    
    def is_completed(self):
        """Check if current trial is completed"""
        return 1 if self.task.completed else 0
        
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
        """No-op generator retained for compatibility; environment is event-driven now."""
        trials = [[None] for _ in range(num_trials)]
        conditions = {'training_guide': [[True, 0, 0] for _ in range(num_trials)]}
        return trials, conditions
