"""
MDP Two-Step Task Environment (event-driven, Daw 2011 style)

Stage 1: show s1 until agent picks a1
Stage 2: show s2 (B1/B2) until agent picks a2
Then deliver reward and terminate.

Transition parameter `trans_prob` is the probability of a rare (undesired) transition:
- If a1 selects the path toward B1, go to B2 with probability trans_prob (rare),
  otherwise to B1 with probability 1 - trans_prob (common). Symmetric for selecting B2.
Reward probabilities drift with Gaussian noise and are kept within bounds via
either reflecting or clipping (see `reward_boundary`).
When `permute_actions` is enabled, left/right action mappings are randomized
each trial (observations do not encode the mapping).
"""

import numpy as np
from .base_env import BaseEnvironment


class TwoStepTask:
    """Event-driven Two-Step Task with minimal observations.

    Observation channels (n_input = 3): [s1, s2_B1, s2_B2]
    Phases:
      - phase = 0: show s1 (1,0,0) until agent picks a1 in {0,1}
      - phase = 1: show s2 one-hot depending on transition; wait for a2 in {0,1}
      - phase = 2: terminal; reward exposed via step() return
    Action permutation:
      - when enabled, action indices are remapped per trial to simulate
        left/right position randomization.
    """

    def __init__(
        self,
        s1_duration=1,
        s2_duration=1,
        trans_prob=0.3,
        reward_prob=0.5,
        reward_boundary="reflect",
        permute_actions=True,
        reward_sd=0.025,
        reward_min=0.25,
        reward_max=0.75,
        **kwargs,
    ):
        # durations kept for compatibility but unused
        self.n_input = 3

        # Task parameters
        self.trans_prob = float(trans_prob)  # probability of rare transition
        self.reward_boundary = str(reward_boundary).lower()
        if self.reward_boundary not in ("reflect", "clip"):
            raise ValueError("reward_boundary must be 'reflect' or 'clip'")
        self.permute_actions = bool(permute_actions)
        self.reward_sd = float(reward_sd)
        self.reward_min = float(reward_min)
        self.reward_max = float(reward_max)

        # Reward probabilities for 4 stage-2 options (state x action)
        init_p = float(reward_prob)
        init_p = min(max(init_p, self.reward_min), self.reward_max)
        self.reward_probs = np.full((2, 2), init_p, dtype=float)

        # Trial state
        self.phase = 0
        self.stage2 = None  # 1: B1, 2: B2
        self.choice1 = None
        self.choice2 = None
        self.action1 = None
        self.action2 = None
        self.reward = None
        self.completed = False
        self.common = None  # whether transition followed desired mapping
        self.choice = None  # backward-compat alias for choice1
        self.stage1_action_map = None
        self.stage2_action_map = None

    def reset(self):
        self.phase = 0
        self.stage2 = None
        self.choice1 = None
        self.choice2 = None
        self.action1 = None
        self.action2 = None
        self.reward = None
        self.completed = False
        self.common = None
        self.choice = None
        self._init_action_maps()
        return np.array([1.0, 0.0, 0.0], dtype=float)  # s1

    def configure(self, trial_data, trial_settings):
        # Ignore any durations or precomputed sequences; this task is event-driven.
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
        if self.stage2 is None or self.choice2 is None:
            return 0
        stage_index = self.stage2 - 1  # 0: B1, 1: B2
        p = float(self.reward_probs[stage_index, self.choice2])
        return 1 if np.random.rand() < p else 0

    def _init_action_maps(self):
        if self.permute_actions:
            self.stage1_action_map = np.random.permutation(2)
            self.stage2_action_map = {
                1: np.random.permutation(2),
                2: np.random.permutation(2),
            }
        else:
            self.stage1_action_map = np.array([0, 1], dtype=int)
            self.stage2_action_map = {
                1: np.array([0, 1], dtype=int),
                2: np.array([0, 1], dtype=int),
            }

    def _apply_reward_boundary(self, values):
        if self.reward_boundary == "reflect":
            span = self.reward_max - self.reward_min
            if span <= 0:
                return np.clip(values, self.reward_min, self.reward_max)
            shifted = np.mod(values - self.reward_min, 2.0 * span)
            reflected = np.where(shifted <= span, shifted, 2.0 * span - shifted)
            return reflected + self.reward_min
        return np.clip(values, self.reward_min, self.reward_max)

    def _drift_reward_probs(self):
        noise = np.random.normal(0.0, self.reward_sd, size=self.reward_probs.shape)
        self.reward_probs = self._apply_reward_boundary(self.reward_probs + noise)

    def step(self, action):
        if self.phase == 0:
            # Expect a1 in {0,1}
            if action in (0, 1):
                self.action1 = int(action)
                self.choice1 = int(self.stage1_action_map[self.action1])
                self.choice = self.choice1
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
                self.action2 = int(action)
                stage_map = self.stage2_action_map.get(self.stage2)
                if stage_map is None:
                    stage_map = np.array([0, 1], dtype=int)
                self.choice2 = int(stage_map[self.action2])
                self.reward = self._sample_reward()
                self.phase = 2
                self.completed = True
                self._drift_reward_probs()
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
            "action1": self.action1,
            "action2": self.action2,
            "choice1": self.choice1,
            "choice2": self.choice2,
            "choice": self.choice,  # Backward compatibility
            "stage2": self.stage2,
            "reward": self.reward,
            "common": self.common,
            "completed": self.completed,
            "stage1_action_map": None if self.stage1_action_map is None else self.stage1_action_map.tolist(),
            "stage2_action_map": None if self.stage2_action_map is None else {
                k: v.tolist() for k, v in self.stage2_action_map.items()
            },
        }


class MDPEnvironment(BaseEnvironment):
    """
    MDP Two-Step Task Environment (event-driven)
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
        """Success if the trial delivered a reward."""
        return 1 if self.task.reward else 0
    
    def is_completed(self):
        """Check if current trial is completed"""
        return 1 if self.task.completed else 0
        
    def render(self):
        """Render current state"""
        print(f"Two-Step Task, Reward: {self.episode_reward}, Done: {self.done}")
        if hasattr(self.task, 'choice1') and self.task.choice1 is not None:
            print(f"Choice1: {self.task.choice1}, Choice2: {self.task.choice2}")
            print(f"Stage2: {self.task.stage2}, Reward: {self.task.reward}")
            print(f"Common transition: {self.task.common}")
            print(f"Reward probs: {self.task.reward_probs}")


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
