# Unified RL Framework

A unified framework for training GRU and DRQN models on T-maze and Two-Step MDP task environments, preserving the original logic from both source projects.

## Overview

This framework allows you to:
- Train GRU or DRQN models
- Run on T-maze or Two-Step MDP environments
- Configure model, environment, and training parameters via a single CLI

## Installation

```bash
pip install -r src/requirements.txt
```

## Quick start

From the repo root:

```bash
python src/main.py \
  --model drqn \
  --environment tmaze \
  --num-episodes 5000 \
  --batch-size 32 \
  --length 20 \
  --epsilon 0.1 \
  --name drqn_tmaze
```

Use the parameter groups below to switch model/environment and tune the run.

## Parameters

### Required
- `--model`: `gru` or `drqn`
- `--environment`: `tmaze` or `mdp`

### Model hyperparameters (shared)
- `--hidden-size`: Hidden layer size
- `--num-layers`: Number of RNN layers
- `--learning-rate`: Learning rate
- `--batch-size`: Batch size

### GRU-specific
- No additional flags beyond the shared model hyperparameters.

### DRQN-specific
- `--epsilon`: Epsilon for epsilon-greedy
- `--gamma`: Discount factor
- `--buffer-capacity`: Replay buffer capacity
- `--cell-type`: RNN cell type (`gru` or `lstm`)

### T-maze hyperparameters
- `--length`: Corridor length
- `--stochasticity`: Transition stochasticity
- `--irrelevant`: Number of irrelevant features
- `--tmaze-obs-mode`: Observation mode (`type` or `position`)

### Two-Step MDP hyperparameters
- `--s1-duration`: Stage 1 stimulus duration (compat, unused)
- `--s2-duration`: Stage 2 stimulus duration (compat, unused)
- `--trans-prob`: Rare transition probability (common = 1 - `--trans-prob`)
- `--reward-prob`: Initial reward probability for each S2 state-action option

### Two-Step MDP behavior (Daw 2011 style)
- Stage 1: choose between two actions at S1, then transition to B1/B2 with common vs rare dynamics.
- Stage 2: choose between two actions in the reached B state.
- Reward attaches to the S2 state-action pair (4 options total), not just the state.
- Each option’s reward probability drifts every trial via an independent Gaussian random walk,
  clipped to [0.25, 0.75] with SD 0.025 (hardcoded defaults).

## Environment diagrams and variables

### T-maze (tmaze)

```
            [L+1] Up terminal (reward if goal_up)
                |
[0] start - [1] - ... - [L-1] - [L] junction
                |
            [L+2] Down terminal (reward if not goal_up)
```

Variable mapping (creation in `src/main.py` and `src/environments/tmaze_env.py`):
- `--length` -> `TMazeEnvironment.length` (corridor length L, controls state indexing and obs size)
- `--stochasticity` -> `TMazeEnvironment.stochasticity` (chance to randomize actions)
- `--irrelevant` -> `TMazeEnvironment.irrelevant_features` (extra random features appended)
- `--tmaze-obs-mode` -> `TMazeEnvironment.obs_mode` (`type` or `position`)
- `bayes` (not exposed via CLI) -> `TMazeEnvironment.bayes` (enables belief updates)

Observation modes:
- `type`: 4-d one-hot `[UP, DOWN, CORRIDOR, CROSSROAD]`; at start (pos 0) shows UP/DOWN hint.
- `position`: one-hot of size `L+3` with indices 0..L (corridor + junction), L+1 (up), L+2 (down).

### Two-Step MDP (mdp)

```
S1 (obs [1,0,0]) --a1 in {0,1}-->
  common (1 - trans_prob) -> B1 (obs [0,1,0])
  rare   (trans_prob)     -> B2 (obs [0,0,1])
Then a2 in {0,1} -> reward (Bernoulli p = reward_probs[state, action]) -> terminal
```

Variable mapping (creation in `src/main.py` and `src/environments/mdp_env.py`):
- `--trans-prob` -> `TwoStepTask.trans_prob` (rare transition probability)
- `--reward-prob` -> initial `TwoStepTask.reward_probs` values for all 4 options
- `--s1-duration`, `--s2-duration` -> accepted but unused in event-driven task
- Internal defaults: `reward_sd=0.025`, `reward_min=0.25`, `reward_max=0.75`

Reward drift:
- After each trial, `reward_probs` add Gaussian noise (SD `reward_sd`) and clip to
  [`reward_min`, `reward_max`].

### Training and evaluation
- `--num-episodes`: Number of training episodes
- `--eval-period`: Evaluation period
- `--eval-every`: Evaluation frequency during training
- `--save-every`: Model saving frequency (0 = only at end)
- `--num-rollouts`: Number of evaluation rollouts
- `--seed`: Random seed
- `--cuda`: Use CUDA if available
- `--name`: Experiment name
- `--log-dir`: Log directory
- `--save-dir`: Model save directory
- `--preload`: Path to model checkpoint for evaluation-only runs
- `--eval-num-episodes`: Number of episodes for evaluation mode

Evaluation-only runs use `--preload` and `--eval-num-episodes` and write logs under the chosen `--log-dir`.

## Project Structure

```
.
├── README.md
└── src/
    ├── main.py
    ├── config.py
    ├── eval.py
    ├── models/
    ├── environments/
    └── requirements.txt
```
