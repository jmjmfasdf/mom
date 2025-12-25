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

## Original Papers

**T-maze Environment:**
```bibtex
@article{lambrechts2022recurrent,
    title={Recurrent Networks, Hidden States and Beliefs in Partially Observable Environments},
    author={Gaspard Lambrechts and Adrien Bolland and Damien Ernst},
    journal={Transactions on Machine Learning Research},
    year={2022}
}
```

**Two-Step MDP Environment:**
```bibtex
@article{daw2011modelbased,
    title={Model-based influences on humans' choices and striatal prediction errors},
    author={Nathaniel D. Daw and Samuel J. Gershman and Ben Seymour and Peter Dayan and Raymond J. Dolan},
    journal={Neuron},
    year={2011}
}
```
