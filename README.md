# Unified RL Framework

A unified framework for training GRU and DRQN models on T-maze and Two-Step MDP task environments, preserving the original logic from both source projects.

## Overview

This framework allows you to:
- Train **GRU** or **DRQN** models
- On **T-maze** (belief-rnn) or **Two-Step MDP** (sequence learning) environments  
- Customize environment parameters with command-line arguments
- Preserve original implementations while providing unified interface

## Installation

```bash
pip install -r requirements.txt
```

## Usage
 
### Training Mode

#### GRU on Two-Step MDP Task
```bash
python main.py --model gru --environment mdp --num-episodes 5000 --batch-size 32
```

#### DRQN on T-maze
```bash
python main.py --model drqn --environment tmaze --num-episodes 5000 --batch-size 32
```

### Evaluation Mode

#### Load Trained Model and Evaluate
```bash
# Evaluate GRU model on MDP task
python main.py --model gru --environment mdp --preload 'experiment_name/model_name.pt' --eval-num-episodes 100

# Evaluate DRQN model on T-maze task  
python main.py --model drqn --environment tmaze --preload 'experiment_name/model_name.pt' --eval-num-episodes 50
```

#### Evaluation Output
- **Trajectory Data**: Saved as `trajectories_{env_name}.npz`
  - MDP: (s1, a1, s2, a2, r) tuples for each episode
  - T-maze: (s, a, s, a, r) trajectory sequences
- **Model Activations**: Saved as `activations_{env_name}.npz`
  - Hidden states for each timestep of each episode
  - Shape: (num_episodes, num_timesteps, hidden_size)
- **Performance Metrics**: Avg Reward, Success Rate logged to console and file

### Key Training Parameters

- **`--num-episodes`**: Number of training episodes (default: 1000)
  - Controls total training duration
  - More episodes = longer training, better performance
  - Recommended: 5000-10000 for complex tasks

- **`--batch-size`**: Batch size for training (default: 32 for DRQN, 128 for GRU)
  - Number of episodes processed in parallel
  - Larger batches = more stable training, more memory usage
  - GRU: 32-128, DRQN: 16-64

- **`--learning-rate`**: Learning rate for optimizer (default: 0.001)
  - Controls step size in parameter updates
  - Higher = faster learning, risk of instability
  - Lower = more stable, slower convergence

- **`--hidden-size`**: Hidden layer size (default: 128 for GRU, 32 for DRQN)
  - Model capacity and complexity
  - Larger = more expressive, more parameters
  - Smaller = faster training, less memory

- **`--eval-every`**: Evaluation frequency (default: 50)
  - How often to run evaluation during training
  - Lower = more frequent monitoring, higher overhead
  - Higher = less frequent monitoring, lower overhead

### Advanced Usage

## T-maze Environment (DRQN)

The T-maze environment supports various customization options for studying partially observable navigation tasks.

#### T-maze Reward Structure
- **Success**: +4.0 (reaching correct goal)
- **Failure**: -0.1 (reaching wrong goal or hitting wall)
- **Intermediate**: 0.0 (in corridor or at junction)
- **Terminal**: 0.0 (already completed episode)

The task requires working memory: agents must remember the initial hint (goal_up/goal_down) and make the correct choice at the T-junction.

### T-maze Arguments and Their Roles

- **`--length`**: Corridor length (default: 10)
  - Controls the number of steps needed to reach the T-junction
  - Longer mazes require more memory and planning
  - Affects belief update complexity

- **`--stochasticity`**: Transition stochasticity (default: 0.0, range: 0.0-1.0)
  - Probability that actions are randomly altered
  - 0.0 = deterministic, 1.0 = completely random
  - Tests robustness to noise and uncertainty

- **`--irrelevant`**: Number of irrelevant features (default: 0)
  - Adds noise dimensions to observations
  - Tests ability to ignore distractors
  - Increases observation space complexity

### T-maze Usage Examples

#### Basic T-maze with DRQN
```bash
python main.py \
  --model drqn \
  --environment tmaze \
  --length 20 \
  --num-episodes 5000
```

#### Stochastic T-maze (20% noise)
```bash
python main.py \
  --model drqn \
  --environment tmaze \
  --length 30 \
  --stochasticity 0.2 \
  --num-episodes 7000 \
  --name drqn_stochastic
```

#### T-maze with irrelevant features
```bash
python main.py \
  --model drqn \
  --environment tmaze \
  --length 15 \
  --irrelevant 5 \
  --num-episodes 6000 \
  --name drqn_irrelevant
```

#### Complex T-maze (long + stochastic + irrelevant)
```bash
python main.py \
  --model drqn \
  --environment tmaze \
  --length 50 \
  --stochasticity 0.15 \
  --irrelevant 3 \
  --epsilon 0.1 \
  --gamma 0.98 \
  --hidden-size 64 \
  --num-episodes 10000 \
  --name drqn_complex_tmaze
```

## Two-Step MDP Environment (GRU)

The Two-Step MDP environment supports detailed timing control for studying temporal credit assignment and model-based learning.

### Trial Structure and Duration Control

The Two-Step task consists of 5 fixed slots:
1. **Slot 0**: Stage 1 Stimulus (s1) - 1 timestep
2. **Slot 1**: Stage 1 Action (a1) - 1 timestep  
3. **Slot 2**: Stage 2 Stimulus (s2) - 1 timestep
4. **Slot 3**: Stage 2 Action (a2) - 1 timestep
5. **Slot 4**: Reward - 1 timestep

**Total Trial Length = 5 timesteps (fixed)**

### Two-Step Arguments and Their Roles

- **`--s1-duration`**: Stage 1 stimulus duration (default: 1, kept for compatibility)
  - Note: Fixed to 1 timestep in 5-slot structure
  - Controls initial choice options (A1/A2) presentation

- **`--s2-duration`**: Stage 2 stimulus duration (default: 1, kept for compatibility)
  - Note: Fixed to 1 timestep in 5-slot structure
  - Controls outcome states (B1/B2) presentation

- **`--trans-prob`**: Transition probability (default: 0.8, range: 0.0-1.0)
  - Probability that A1→B1 and A2→B2 (common transitions)
  - 1.0 = deterministic, 0.5 = random, 0.0 = reversed
  - Controls predictability of state transitions

- **`--reward-prob`**: Reward probability (default: 0.8, range: 0.0-1.0)
  - Base probability of receiving reward in optimal states
  - **Block structure**: Switches between B1/B2 every 50 trials
    - Trials 1-50: Block 0 (B1 high reward)
    - Trials 51-100: Block 1 (B2 high reward)  
    - Trials 101-150: Block 0 (B1 high reward)
    - And so on...
  - Controls reward contingency strength


## Complete Workflow Examples

### 2 step MDP

#### Step 1: Training
```bash
python main.py \
  --model gru \
  --environment mdp \
  --s1-duration 1 \
  --s2-duration 1 \
  --trans-prob 0.6 \
  --reward-prob 0.6 \
  --num-episodes 10000 \
  --batch-size 64 \
  --eval-every 500 \
  --save-every 1000 \
  --name gru_mdp
```
**Trial length: 5 timesteps (s1→a1→s2→a2→reward)**

#### Step 2: Evaluation
```bash
python main.py \
  --model gru \
  --environment mdp \
  --preload 'gru_mdp_experiment/gru_mdp_experiment_final.pt' \
  --eval-num-episodes 200
```

### 2. T-maze

#### Step 1: Training
```bash
python main.py \
  --model drqn \
  --environment tmaze \
  --length 1 \
  --stochasticity 0.1 \
  --epsilon 0.1 \
  --num-episodes 10000 \
  --batch-size 64 \
  --buffer-capacity 2048 \
  --eval-every 500 \
  --save-every 1000 \
  --name drqn_tmaze \
  --cuda
```

```
python main.py \
  --model gru \
  --environment tmaze \
  --length 1 \
  --stochasticity 0.1 \
  --num-episodes 10000 \
  --batch-size 128 \
  --eval-every 500 \
  --save-every 1000 \
  --name gru_tmaze 
```

#### Step 2: Evaluation
```bash
python main.py \
  --model drqn \
  --environment tmaze \
  --preload 'drqn_tmaze_experiment/drqn_tmaze_experiment_final.pt' \
  --eval-num-episodes 100
```

## Complete Arguments Reference

### Core Arguments
- `--model`: Choose between `gru` or `drqn`
- `--environment`: Choose between `tmaze` or `mdp`
- `--num-episodes`: Number of training episodes (default: 1000)
- `--name`: Experiment name for logging and saving

### Model Hyperparameters
- `--hidden-size`: Hidden layer size (default: GRU=128, DRQN=32)
- `--num-layers`: Number of RNN layers (default: GRU=1, DRQN=2)  
- `--learning-rate`: Learning rate (default: 1e-3)
- `--batch-size`: Batch size (default: GRU=128, DRQN=32)

### DRQN Specific (for T-maze)
- `--epsilon`: Epsilon for epsilon-greedy policy (default: 0.2)
- `--gamma`: Discount factor (default: 0.98)
- `--buffer-capacity`: Replay buffer capacity (default: 8192)
- `--cell-type`: RNN cell type - `gru` or `lstm` (default: gru)

### T-maze Environment
- `--length`: Corridor length (default: 10) 
- `--stochasticity`: Transition stochasticity, range 0.0-1.0 (default: 0.0)
- `--irrelevant`: Number of irrelevant features (default: 0)

### Two-Step MDP Environment  
- `--s1-duration`: Stage 1 stimulus duration (default: 1, fixed in 5-slot structure)
- `--s2-duration`: Stage 2 stimulus duration (default: 1, fixed in 5-slot structure)
- `--trans-prob`: Transition probability, range 0.0-1.0 (default: 0.8)
- `--reward-prob`: Reward probability, range 0.0-1.0 (default: 0.8)

### Training & Evaluation
- `--eval-every`: Evaluation frequency during training (default: 50)
- `--num-rollouts`: Number of evaluation rollouts (default: 50)
- `--seed`: Random seed (default: 42)
- `--cuda`: Use CUDA if available

### Model Saving
- `--save-every`: Model saving frequency (default: 0)
  - 0 = save only at the end
  - >0 = save every N episodes (e.g., --save-every 100)

### Evaluation Mode
- `--preload`: Path to model checkpoint for evaluation mode
  - Example: `--preload 'experiment_name/model_name.pt'`
  - Automatically prepends `logs/` if not present
- `--eval-num-episodes`: Number of episodes for evaluation mode (default: 100)

### Logging & Saving
- `--log-dir`: Log directory (default: ./logs)
- `--save-dir`: Model save directory (default: ./models)

## Project Structure

```
unified_rl_framework/
├── models/
│   ├── __init__.py
│   ├── base_agent.py      # Base agent interface
│   ├── gru_model.py       # GRU agent (from Sequence_learning)
│   └── drqn_model.py      # DRQN agent (from belief-rnn)
├── environments/
│   ├── __init__.py
│   ├── base_env.py        # Base environment interface
│   ├── tmaze_env.py       # T-maze environment
│   └── mdp_env.py         # Two-Step MDP environment
├── config.py              # Configuration management
├── main.py                # Main training script
├── requirements.txt       # Dependencies
└── README.md              # This file
```

## Supported Combinations

All model-environment combinations are fully supported:

| Model | Environment | Description |
|-------|-------------|-------------|
| **DRQN** | **T-maze** | Partially observable navigation with belief tracking (original) |
| **DRQN** | **Two-Step MDP** | Q-learning approach to temporal credit assignment |
| **GRU** | **T-maze** | RNN-based policy learning for navigation tasks |
| **GRU** | **Two-Step MDP** | Sequence learning for model-based decision making (original) |

## Key Features

- **Full Cross-Compatibility**: All 4 model-environment combinations supported
- **Two Validated Environments**: T-maze (belief-rnn) and Two-Step MDP (sequence learning)
- **Original Logic Preserved**: Core algorithms unchanged from source projects
- **Adaptive Model Sizing**: Observation/action spaces automatically matched to environments
- **Fixed 5-Slot Structure**: s1→a1→s2→a2→reward sequence for Two-Step MDP
- **Environment-Specific Tuning**: All parameters from original papers accessible
- **Comprehensive Logging**: Training progress and evaluation metrics
- **Model Persistence**: Save and load trained models
- **Evaluation Mode**: Load trained models and run evaluation-only experiments
- **Trajectory Saving**: Save complete (s,a,r) sequences for analysis
- **Activation Saving**: Save model hidden states for each timestep
- **Flexible Model Saving**: Save models at regular intervals or only at the end
- **Unified Output Format**: Consistent logging across all model-environment combinations

## Expected Behavior

### DRQN on T-maze (Original)
- Learns optimal navigation policy for partially observable maze
- Develops internal belief representation about goal location
- Handles stochasticity and irrelevant features robustly
- Uses experience replay for stable learning

### GRU on Two-Step MDP (Original)
- Learns model-based vs model-free strategies
- Adapts to changing reward contingencies (block structure)
- Shows temporal credit assignment across multi-step trials
- Balances exploration vs exploitation in probabilistic environment

### GRU on T-maze (Cross-Training)
- Adapts RNN sequence learning to spatial navigation
- Develops memory for corridor position and goal location
- Shows different learning dynamics compared to DRQN
- May require more episodes due to different training approach

### DRQN on Two-Step MDP (Cross-Training)
- Applies Q-learning to temporal credit assignment
- Uses replay buffer for experience-based learning
- May show different exploration patterns compared to GRU
- Learns action-value functions over sequence states

## Troubleshooting

1. **CUDA Issues**: Use `--cuda` flag only if CUDA is available
2. **Memory Issues**: Reduce `--batch-size` or `--buffer-capacity` 
3. **Convergence Issues**: 
   - T-maze: Try longer episodes or lower stochasticity
   - Two-Step: Adjust transition probabilities or reward probabilities
4. **Parameter Ranges**: 
   - Probabilities should be 0.0-1.0
   - Two-Step durations are fixed to 1 in 5-slot structure
   - Episode counts depend on task complexity

## Original Papers

This framework preserves the exact logic from:

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
@article{zhang2020sequence,
    title={A Sequence Learning Model for Decision Making in the Brain},
    author={Zhewei Zhang and others},
    year={2020}
}
```
