# Unified RL Framework

A unified framework for training GRU and DRQN models on T-maze and Two-Step MDP task environments, preserving the original logic from both source projects.

## Overview

This framework allows you to:
- Train GRU or DRQN models
- Run on T-maze or Two-Step MDP environments
- Configure model, environment, and training parameters via a single CLI

## Model description

### GRU (GRUAgent)
- 입력: 관측 벡터(1-step 시퀀스)와 hidden state. 관측은 `(1, 1, obs_size)` 형태로 GRU에 들어갑니다.
- 내부 구조: GRU → ReLU → Linear(decoder) → Sigmoid. decoder는 입력 크기와 동일한 차원을 출력합니다.
- 행동 선택:
  - MDP에서는 hidden state의 마지막 층을 policy head(`hidden_size -> action_size`)로 통과시켜 logits를 만들고, 이를 기반으로 행동을 선택합니다.
  - T-maze에서는 decoder 출력의 앞 4차원을 softmax로 변환해 행동을 샘플링합니다.
- 학습:
  - MDP에서는 REINFORCE 방식으로 episode return을 사용해 policy head와 GRU를 함께 업데이트합니다.
  - T-maze에서는 다음 관측 예측(MSE)을 기반으로 GRU를 업데이트합니다.
- 출력: 선택된 action과 업데이트된 hidden state. 학습 단계에서는 loss 스칼라를 추가로 반환합니다.

### DRQN (DRQNAgent)
- 입력: 현재 관측 벡터와 이전 행동 one-hot을 이어 붙인 벡터(`action_size + obs_size`).
- 내부 구조: RNN 셀(GRU/LSTM)로 시퀀스를 처리한 뒤 Linear로 Q-value를 출력합니다.
- 행동 선택: epsilon-greedy. 랜덤 선택 시 hidden state는 유지하고, greedy 선택 시 Q-network 출력의 argmax를 사용합니다.
- 학습:
  - trajectory replay buffer에서 시퀀스를 샘플링합니다.
  - target network를 사용한 TD 타깃으로 MSE 손실을 계산하고, 주기적으로 target network를 동기화합니다.
- 출력: 선택된 action과 업데이트된 hidden state. 학습 단계에서는 loss 스칼라를 추가로 반환합니다.

### 환경별 입력/출력 크기 요약
- MDP: 관측 3차원(one-hot), 행동 2개 (`action_size=2`)
- T-maze: 관측 4차원(또는 위치 one-hot), 행동 4개 (`action_size=4`)

## Installation

```bash
pip install -r src/requirements.txt
```

## Quick start

From the repo root:

```bash
python main.py \
  --model gru \
  --environment tmaze \
  --length 2 \
  --stochasticity 0.1 \
  --num-episodes 500 \
  --batch-size 1 \
  --eval-every 100 \
  --save-every 100 \
  --tmaze-obs-mode position \
  --seed 1 \
  --name gru_tmaze 
```

```bash
python src/main.py \
  --model drqn \
  --epsilon 0.1 \
  --environment mdp \
  --trans-prob 0.3 \
  --reward-prob 0.7 \
  --mdp-reward-boundary reflect \
  --mdp-permute-actions \
  --mdp-block-size 67 \
  --mdp-num-blocks 3 \
  --num-episodes 201 \
  --batch-size 1 \
  --eval-every 201 \
  --save-every 201 \
  --seed 1 \
  --name drqn_mdp \
  --cuda
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
- `--mdp-reward-boundary`: Reward drift boundary handling (`reflect` or `clip`)
- `--mdp-permute-actions`: Permute left/right action mapping each trial (default: on)

### Two-Step MDP behavior (Daw 2011 style)
- Stage 1: choose between two actions at S1, then transition to B1/B2 with common vs rare dynamics.
- Stage 2: choose between two actions in the reached B state.
- Reward attaches to the S2 state-action pair (4 options total), not just the state.
- Each option’s reward probability drifts every trial via an independent Gaussian random walk,
  reflected at [0.25, 0.75] with SD 0.025 (hardcoded defaults).
- When action permutation is enabled, the left/right mapping is randomized per trial
  (the observation does not encode the mapping).

## Environment diagrams and variables

## 한국어 설명

### T-maze (tmaze)
- 시작점에서 복도를 따라 이동해 갈림길에서 위/아래 목표를 선택하는 간단한 미로 과제입니다.
- 관측은 `type`(타입 원‑핫) 또는 `position`(위치 원‑핫) 중 하나로 제공됩니다.
- `stochasticity`는 행동이 랜덤하게 뒤집히는 확률을 의미합니다.

### Two-Step MDP (mdp)
- 1단계에서 두 행동 중 하나를 선택하면, 70/30 전이 확률로 2단계 상태(B1/B2)로 이동합니다.
- 2단계에서 다시 두 행동 중 하나를 선택하고, 상태×행동 쌍에 연결된 보상 확률에 따라 보상을 얻습니다.
- 보상 확률은 매 트라이얼마다 Gaussian random walk로 서서히 변하며, 경계는 `reflect` 또는 `clip`으로 제한됩니다.
  - `reflect`: 경계를 넘어간 값은 초과분만큼 되돌려 반사됩니다(예: 0.78이면 0.72로 반사). 경계에 값이 덜 고이는 효과가 있습니다.
  - `clip`: 경계를 넘어간 값은 경계값으로 잘립니다(예: 0.78이면 0.75로 고정). 경계 근처에 값이 더 많이 쌓일 수 있습니다.
- `mdp-permute-actions`가 켜져 있으면 좌/우 행동 매핑이 매 트라이얼마다 바뀝니다.
  - 에이전트의 `action=0/1`이 항상 같은 옵션을 의미하지 않고, 매 트라이얼마다 랜덤하게 좌/우에 재할당됩니다.
  - 관측에는 좌/우 정보가 없으므로, 같은 관측이라도 행동 번호와 옵션의 대응이 변합니다.
  - Daw 실험에서 2단계 옵션의 좌/우 위치를 trial마다 섞는 조작을 모사합니다.

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
                                      S1 (obs [1,0,0])
                                          a1 in {0,1}
              (per-trial action map: left/right permuted)
                                /                                 \
      common (1 - trans_prob)           rare (trans_prob)
                       v                                           v
            B1 (obs [0,1,0])                     B2 (obs [0,0,1])
                a2 in {0,1}                            a2 in {0,1}
        r ~ Bernoulli(p[B1,a2])          r ~ Bernoulli(p[B2,a2])
                        \                                          /
                           \------- terminal ------/
```

Notes:
- Action permutation: when enabled, `action=0/1` is remapped per trial to left/right options at both stages.
- Transition dynamics: each S1 action has a preferred B state (common), and the other is rare.
- Reward dynamics: `p[state, action]` drifts each trial via Gaussian random walk with boundary handling.

Variable mapping (creation in `src/main.py` and `src/environments/mdp_env.py`):
- `--trans-prob` -> `TwoStepTask.trans_prob` (rare transition probability)
- `--reward-prob` -> initial `TwoStepTask.reward_probs` values for all 4 options
- `--mdp-reward-boundary` -> `TwoStepTask.reward_boundary` (`reflect` or `clip`)
- `--mdp-permute-actions` -> `TwoStepTask.permute_actions` (per-trial left/right mapping)
- `--s1-duration`, `--s2-duration` -> accepted but unused in event-driven task
- Internal defaults: `reward_sd=0.025`, `reward_min=0.25`, `reward_max=0.75`

Reward drift:
- After each trial, `reward_probs` add Gaussian noise (SD `reward_sd`) and apply
  `--mdp-reward-boundary` to keep values within [`reward_min`, `reward_max`].

### Training and evaluation
- `--num-episodes`: Number of training episodes
- `--eval-period`: Evaluation period
- `--eval-every`: Evaluation frequency during training
- `--save-every`: Model saving frequency (0 = only at end)
- `--num-rollouts`: Number of evaluation rollouts
- `--seed`: Random seed
- `--cuda`: Use CUDA if available
- `--mdp-block-size`: Block size in trials for MDP (e.g., 67 for Daw 2011)
- `--mdp-num-blocks`: Number of MDP blocks (e.g., 3 for Daw 2011)
- `--name`: Experiment name
- `--log-dir`: Log directory
- `--save-dir`: Model save directory
- `--preload`: Path to model checkpoint for evaluation-only runs
- `--eval-num-episodes`: Number of episodes for evaluation mode

MDP block markers only add logging (no reward or transition resets between blocks).

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
