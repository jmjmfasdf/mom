"""
Evaluation script

Usage example (run from src/):
    python eval.py \
      --load logs/drqn_mdp \
      --eval_eps 100 \
      --name drqn_mdp

Loads 'model.pt' and 'config.yaml' from --load directory, rebuilds the
model and environment, then evaluates for N episodes while saving
per-episode trajectories and hidden activations.
"""

import os
import sys
import time
import json
import argparse
import random
import numpy as np
import torch
import yaml

# Ensure local imports work when running from src/
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import GRUAgent, DRQNAgent
from environments import TMazeEnvironment, MDPEnvironment


def set_random_seeds(seed: int):
    """Set random seeds for reproducibility in evaluation."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate a trained model and save episodes + activations')
    parser.add_argument('--load', type=str, required=True,
                        help='Directory containing model.pt and config.yaml (e.g., logs/drqn_mdp)')
    parser.add_argument('--eval_eps', type=int, default=100,
                        help='Number of evaluation episodes to run')
    parser.add_argument('--name', type=str, required=True,
                        help='Name tag for output files (e.g., drqn_mdp)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for evaluation runs')
    return parser.parse_args()


def hidden_to_numpy(hidden_state):
    """Extract a 1D numpy activation vector from various hidden state types."""
    if isinstance(hidden_state, tuple):
        # DRQN GRU returns (h,), LSTM returns (h, c)
        if len(hidden_state) == 1:
            h = hidden_state[0]
        else:
            h = hidden_state[0]
    else:
        h = hidden_state
    # h shape: (num_layers, batch, hidden_size)
    if isinstance(h, torch.Tensor):
        return h[-1, 0, :].detach().cpu().numpy()
    # Fallback (should not happen)
    return np.array(h)


def build_environment(env_name, env_cfg):
    if env_name == 'tmaze':
        length = int(env_cfg.get('length', 20))
        stochasticity = float(env_cfg.get('stochasticity', 0.0))
        irrelevant_features = int(env_cfg.get('irrelevant_features', 0))
        obs_mode = str(env_cfg.get('obs_mode', 'type'))
        return TMazeEnvironment(length=length,
                                stochasticity=stochasticity,
                                irrelevant_features=irrelevant_features,
                                obs_mode=obs_mode)
    elif env_name == 'mdp':
        kwargs = {}
        for k in ('s1_duration', 's2_duration', 'trans_prob', 'reward_prob'):
            if k in env_cfg:
                kwargs[k] = env_cfg[k]
        return MDPEnvironment(**kwargs)
    else:
        raise ValueError(f'Unknown environment: {env_name}')


def build_agent(model_name, model_cfg, obs_size, act_size, sequence_length, cuda_enabled):
    if model_name == 'gru':
        return GRUAgent(
            observation_size=obs_size,
            action_size=act_size,
            hidden_size=int(model_cfg.get('hidden_size', 128)),
            num_layers=int(model_cfg.get('num_layers', 1)),
            learning_rate=float(model_cfg.get('learning_rate', 1e-3)),
            batch_size=int(model_cfg.get('batch_size', 128)),
            cuda_enabled=cuda_enabled,
        )
    elif model_name == 'drqn':
        return DRQNAgent(
            observation_size=obs_size,
            action_size=act_size,
            hidden_size=int(model_cfg.get('hidden_size', 32)),
            num_layers=int(model_cfg.get('num_layers', 2)),
            cell=str(model_cfg.get('cell_type', 'gru')),
            learning_rate=float(model_cfg.get('learning_rate', 1e-3)),
            gamma=float(model_cfg.get('gamma', 0.98)),
            epsilon=float(model_cfg.get('epsilon', 0.0)),
            buffer_capacity=int(model_cfg.get('buffer_capacity', 8192)),
            target_update_freq=int(model_cfg.get('target_update_freq', 10)),
            sequence_length=int(sequence_length),
            cuda_enabled=cuda_enabled,
        )
    else:
        raise ValueError(f'Unknown model: {model_name}')


def eval_mdp(agent, environment, model_name, num_episodes):
    """Evaluate on two-step MDP and return list of episode dicts.

    Per episode keys: s1, a1, s2, a2, r, act_s1, act_a1, act_s2, act_a2, act_r
    Activations are taken right after processing the corresponding observation.
    """
    episodes = []
    for _ in range(num_episodes):
        s1 = environment.reset()
        h = agent.reset_hidden(1)

        # First choice (a1) from s1
        if model_name == 'gru':
            a1, h1 = agent.get_action(s1, h, epsilon=0.0, env_name='mdp')
        else:
            a1, h1 = agent.get_action(s1, h, epsilon=0.0)
        act_s1 = hidden_to_numpy(h1)
        act_a1 = act_s1  # same hidden immediately after choosing a1

        s2, r1, done, info = environment.step(a1)

        # Second choice (a2) from s2
        if model_name == 'gru':
            a2, h2 = agent.get_action(s2, h1, epsilon=0.0, env_name='mdp')
        else:
            a2, h2 = agent.get_action(s2, h1, epsilon=0.0)
        act_s2 = hidden_to_numpy(h2)
        act_a2 = act_s2

        _, r, done, info = environment.step(a2)
        # Terminal activation (reuse last hidden)
        act_r = act_s2

        ep = {
            's1': np.asarray(s1, dtype=float),
            'a1': int(a1),
            's2': np.asarray(s2, dtype=float),
            'a2': int(a2),
            'r': float(r),
            'act_s1': act_s1,
            'act_a1': act_a1,
            'act_s2': act_s2,
            'act_a2': act_a2,
            'act_r': act_r,
        }
        episodes.append(ep)
    return episodes


def eval_tmaze(agent, environment, model_name, num_episodes):
    """Evaluate on T-maze and return list of episode dicts.

    Per episode keys: states, actions, activations, final_reward, done
    Activations are recorded after processing each observation.
    """
    episodes = []
    for _ in range(num_episodes):
        obs = environment.reset()
        h = agent.reset_hidden(1)

        states = []
        actions = []
        activations = []
        final_reward = 0.0

        step = 0
        max_steps = environment.horizon()
        done = False
        while step < max_steps:
            states.append(np.asarray(obs, dtype=float))
            if model_name == 'gru':
                act, h_new = agent.get_action(obs, h, epsilon=0.0, env_name='tmaze')
            else:
                act, h_new = agent.get_action(obs, h, epsilon=0.0)
            activations.append(hidden_to_numpy(h_new))
            actions.append(int(act))

            obs, reward, done, info = environment.step(act)
            final_reward += float(reward)
            h = h_new
            step += 1
            if done:
                break

        episodes.append({
            'states': states,
            'actions': actions,
            'activations': activations,
            'final_reward': final_reward,
            'done': bool(done),
        })
    return episodes


def main():
    args = parse_args()
    set_random_seeds(args.seed)
    load_dir = args.load
    model_path = os.path.join(load_dir, 'model.pt')
    config_path = os.path.join(load_dir, 'config.yaml')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    model_name = cfg.get('model')
    env_name = cfg.get('environment')
    model_cfg = cfg.get('model_config', {})
    env_cfg = cfg.get('env_config', {})
    training_cfg = cfg.get('training_config', {})
    sequence_length = int(cfg.get('sequence_length', 20))

    cuda_enabled = bool(training_cfg.get('cuda', False)) and torch.cuda.is_available()

    # Build env and agent
    env = build_environment(env_name, env_cfg)
    agent = build_agent(model_name, model_cfg,
                        obs_size=env.get_observation_space(),
                        act_size=env.get_action_space(),
                        sequence_length=sequence_length,
                        cuda_enabled=cuda_enabled)

    # Load weights
    agent.load_model(model_path)

    # Evaluate
    if env_name == 'mdp':
        episodes = eval_mdp(agent, env, model_name, args.eval_eps)
    elif env_name == 'tmaze':
        episodes = eval_tmaze(agent, env, model_name, args.eval_eps)
    else:
        raise ValueError(f'Unsupported environment: {env_name}')

    # Output directory (inside load dir)
    out_dir = os.path.join(load_dir, f'eval_{args.name}')
    os.makedirs(out_dir, exist_ok=True)

    # Save results
    ts = int(time.time())
    npz_path = os.path.join(out_dir, f'{args.name}_episodes_{ts}.npz')
    # Use numpy object array to store heterogeneous dicts
    eval_meta = {
        'env': env_name,
        'model': model_name,
        'timestamp': ts,
        'source': 'eval',
        'train_seed': training_cfg.get('seed', None),
        'eval_seed': int(getattr(args, 'seed', 0)),
    }
    np.savez(npz_path,
             episodes=np.array(episodes, dtype=object),
             meta=eval_meta)

    # Also save a lightweight JSON summary (without activations) for quick inspection
    json_summary = []
    if env_name == 'mdp':
        for ep in episodes:
            json_summary.append({k: (v if not isinstance(v, np.ndarray) else v.tolist())
                                 for k, v in ep.items() if not k.startswith('act_')})
    else:
        for ep in episodes:
            json_summary.append({
                'len': len(ep.get('states', [])),
                'final_reward': ep.get('final_reward', 0.0),
                'done': ep.get('done', False)
            })
    json_path = os.path.join(out_dir, f'{args.name}_summary_{ts}.json')
    with open(json_path, 'w') as f:
        json.dump(json_summary, f)

    print(f"Saved episodes to {npz_path}")
    print(f"Saved summary to {json_path}")


if __name__ == '__main__':
    main()
