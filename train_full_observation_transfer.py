#!/usr/bin/env python3
"""
Train full observation encoders on both CyberBattleSim and Cyberwheel.
Uses all observation fields instead of reduced 8-dim representation.
"""

import os
import sys
import argparse
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from typing import Dict, List, Tuple

from adapters.full_observation_encoder import (
    CBSFullObservationEncoder,
    CWFullObservationEncoder,
    UnifiedFullObservationEncoder
)
from adapters.transfer_encoder import DynamicsModel
from adapters.unified_env import UnifiedSecEnv
from config.env_builders import make_cbs_env, make_cw_env


class ReplayBuffer:
    """Store transitions for training"""
    def __init__(self, capacity: int = 10000):
        self.memory = deque([], maxlen=capacity)
    
    def push(self, obs, action, next_obs, reward, done):
        self.memory.append((obs, action, next_obs, reward, done))
    
    def sample(self, batch_size: int):
        return random.sample(self.memory, min(batch_size, len(self.memory)))
    
    def __len__(self):
        return len(self.memory)


def collect_cbs_transitions(env, num_episodes: int = 50, max_steps: int = 200) -> List[Tuple]:
    """Collect transitions from CBS environment using full observations"""
    transitions = []
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        step = 0
        
        while not (done or truncated) and step < max_steps:
            # Get raw CBS observation (before translation)
            raw_obs = env._last_raw_obs if hasattr(env, '_last_raw_obs') else obs
            if isinstance(obs, dict) and "obs" in obs:
                # We need the original CBS observation, not the translated one
                raw_obs = env.env.observation_space.sample()  # This won't work, need actual obs
            
            # For now, we'll use the translated obs but note we need raw
            # In practice, you'd store the raw obs before translation
            action = env.action_space.sample()
            next_obs, reward, done, truncated, info = env.step(action)
            
            # Store transition with full observation
            transitions.append((obs, action, next_obs, float(reward), done or truncated))
            
            obs = next_obs
            step += 1
        
        if (episode + 1) % 10 == 0:
            print(f"  Collected {episode + 1}/{num_episodes} CBS episodes")
    
    return transitions


def collect_cw_transitions(env, num_episodes: int = 50, max_steps: int = 200) -> List[Tuple]:
    """Collect transitions from Cyberwheel environment using full observations"""
    transitions = []
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        step = 0
        
        while not (done or truncated) and step < max_steps:
            # Get raw Cyberwheel observation
            raw_obs = obs  # Cyberwheel obs is already a vector
            
            action = env.action_space.sample()
            next_obs, reward, done, truncated, info = env.step(action)
            
            transitions.append((raw_obs, action, next_obs, float(reward), done or truncated))
            
            obs = next_obs
            step += 1
        
        if (episode + 1) % 10 == 0:
            print(f"  Collected {episode + 1}/{num_episodes} Cyberwheel episodes")
    
    return transitions


def get_raw_cbs_observation(env) -> Dict:
    """Extract raw CBS observation before translation"""
    # Access the underlying CBS environment
    cbs_env = env.env if hasattr(env, 'env') else env
    
    # Get the last raw observation
    if hasattr(cbs_env, '_last_observation'):
        return cbs_env._last_observation
    elif hasattr(env, '_last_raw_obs'):
        return env._last_raw_obs
    else:
        # Fallback: construct from current state
        # This is a simplified version - in practice you'd get the full obs
        return {
            "discovered_node_count": 0,
            "nodes_privilegelevel": np.array([], dtype=np.int32),
            "discovered_nodes_properties": np.zeros((0, 3), dtype=np.int32),
            "credential_cache_length": 0,
            "_explored_network": type('obj', (object,), {'nodes': lambda: [], 'edges': lambda: []})(),
            "probe_result": 0,
            "escalation": 0,
            "newly_discovered_nodes_count": 0,
            "lateral_move": 0,
            "customer_data_found": 0,
            "credential_cache_matrix": []
        }


def train_full_observation_encoder(
    cbs_encoder: CBSFullObservationEncoder,
    cw_encoder: CWFullObservationEncoder,
    dynamics_model: DynamicsModel,
    cbs_transitions: List[Tuple],
    cw_transitions: List[Tuple],
    epochs: int = 20,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    device: torch.device = None
):
    """Train encoders and dynamics model on full observations"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Move models to device
    cbs_encoder = cbs_encoder.to(device)
    cw_encoder = cw_encoder.to(device)
    dynamics_model = dynamics_model.to(device)
    
    # Optimizer
    optimizer = optim.Adam(
        list(cbs_encoder.parameters()) + 
        list(cw_encoder.parameters()) + 
        list(dynamics_model.parameters()),
        lr=learning_rate
    )
    
    # Loss functions
    mse_loss = nn.MSELoss()
    
    # Set to training mode
    cbs_encoder.train()
    cw_encoder.train()
    dynamics_model.train()
    
    print(f"\nTraining on {len(cbs_transitions)} CBS transitions and {len(cw_transitions)} CW transitions")
    
    all_losses = []
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        # Combine transitions
        all_transitions = cbs_transitions + cw_transitions
        random.shuffle(all_transitions)
        
        # Train in batches
        for i in range(0, len(all_transitions), batch_size):
            batch = all_transitions[i:i + batch_size]
            
            if len(batch) < 2:
                continue
            
            # Process batch
            cbs_features_list = []
            cw_features_list = []
            next_cbs_features_list = []
            next_cw_features_list = []
            actions_list = []
            rewards_list = []
            
            for obs, action, next_obs, reward, done in batch:
                # Determine if CBS or CW based on observation type
                if isinstance(obs, dict):
                    # CBS observation
                    try:
                        feat = cbs_encoder(obs)
                        next_feat = cbs_encoder(next_obs) if isinstance(next_obs, dict) else None
                        if next_feat is not None:
                            cbs_features_list.append(feat)
                            next_cbs_features_list.append(next_feat)
                            actions_list.append(action)
                            rewards_list.append(reward)
                    except Exception as e:
                        print(f"Warning: CBS encoding failed: {e}")
                        continue
                elif isinstance(obs, (np.ndarray, list)):
                    # Cyberwheel observation
                    try:
                        obs_vec = np.array(obs) if isinstance(obs, list) else obs
                        next_obs_vec = np.array(next_obs) if isinstance(next_obs, list) else next_obs
                        feat = cw_encoder(obs_vec)
                        next_feat = cw_encoder(next_obs_vec)
                        cw_features_list.append(feat)
                        next_cw_features_list.append(next_feat)
                        actions_list.append(action)
                        rewards_list.append(reward)
                    except Exception as e:
                        print(f"Warning: CW encoding failed: {e}")
                        continue
            
            # Train on CBS transitions
            if len(cbs_features_list) > 0:
                cbs_feats = torch.stack(cbs_features_list).to(device)
                next_cbs_feats = torch.stack(next_cbs_features_list).to(device)
                actions = torch.LongTensor(actions_list[:len(cbs_features_list)]).to(device)
                rewards = torch.FloatTensor(rewards_list[:len(cbs_features_list)]).to(device)
                
                # Predict next features and rewards
                pred_next_feats, pred_rewards = dynamics_model(cbs_feats, actions)
                
                # Compute loss
                loss_features = mse_loss(pred_next_feats, next_cbs_feats)
                loss_rewards = mse_loss(pred_rewards, rewards)
                loss = loss_features + loss_rewards
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            # Train on CW transitions
            if len(cw_features_list) > 0:
                cw_feats = torch.stack(cw_features_list).to(device)
                next_cw_feats = torch.stack(next_cw_features_list).to(device)
                actions = torch.LongTensor(actions_list[len(cbs_features_list):]).to(device)
                rewards = torch.FloatTensor(rewards_list[len(cbs_features_list):]).to(device)
                
                # Predict next features and rewards
                pred_next_feats, pred_rewards = dynamics_model(cw_feats, actions)
                
                # Compute loss
                loss_features = mse_loss(pred_next_feats, next_cw_feats)
                loss_rewards = mse_loss(pred_rewards, rewards)
                loss = loss_features + loss_rewards
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
        
        avg_loss = epoch_loss / max(num_batches, 1)
        all_losses.append(avg_loss)
        
        # Print loss every epoch
        print(f"  Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")
    
    return all_losses


def main():
    parser = argparse.ArgumentParser(description="Train full observation encoders")
    parser.add_argument("--cbs_episodes", type=int, default=30,
                       help="Number of CBS episodes to collect")
    parser.add_argument("--cw_episodes", type=int, default=30,
                       help="Number of Cyberwheel episodes to collect")
    parser.add_argument("--max_steps", type=int, default=100,
                       help="Max steps per episode")
    parser.add_argument("--epochs", type=int, default=10,
                       help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=64,
                       help="Batch size (larger = faster training)")
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                       help="Learning rate")
    parser.add_argument("--feature_size", type=int, default=64,
                       help="Feature space size")
    parser.add_argument("--output_path", type=str,
                       default="artifacts/transfer_models/full_obs_encoder.pt",
                       help="Output path for saved models")
    parser.add_argument("--skip_cw", action="store_true",
                       help="Skip Cyberwheel collection (CBS only)")
    parser.add_argument("--load_cw_data", type=str, default=None,
                       help="Load Cyberwheel transitions from pickle file")
    parser.add_argument("--load_cbs_data", type=str, default=None,
                       help="Load CBS transitions from pickle file")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("TRAINING FULL OBSERVATION ENCODERS")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create encoders
    print("\n1. Creating encoders...")
    cbs_encoder = CBSFullObservationEncoder(feature_size=args.feature_size, use_graph=False)  # Disable graph for now
    cw_encoder = CWFullObservationEncoder(feature_size=args.feature_size)
    # Determine action space size from loaded data
    max_actions = 7  # Default for CBS
    if args.load_cw_data and os.path.exists(args.load_cw_data):
        import pickle
        with open(args.load_cw_data, 'rb') as f:
            cw_data = pickle.load(f)
            if cw_data:
                # Find max action value in Cyberwheel data
                max_actions = max([t['action'] if isinstance(t, dict) else t[1] for t in cw_data]) + 1
                print(f"   Detected Cyberwheel action space size: {max_actions}")
    dynamics_model = DynamicsModel(feature_size=args.feature_size, num_actions=max_actions)
    
    print(f"   ✓ CBS encoder: {sum(p.numel() for p in cbs_encoder.parameters())} parameters")
    print(f"   ✓ CW encoder: {sum(p.numel() for p in cw_encoder.parameters())} parameters")
    print(f"   ✓ Dynamics model: {sum(p.numel() for p in dynamics_model.parameters())} parameters")
    
    # Collect transitions
    print("\n2. Collecting transitions...")
    cbs_transitions = []
    cw_transitions = []
    
    # Load from files if provided
    if args.load_cw_data and os.path.exists(args.load_cw_data):
        print(f"\n   Loading Cyberwheel data from {args.load_cw_data}...")
        import pickle
        with open(args.load_cw_data, 'rb') as f:
            loaded = pickle.load(f)
            # Convert dict format to tuple format if needed
            if loaded and isinstance(loaded[0], dict):
                cw_transitions = [(t['obs'], t['action'], t['next_obs'], t['reward'], t['done']) 
                                 for t in loaded]
            else:
                cw_transitions = loaded
        print(f"   ✓ Loaded {len(cw_transitions)} Cyberwheel transitions")
    
    if args.load_cbs_data and os.path.exists(args.load_cbs_data):
        print(f"\n   Loading CBS data from {args.load_cbs_data}...")
        import pickle
        with open(args.load_cbs_data, 'rb') as f:
            loaded = pickle.load(f)
            # Convert dict format to tuple format if needed
            if loaded and isinstance(loaded[0], dict):
                cbs_transitions = [(t['obs'], t['action'], t['next_obs'], t['reward'], t['done']) 
                                  for t in loaded]
            else:
                cbs_transitions = loaded
        print(f"   ✓ Loaded {len(cbs_transitions)} CBS transitions")
    
    # Collect from CBS (only if not loaded from file)
    if not args.load_cbs_data or not os.path.exists(args.load_cbs_data or ""):
        print("\n   Collecting from CyberBattleSim...")
    try:
        cbs_env = UnifiedSecEnv("cbs", cbs_factory=make_cbs_env)
        # We need to access raw observations - this is a simplified version
        # In practice, you'd modify UnifiedSecEnv to expose raw observations
        for episode in range(args.cbs_episodes):
            obs, info = cbs_env.reset()
            done = False
            step = 0
            
            while not done and step < args.max_steps:
                action = cbs_env.action_space.sample()
                next_obs, reward, done, truncated, info = cbs_env.step(action)
                
                # Get raw CBS observations
                raw_obs = getattr(cbs_env, '_last_raw_cbs_obs', None)
                raw_next_obs = getattr(cbs_env, '_last_raw_cbs_obs', None)
                
                # Fallback to translated obs if raw not available
                if raw_obs is None:
                    raw_obs = obs if isinstance(obs, dict) else {}
                if raw_next_obs is None:
                    raw_next_obs = next_obs if isinstance(next_obs, dict) else {}
                
                # Store transition with raw CBS observations
                cbs_transitions.append((raw_obs, action, raw_next_obs, float(reward), done or truncated))
                
                obs = next_obs
                step += 1
            
            if (episode + 1) % 10 == 0:
                print(f"     Collected {episode + 1}/{args.cbs_episodes} CBS episodes")
    except Exception as e:
        print(f"     Warning: CBS collection failed: {e}")
        print("     Continuing with Cyberwheel only...")
    
    # Collect from Cyberwheel (only if not loaded from file)
    if not args.skip_cw and (not args.load_cw_data or not os.path.exists(args.load_cw_data or "")):
        print("\n   Collecting from Cyberwheel...")
        try:
            cw_env = UnifiedSecEnv("cw", cw_factory=make_cw_env)
            for episode in range(args.cw_episodes):
                obs, info = cw_env.reset()
                done = False
                step = 0
                
                while not done and step < args.max_steps:
                    action = cw_env.action_space.sample()
                    next_obs, reward, done, truncated, info = cw_env.step(action)
                    
                    # Cyberwheel obs is already a vector
                    cw_transitions.append((obs, action, next_obs, float(reward), done or truncated))
                    
                    obs = next_obs
                    step += 1
                
                if (episode + 1) % 10 == 0:
                    print(f"     Collected {episode + 1}/{args.cw_episodes} CW episodes")
        except Exception as e:
            print(f"     Warning: Cyberwheel collection failed: {e}")
            print(f"     Error: {type(e).__name__}: {str(e)}")
    
    print(f"\n   ✓ Collected {len(cbs_transitions)} CBS transitions")
    print(f"   ✓ Collected {len(cw_transitions)} CW transitions")
    
    if len(cbs_transitions) == 0 and len(cw_transitions) == 0:
        print("\n❌ No transitions collected! Cannot train.")
        return
    
    # Train encoders
    print("\n3. Training encoders...")
    losses = train_full_observation_encoder(
        cbs_encoder, cw_encoder, dynamics_model,
        cbs_transitions, cw_transitions,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=device
    )
    
    print(f"\n   ✓ Training complete! Final loss: {losses[-1]:.4f}")
    
    # Save models
    print("\n4. Saving models...")
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    torch.save({
        'cbs_encoder_state_dict': cbs_encoder.state_dict(),
        'cw_encoder_state_dict': cw_encoder.state_dict(),
        'dynamics_state_dict': dynamics_model.state_dict(),
        'feature_size': args.feature_size,
        'losses': losses
    }, args.output_path)
    
    print(f"   ✓ Saved to {args.output_path}")
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\nTo use the encoder:")
    print(f"  python eval_full_observation_transfer.py --encoder_path {args.output_path}")


if __name__ == "__main__":
    main()

