#!/usr/bin/env python3
"""
Quick demo of transfer learning - shows the concept without full training.
Creates a simple encoder, evaluates on CBS with/without transfer.
"""

import os
import sys
import json
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
from adapters.unified_env import UnifiedSecEnv
from adapters.observation_translator import ObservationTranslator
from adapters.transfer_encoder import ObservationEncoder, DynamicsModel, save_transfer_models
from adapters.transfer_training import ReplayBuffer, train_dynamics_model
from config.env_builders import make_cbs_env


def quick_demo():
    """Quick demo showing transfer learning concept"""
    print("=" * 60)
    print("QUICK TRANSFER LEARNING DEMO")
    print("=" * 60)
    
    # Step 1: Create encoder and dynamics model
    print("\n1. Creating encoder and dynamics model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = ObservationEncoder(input_dim=8, feature_size=64).to(device)
    dynamics_model = DynamicsModel(feature_size=64, num_actions=7).to(device)
    print("   ✓ Encoder: 8-dim → 64-dim features")
    print("   ✓ Dynamics model: Predicts next features + rewards")
    
    # Step 2: Quick training on CBS (collect some data)
    print("\n2. Collecting transitions and training dynamics model...")
    env = UnifiedSecEnv("cbs", cbs_factory=make_cbs_env)
    replay_buffer = ReplayBuffer(capacity=1000)
    
    # Collect 10 episodes quickly
    for episode in range(10):
        obs, _ = env.reset()
        done = False
        step = 0
        
        while not done and step < 50:  # Short episodes
            if isinstance(obs, dict):
                obs_raw = obs.get("obs", obs)
            else:
                obs_raw = obs
            
            action = env.action_space.sample()
            next_obs, reward, done, truncated, info = env.step(action)
            
            if isinstance(next_obs, dict):
                next_obs_raw = next_obs.get("obs", next_obs)
            else:
                next_obs_raw = next_obs
            
            replay_buffer.push(obs_raw, action, next_obs_raw, float(reward), done or truncated)
            obs = next_obs
            step += 1
    
    print(f"   ✓ Collected {len(replay_buffer)} transitions")
    
    # Quick dynamics training
    if len(replay_buffer) > 32:
        losses = train_dynamics_model(
            encoder, dynamics_model, replay_buffer,
            batch_size=32, num_epochs=5, device=device
        )
        print(f"   ✓ Dynamics model trained (final loss: {losses[-1]:.4f})")
    
    # Save models
    encoder_path = "artifacts/transfer_models/demo_encoder.pt"
    os.makedirs(os.path.dirname(encoder_path), exist_ok=True)
    save_transfer_models(encoder, dynamics_model, encoder_path, input_dim=8, feature_size=64, num_actions=7)
    print(f"   ✓ Saved to {encoder_path}")
    
    # Step 3: Demonstrate observation encoding
    print("\n3. Demonstrating observation encoding...")
    obs_t_no_transfer = ObservationTranslator(use_transfer=False)
    obs_t_with_transfer = ObservationTranslator(use_transfer=True, encoder_path=encoder_path)
    
    # Get a sample observation
    obs, _ = env.reset()
    if isinstance(obs, dict):
        obs_dict = obs
        obs_raw = obs.get("obs", obs)
    else:
        obs_raw = obs
        obs_dict = {"obs": obs}
    
    # Convert using CBS translator
    cbs_obs = {
        "discovered_node_count": 3,
        "nodes_privilegelevel": np.array([1, 0, 0], dtype=np.int32),
        "discovered_nodes_properties": np.zeros((3, 3), dtype=np.int32),
        "credential_cache_length": 1,
        "_explored_network": type('obj', (object,), {'number_of_edges': lambda: 5})(),
        "probe_result": 0,
        "escalation": 0,
    }
    
    obs_8d = obs_t_no_transfer.from_cbs(cbs_obs)
    obs_64d = obs_t_with_transfer.from_cbs(cbs_obs)
    
    print(f"   Without transfer: {obs_8d.shape} (raw observation)")
    print(f"   With transfer:    {obs_64d.shape} (encoded features)")
    print(f"   ✓ Observations successfully encoded to feature space")
    
    # Step 4: Show the difference
    print("\n4. Key Differences:")
    print("   • Without transfer: Policy sees 8-dim raw observations")
    print("   • With transfer:    Policy sees 64-dim learned features")
    print("   • Features capture task-agnostic patterns")
    print("   • Better generalization across environments")
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETE!")
    print("=" * 60)
    print(f"\nEncoder saved to: {encoder_path}")
    print("\nTo use in evaluation:")
    print(f"  python eval/eval_cw_checkpoints_on_cbs.py")
    print("  (and modify to use encoder_path)")
    print("\nOr use the encoder in your training:")
    print("  from adapters.observation_translator import ObservationTranslator")
    print(f"  obs_t = ObservationTranslator(use_transfer=True, encoder_path='{encoder_path}')")


if __name__ == "__main__":
    quick_demo()

