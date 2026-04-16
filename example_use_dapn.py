#!/usr/bin/env python3
"""
Example script showing how to use DAPN for observation handling.

This demonstrates:
1. Creating an environment with DAPN observation encoding
2. Using DAPN with existing code
3. Training with DAPN-encoded observations
"""

import os
import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import numpy as np
from stable_baselines3 import PPO
from adapters.unified_env import UnifiedSecEnv
from adapters.dapn_env_wrapper import DAPNEnvWrapper
from adapters.dapn_observation_encoder import DAPNObservationTranslator
from config.env_builders import make_cbs_env


def example_1_basic_usage():
    """Example 1: Basic usage with DAPN wrapper."""
    print("=" * 60)
    print("Example 1: Basic DAPN Usage")
    print("=" * 60)
    
    # Create base environment
    base_env = UnifiedSecEnv("cbs", cbs_factory=make_cbs_env)
    
    # Wrap with DAPN (without pre-trained encoder - uses random initialization)
    dapn_env = DAPNEnvWrapper(
        base_env,
        encoder_path=None,  # No pre-trained encoder
        feature_size=256,
        use_dapn=True
    )
    
    # Test the environment
    obs, info = dapn_env.reset()
    print(f"Observation shape: {obs['obs'].shape if isinstance(obs, dict) else obs.shape}")
    print(f"Observation space: {dapn_env.observation_space}")
    
    # Take a step
    action = dapn_env.action_space.sample()
    obs, reward, done, truncated, info = dapn_env.step(action)
    print(f"Step successful! Reward: {reward}")
    print()


def example_2_with_pretrained_encoder():
    """Example 2: Using DAPN with a pre-trained encoder."""
    print("=" * 60)
    print("Example 2: Using Pre-trained DAPN Encoder")
    print("=" * 60)
    
    # Path to your trained DAPN encoder
    encoder_path = "artifacts/transfer_models/dapn_encoder.pt"
    
    if os.path.exists(encoder_path):
        # Create environment with pre-trained DAPN encoder
        base_env = UnifiedSecEnv("cbs", cbs_factory=make_cbs_env)
        dapn_env = DAPNEnvWrapper(
            base_env,
            encoder_path=encoder_path,
            feature_size=256,
            use_dapn=True
        )
        
        print(f"Loaded DAPN encoder from {encoder_path}")
        obs, info = dapn_env.reset()
        print(f"Observation shape: {obs['obs'].shape if isinstance(obs, dict) else obs.shape}")
    else:
        print(f"Encoder not found at {encoder_path}")
        print("Train a DAPN encoder first using train_dapn_encoder.py")
    print()


def example_3_direct_translator():
    """Example 3: Using DAPN translator directly."""
    print("=" * 60)
    print("Example 3: Direct DAPN Translator Usage")
    print("=" * 60)
    
    # Create DAPN translator
    translator = DAPNObservationTranslator(
        use_dapn=True,
        encoder_path=None,
        feature_size=256
    )
    
    # Example CBS observation
    cbs_obs = {
        "discovered_node_count": 5,
        "nodes_privilegelevel": np.array([0, 1, 0, 1, 0]),
        "discovered_nodes_properties": np.array([[1, 2, 3], [4, 5, 6]]),
        "credential_cache_length": 2,
        "probe_result": 1,
        "escalation": 0
    }
    
    # Encode CBS observation
    encoded_cbs = translator.from_cbs(cbs_obs)
    print(f"CBS observation encoded to shape: {encoded_cbs.shape}")
    print(f"Encoded features (first 10): {encoded_cbs[:10]}")
    
    # Example Cyberwheel observation
    cw_obs = np.array([1, 1, 1, 1, 0, 0, 0] * 3 + [2])  # 3 hosts + 1 standalone
    
    # Encode Cyberwheel observation
    encoded_cw = translator.from_cw(cw_obs)
    print(f"Cyberwheel observation encoded to shape: {encoded_cw.shape}")
    print(f"Encoded features (first 10): {encoded_cw[:10]}")
    print()


def example_4_training_with_dapn():
    """Example 4: Training a policy with DAPN-encoded observations."""
    print("=" * 60)
    print("Example 4: Training Policy with DAPN")
    print("=" * 60)
    
    # Create environment with DAPN
    base_env = UnifiedSecEnv("cbs", cbs_factory=make_cbs_env)
    dapn_env = DAPNEnvWrapper(
        base_env,
        encoder_path=None,  # Can use pre-trained encoder here
        feature_size=256,
        use_dapn=True
    )
    
    # Create PPO policy
    # Note: For Dict observation spaces, use MultiInputPolicy
    import gymnasium as gym
    
    if isinstance(dapn_env.observation_space, gym.spaces.Dict):
        policy_kwargs = dict(
            net_arch=[dict(pi=[256, 256], vf=[256, 256])]
        )
        model = PPO(
            "MultiInputPolicy",
            dapn_env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            n_steps=2048,
            batch_size=64
        )
    else:
        model = PPO(
            "MlpPolicy",
            dapn_env,
            verbose=1,
            n_steps=2048,
            batch_size=64
        )
    
    print("Training PPO with DAPN-encoded observations...")
    print("(This is just a demonstration - actual training would take longer)")
    
    # Train for a few steps as demonstration
    model.learn(total_timesteps=1000)
    
    print("Training complete!")
    print()


def example_5_comparing_with_without_dapn():
    """Example 5: Compare observations with and without DAPN."""
    print("=" * 60)
    print("Example 5: Comparing Observations With/Without DAPN")
    print("=" * 60)
    
    # Create environment without DAPN
    env_no_dapn = UnifiedSecEnv("cbs", cbs_factory=make_cbs_env)
    obs_no_dapn, _ = env_no_dapn.reset()
    
    # Create environment with DAPN
    env_dapn = DAPNEnvWrapper(
        UnifiedSecEnv("cbs", cbs_factory=make_cbs_env),
        use_dapn=True,
        feature_size=256
    )
    obs_dapn, _ = env_dapn.reset()
    
    # Extract observation vectors
    obs_vec_no_dapn = obs_no_dapn['obs'] if isinstance(obs_no_dapn, dict) else obs_no_dapn
    obs_vec_dapn = obs_dapn['obs'] if isinstance(obs_dapn, dict) else obs_dapn
    
    print(f"Without DAPN: shape={obs_vec_no_dapn.shape}, dtype={obs_vec_no_dapn.dtype}")
    print(f"  First 10 values: {obs_vec_no_dapn[:10]}")
    print(f"With DAPN: shape={obs_vec_dapn.shape}, dtype={obs_vec_dapn.dtype}")
    print(f"  First 10 values: {obs_vec_dapn[:10]}")
    print()


if __name__ == "__main__":
    import gymnasium as gym
    
    print("\n" + "=" * 60)
    print("DAPN Observation Handling Examples")
    print("=" * 60 + "\n")
    
    try:
        example_1_basic_usage()
        example_2_with_pretrained_encoder()
        example_3_direct_translator()
        # Uncomment to run training example (takes longer)
        # example_4_training_with_dapn()
        example_5_comparing_with_without_dapn()
        
        print("=" * 60)
        print("All examples completed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()

