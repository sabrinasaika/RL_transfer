#!/usr/bin/env python3
"""
Complete example showing how to use DAPN from start to finish.
"""

import os
import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("=" * 60)
print("DAPN Usage Example")
print("=" * 60)
print()

# ============================================================================
# STEP 1: Check if encoder exists, if not, show how to train it
# ============================================================================
encoder_path = "artifacts/transfer_models/dapn_encoder.pt"

if not os.path.exists(encoder_path):
    print("Step 1: Train the encoder first")
    print("-" * 60)
    print("Encoder not found. To train it, run:")
    print()
    print("  python train_dapn_encoder.py --num-samples 1000 --epochs 50")
    print()
    print("Or for a quick test:")
    print("  python train_dapn_encoder.py --num-samples 100 --epochs 5")
    print()
    print("Using random initialization for this demo...")
    encoder_path = None
else:
    print("Step 1: Found trained encoder!")
    print("-" * 60)
    print(f"  Encoder: {encoder_path}")
    print()

# ============================================================================
# STEP 2: Use DAPN with Environment Wrapper
# ============================================================================
print("Step 2: Using DAPN with Environment")
print("-" * 60)

from adapters.unified_env import UnifiedSecEnv
from adapters.dapn_env_wrapper import DAPNEnvWrapper
from config.env_builders import make_cbs_env

# Create base environment
base_env = UnifiedSecEnv("cbs", cbs_factory=make_cbs_env)

# Wrap with DAPN
dapn_env = DAPNEnvWrapper(
    base_env,
    encoder_path=encoder_path,  # None = random init, path = trained encoder
    feature_size=256,
    use_dapn=True
)

print(f"  Observation space: {dapn_env.observation_space}")
print(f"  Action space: {dapn_env.action_space}")

# Test the environment
obs, info = dapn_env.reset()
obs_shape = obs['obs'].shape if isinstance(obs, dict) else obs.shape
print(f"  Observation shape: {obs_shape}")
print(f"  ✓ Environment working with DAPN!")
print()

# ============================================================================
# STEP 3: Use DAPN Translator Directly
# ============================================================================
print("Step 3: Using DAPN Translator Directly")
print("-" * 60)

from adapters.dapn_observation_encoder import DAPNObservationTranslator
import numpy as np

translator = DAPNObservationTranslator(
    use_dapn=True,
    encoder_path=encoder_path,
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

encoded_cbs = translator.from_cbs(cbs_obs)
print(f"  CBS observation: 8-dim → {encoded_cbs.shape[0]}-dim")
print(f"  First 5 features: {encoded_cbs[:5]}")

# Example Cyberwheel observation
cw_obs = np.array([1, 1, 1, 1, 0, 0, 0] * 3 + [2])
encoded_cw = translator.from_cw(cw_obs)
print(f"  Cyberwheel observation: variable-dim → {encoded_cw.shape[0]}-dim")
print(f"  First 5 features: {encoded_cw[:5]}")
print(f"  ✓ Both in same 256-dim feature space!")
print()

# ============================================================================
# STEP 4: Train a Policy with DAPN (Optional)
# ============================================================================
print("Step 4: Training Policy with DAPN (Optional)")
print("-" * 60)
print("To train a policy with DAPN, use:")
print()
print("  from stable_baselines3 import PPO")
print("  model = PPO('MultiInputPolicy', dapn_env, verbose=1)")
print("  model.learn(total_timesteps=100000)")
print("  model.save('artifacts/policies/ppo_dapn')")
print()

# ============================================================================
# Summary
# ============================================================================
print("=" * 60)
print("Summary")
print("=" * 60)
print()
print("✓ DAPN is working!")
print()
print("Next steps:")
print("  1. Train encoder: python train_dapn_encoder.py --num-samples 1000 --epochs 50")
print("  2. Use in your code: Wrap environment with DAPNEnvWrapper")
print("  3. Train policy: Use wrapped environment with your RL algorithm")
print()
print("See HOW_TO_USE_DAPN.md for complete guide")
print("=" * 60)

