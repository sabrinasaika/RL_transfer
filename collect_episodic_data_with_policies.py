#!/usr/bin/env python3
"""
Collect observations for episodic DAPN training using trained policies.

This script collects observations from Cyberwheel (source) and CyberBattleSim (target)
using trained PPO policies instead of random actions. This gives better state
distribution and more balanced classes for few-shot learning.

Usage:
    python collect_episodic_data_with_policies.py \
        --cw-policy artifacts/policies/cw_ppo_dapn.zip \
        --cbs-policy artifacts/policies/cbs_ppo_final.zip \
        --cw-episodes 50 \
        --cbs-episodes 50 \
        --max-steps 200 \
        --output artifacts/training_data/episodic_obs_policy.npz
"""

import os
import sys
import argparse
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Add cyberwheel to path (needed for imports)
cyberwheel_path = project_root / "cyberwheel"
if cyberwheel_path.exists():
    sys.path.insert(0, str(cyberwheel_path))

from adapters.unified_env import UnifiedSecEnv
from config.env_builders import make_cbs_env, make_cw_env


def load_ppo_policy(policy_path, env=None):
    """Load a Stable-Baselines3 PPO policy without environment (handles obs space mismatches)."""
    try:
        from stable_baselines3 import PPO
        import warnings
        
        if not os.path.exists(policy_path):
            print(f"  Warning: Policy not found at {policy_path}")
            return None
        
        print(f"  Loading PPO policy from {policy_path}...")
        
        # Suppress warnings about deserialization (these are usually non-fatal)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", message=".*Could not deserialize.*")
            warnings.filterwarnings("ignore", message=".*code expected at most.*")
            warnings.filterwarnings("ignore", message=".*Observation spaces do not match.*")
            
            # Always load without env to avoid observation space mismatch errors
            # We'll handle observation formatting manually during prediction
            policy = PPO.load(policy_path, print_system_info=False)
        
        print(f"  ✓ Loaded PPO policy (will handle observation format conversion during prediction)")
        return policy
    except Exception as e:
        print(f"  Error: Could not load PPO policy: {e}")
        return None


def load_cyberwheel_policy(policy_path):
    """Load a Cyberwheel native policy (RLPolicy)."""
    try:
        import torch
        from cyberwheel.utils import RLPolicy
        from eval.eval_cw_checkpoints_on_cbs import infer_cyberwheel_config
        
        if not os.path.exists(policy_path):
            print(f"  Warning: Policy not found at {policy_path}")
            return None
        
        print(f"  Loading Cyberwheel policy from {policy_path}...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        action_space_size, obs_space_shape = infer_cyberwheel_config(policy_path)
        policy = RLPolicy(action_space_shape=action_space_size, obs_space_shape=obs_space_shape).to(device)
        state_dict = torch.load(policy_path, map_location=device)
        policy.load_state_dict(state_dict)
        policy.eval()
        print(f"  ✓ Loaded Cyberwheel RLPolicy (action_space={action_space_size}, obs_shape={obs_space_shape})")
        return policy, device
    except Exception as e:
        print(f"  Warning: Could not load Cyberwheel policy: {e}")
        return None, None


def collect_with_policy_episodic(
    cw_policy_path=None,
    cbs_policy_path=None,
    cw_episodes=50,
    cbs_episodes=50,
    max_steps_per_episode=200,
    save_path=None,
    val_fraction=0.2,
    seed=None,
    epsilon=0.0,  # Epsilon for epsilon-greedy (0.0 = pure policy, 1.0 = pure random)
):
    """
    Collect observations using trained policies.
    
    Args:
        cw_policy_path: Path to Cyberwheel policy (PPO .zip or Cyberwheel .pt)
        cbs_policy_path: Path to CBS policy (PPO .zip)
        cw_episodes: Number of Cyberwheel episodes
        cbs_episodes: Number of CBS episodes
        max_steps_per_episode: Maximum steps per episode
        save_path: Path to save observations
        val_fraction: Fraction for validation split
        seed: Random seed
        epsilon: Epsilon for epsilon-greedy exploration (0.0 = pure policy)
    """
    print("=" * 80)
    print("COLLECTING OBSERVATIONS WITH TRAINED POLICIES")
    print("=" * 80)
    print(f"Source domain: Cyberwheel")
    print(f"Target domain: CyberBattleSim")
    print(f"⚠️  Policies are REQUIRED - no random actions will be used")
    if seed is not None:
        print(f"Random seed: {seed}")
    print("=" * 80)
    
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # ---- Source (Cyberwheel) ----
    print("\n📊 Collecting source domain (Cyberwheel) observations...")
    source_obs_list = []
    source_actions_list = []
    
    try:
        cw_env = UnifiedSecEnv("cw", cw_factory=make_cw_env)
        
        # Load policy
        cw_policy = None
        cw_policy_type = None
        cw_device = None
        
        if cw_policy_path:
            if cw_policy_path.endswith('.zip'):
                # Try PPO policy (load without env to avoid obs space mismatch)
                cw_policy = load_ppo_policy(cw_policy_path, env=None)
                cw_policy_type = 'ppo'
            elif cw_policy_path.endswith('.pt'):
                # Try Cyberwheel native policy
                cw_policy, cw_device = load_cyberwheel_policy(cw_policy_path)
                cw_policy_type = 'cyberwheel'
        
        if cw_policy is None:
            if cw_policy_path:
                raise ValueError(f"❌ ERROR: Could not load Cyberwheel policy from {cw_policy_path}. Policy is required (no random actions).")
            else:
                raise ValueError("❌ ERROR: No Cyberwheel policy provided. Use --cw-policy to specify a policy (random actions not allowed).")
        
        print(f"  Collecting from {cw_episodes} episodes (max {max_steps_per_episode} steps each)...")
        samples_collected = 0
        
        for episode in range(cw_episodes):
            obs, _ = cw_env.reset()
            done = False
            truncated = False
            step = 0
            
            while not (done or truncated) and step < max_steps_per_episode:
                # Get raw observation
                raw = getattr(cw_env, "_last_raw_obs", None)
                if raw is None:
                    raw = obs if isinstance(obs, np.ndarray) else np.array([], dtype=np.float32)
                
                # Get action from policy (required, no random actions)
                if cw_policy_type == 'ppo':
                    # Stable-Baselines3 PPO - handle observation format conversion
                    # Policy might expect Box(8,) but env gives Dict or Box(256,)
                    # Extract the actual observation value
                    if isinstance(obs, dict):
                        # If env gives Dict, extract 'obs' key
                        policy_obs = obs.get('obs', obs)
                    else:
                        policy_obs = obs
                    
                    # If policy expects 8D but we have 256D (DAPN encoded), use raw 8D
                    # Check policy's expected obs space
                    if hasattr(cw_policy, 'observation_space'):
                        expected_shape = cw_policy.observation_space.shape if hasattr(cw_policy.observation_space, 'shape') else None
                        if expected_shape and len(expected_shape) > 0 and expected_shape[0] == 8:
                            # Policy expects 8D - use unified observation translator
                            from adapters.observation_translator import ObservationTranslator
                            translator = ObservationTranslator()
                            policy_obs = translator.from_cw(raw) if raw is not None else obs
                    
                    action, _ = cw_policy.predict(policy_obs, deterministic=False)
                elif cw_policy_type == 'cyberwheel':
                    # Cyberwheel native policy
                    import torch
                    obs_tensor = torch.from_numpy(np.array(raw, dtype=np.float32)).float().unsqueeze(0).to(cw_device)
                    with torch.no_grad():
                        action_tensor, _, _, _ = cw_policy.get_action_and_value(obs_tensor)
                        action = int(action_tensor.cpu().numpy()[0])
                else:
                    raise RuntimeError("Policy not loaded but required!")
                
                # Store observation and action
                source_obs_list.append(raw)
                source_actions_list.append(int(action))
                
                # Take step
                obs, reward, done, truncated, info = cw_env.step(action)
                step += 1
                samples_collected += 1
            
            if (episode + 1) % 10 == 0:
                print(f"    Episode {episode + 1}/{cw_episodes}: {samples_collected} samples collected")
        
        print(f"  ✅ Collected {len(source_obs_list)} source (CW) samples.")
    except Exception as e:
        print(f"  ❌ Warning: could not collect Cyberwheel obs: {e}")
        import traceback
        traceback.print_exc()
        source_obs_list = []
        source_actions_list = []

    # ---- Target (CyberBattleSim) ----
    print("\n📊 Collecting target domain (CyberBattleSim) observations...")
    target_obs_list = []
    target_actions_list = []
    
    try:
        cbs_env = UnifiedSecEnv("cbs", cbs_factory=make_cbs_env)
        
        # Load policy (without env to avoid obs space mismatch)
        cbs_policy = None
        if cbs_policy_path:
            cbs_policy = load_ppo_policy(cbs_policy_path, env=None)
        
        if cbs_policy is None:
            if cbs_policy_path:
                raise ValueError(f"❌ ERROR: Could not load CBS policy from {cbs_policy_path}. Policy is required (no random actions).")
            else:
                raise ValueError("❌ ERROR: No CBS policy provided. Use --cbs-policy to specify a policy (random actions not allowed).")
        
        print(f"  Collecting from {cbs_episodes} episodes (max {max_steps_per_episode} steps each)...")
        samples_collected = 0
        
        for episode in range(cbs_episodes):
            obs, _ = cbs_env.reset()
            done = False
            truncated = False
            step = 0
            
            while not (done or truncated) and step < max_steps_per_episode:
                # Get raw observation
                raw = getattr(cbs_env, "_last_raw_cbs_obs", None) or getattr(cbs_env, "_last_raw_obs", None)
                if raw is None:
                    raw = obs if isinstance(obs, dict) else {}
                
                # Get action from policy (required, no random actions)
                # Handle observation format conversion for CBS
                # Check what the policy expects and convert accordingly
                policy_obs_space = getattr(cbs_policy, 'observation_space', None)
                
                if policy_obs_space is not None:
                    # Check if policy expects Dict or Box
                    if hasattr(policy_obs_space, 'spaces'):  # Dict space (has .spaces attribute)
                        # Policy expects Dict
                        if isinstance(obs, dict):
                            policy_obs = obs
                        else:
                            # Convert Box to Dict with mask
                            from adapters.action_translator import ActionTranslator
                            num_actions = len(ActionTranslator().unified_actions)
                            mask = np.ones(num_actions, dtype=np.float32)
                            policy_obs = {'obs': obs, 'mask': mask}
                    else:
                        # Policy expects Box (simple array) - extract from dict if needed
                        if isinstance(obs, dict):
                            # Extract 'obs' value from dict
                            policy_obs = obs.get('obs', obs)
                            # Ensure it's a numpy array
                            if not isinstance(policy_obs, np.ndarray):
                                policy_obs = np.array(policy_obs, dtype=np.float32)
                        else:
                            policy_obs = obs
                else:
                    # Fallback: if obs is dict, extract 'obs', otherwise use as-is
                    if isinstance(obs, dict):
                        policy_obs = obs.get('obs', obs)
                        if not isinstance(policy_obs, np.ndarray):
                            policy_obs = np.array(policy_obs, dtype=np.float32)
                    else:
                        policy_obs = obs
                
                action, _ = cbs_policy.predict(policy_obs, deterministic=False)
                
                # Store observation and action
                target_obs_list.append(raw)
                target_actions_list.append(int(action))
                
                # Take step
                obs, reward, done, truncated, info = cbs_env.step(action)
                step += 1
                samples_collected += 1
            
            if (episode + 1) % 10 == 0:
                print(f"    Episode {episode + 1}/{cbs_episodes}: {samples_collected} samples collected")
        
        print(f"  ✅ Collected {len(target_obs_list)} target (CBS) samples.")
    except Exception as e:
        print(f"  ❌ Warning: could not collect CBS obs: {e}")
        import traceback
        traceback.print_exc()
        target_obs_list = []
        target_actions_list = []

    # ---- Validation split from target ----
    val_obs_list = []
    if val_fraction > 0 and len(target_obs_list) > 0:
        n_val = max(1, int(len(target_obs_list) * val_fraction))
        val_obs_list = target_obs_list[:n_val]
        target_obs_list = target_obs_list[n_val:]
        target_actions_list = target_actions_list[n_val:]
        print(f"\n  📊 Validation split: {len(val_obs_list)} samples; target train: {len(target_obs_list)} samples")

    # Ensure actions match observations
    source_actions_list = source_actions_list[:len(source_obs_list)]
    target_actions_list = target_actions_list[:len(target_obs_list)]

    # Shuffle
    if len(source_obs_list) > 0:
        idxs = list(range(len(source_obs_list)))
        random.shuffle(idxs)
        source_obs_list = [source_obs_list[i] for i in idxs]
        source_actions_list = [source_actions_list[i] for i in idxs]
    if len(target_obs_list) > 0:
        idxs = list(range(len(target_obs_list)))
        random.shuffle(idxs)
        target_obs_list = [target_obs_list[i] for i in idxs]
        target_actions_list = [target_actions_list[i] for i in idxs]

    # Save
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        out = {
            "source_obs": source_obs_list,
            "target_obs": target_obs_list,
            "val_obs": val_obs_list,
        }
        if source_actions_list:
            out["source_actions"] = source_actions_list
        if target_actions_list:
            out["target_actions"] = target_actions_list
        # Use compressed format to save space (raw CBS observations are huge!)
        print(f"\n💾 Saving data (using compressed format to save space)...")
        np.savez_compressed(save_path, **out, allow_pickle=True)
        print(f"\n✅ Saved observations to {save_path}")
        print(f"   - Source (CW): {len(source_obs_list)} samples")
        print(f"   - Target (CBS): {len(target_obs_list)} samples")
        print(f"   - Validation: {len(val_obs_list)} samples")
        print(f"   - Total: {len(source_obs_list) + len(target_obs_list) + len(val_obs_list)} samples")

    # Calculate requirements check
    print("\n" + "=" * 80)
    print("EPISODIC TRAINING REQUIREMENTS CHECK")
    print("=" * 80)
    
    n_sc = 20
    k = 5
    query = 15
    required_source = n_sc * (k + query)
    
    n_dc = 5
    required_target = n_dc * (k + query)
    
    print(f"\nFor n_sc={n_sc}, k={k}, query={query}:")
    print(f"  Source domain needs: {required_source} samples per episode")
    print(f"  Your source samples: {len(source_obs_list)}")
    if len(source_obs_list) >= required_source:
        num_episodes = len(source_obs_list) // required_source
        print(f"  ✅ Can form ~{num_episodes} episodes")
    else:
        shortfall = required_source - len(source_obs_list)
        print(f"  ❌ Need {shortfall} more samples (or use --n-sc {len(source_obs_list) // (k + query)})")
    
    print(f"\nFor n_dc={n_dc}, k={k}, query={query}:")
    print(f"  Target domain needs: {required_target} samples per episode")
    print(f"  Your target samples: {len(target_obs_list)}")
    if len(target_obs_list) >= required_target:
        num_episodes = len(target_obs_list) // required_target
        print(f"  ✅ Can form ~{num_episodes} episodes")
    else:
        shortfall = required_target - len(target_obs_list)
        print(f"  ❌ Need {shortfall} more samples")
    
    print("=" * 80)

    return (
        source_obs_list,
        target_obs_list,
        val_obs_list,
        source_actions_list if source_actions_list else None,
        target_actions_list if target_actions_list else None,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Collect observations for episodic DAPN training using trained policies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Collect with trained policies
  python collect_episodic_data_with_policies.py \\
      --cw-policy artifacts/policies/cw_ppo_dapn.zip \\
      --cbs-policy artifacts/policies/cbs_ppo_final.zip \\
      --cw-episodes 50 \\
      --cbs-episodes 50 \\
      --max-steps 200 \\
      --output artifacts/training_data/episodic_obs_policy.npz
  
  # Collect with epsilon-greedy (70% policy, 30% random)
  python collect_episodic_data_with_policies.py \\
      --cw-policy artifacts/policies/cw_ppo_dapn.zip \\
      --epsilon 0.3 \\
      --cw-episodes 50 \\
      --output artifacts/training_data/episodic_obs_eps03.npz
  
  # Collect with Cyberwheel native policy (.pt file)
  python collect_episodic_data_with_policies.py \\
      --cw-policy artifacts/policies/cw_policy.pt \\
      --cw-episodes 50 \\
      --output artifacts/training_data/episodic_obs_cw_native.npz
        """
    )
    parser.add_argument("--cw-policy", type=str, default=None, required=True,
                       help="Path to Cyberwheel policy (PPO .zip or Cyberwheel .pt) - REQUIRED")
    parser.add_argument("--cbs-policy", type=str, default=None, required=True,
                       help="Path to CBS policy (PPO .zip) - REQUIRED")
    parser.add_argument("--cw-episodes", type=int, default=50,
                       help="Number of Cyberwheel episodes to collect")
    parser.add_argument("--cbs-episodes", type=int, default=50,
                       help="Number of CBS episodes to collect")
    parser.add_argument("--max-steps", type=int, default=200,
                       help="Maximum steps per episode")
    parser.add_argument("--output", "-o", type=str,
                       default="artifacts/training_data/episodic_obs_policy.npz",
                       help="Output file path (.npz format)")
    parser.add_argument("--val-fraction", type=float, default=0.2,
                       help="Fraction of target samples to use for validation")
    parser.add_argument("--epsilon", type=float, default=0.0,
                       help="DEPRECATED: Policies are now required. This parameter is ignored.")
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    collect_with_policy_episodic(
        cw_policy_path=args.cw_policy,
        cbs_policy_path=args.cbs_policy,
        cw_episodes=args.cw_episodes,
        cbs_episodes=args.cbs_episodes,
        max_steps_per_episode=args.max_steps,
        save_path=args.output,
        val_fraction=args.val_fraction,
        seed=args.seed,
        epsilon=args.epsilon,
    )
    
    print("\n✅ Data collection complete!")
    print(f"\nTo train with this data:")
    print(f"  python train_dapn_encoder_episodic.py --load-data {args.output} --n-sc 20")


if __name__ == "__main__":
    main()
