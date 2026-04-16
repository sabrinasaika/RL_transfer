#!/usr/bin/env python3
"""
Diagnostic script to understand why rewards are constant
"""

import os
import sys
from pathlib import Path
import numpy as np
from stable_baselines3 import PPO
from adapters.unified_env import UnifiedSecEnv
from adapters.dapn_env_wrapper import DAPNEnvWrapper
from config.env_builders import make_cbs_env

# Add paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "cyberwheel"))

def diagnose_reward():
    """Run a single episode with detailed logging"""
    print("=" * 60)
    print("DIAGNOSING CONSTANT REWARD ISSUE")
    print("=" * 60)
    
    # Load model
    model_path = "artifacts/policies/cw_ppo_dapn.zip"
    if not os.path.exists(model_path):
        print(f"✗ Model not found: {model_path}")
        return
    
    print(f"\n1. Loading model: {model_path}")
    model = PPO.load(model_path)
    print("   ✓ Model loaded")
    
    # Create environment
    print("\n2. Creating environment...")
    os.environ["CBS_ENV"] = "CyberBattleFlat-v0"
    os.environ["CBS_FLAT_NODES"] = "20"
    os.environ["CBS_CRED_REUSE_PROB"] = "0.6"
    os.environ["CBS_EXPLOIT_PROB"] = "0.3"
    
    base_env = UnifiedSecEnv("cbs", cbs_factory=make_cbs_env)
    
    dapn_encoder_path = "artifacts/transfer_models/dapn_encoder.pt"
    if os.path.exists(dapn_encoder_path):
        print(f"   Using DAPN encoder: {dapn_encoder_path}")
        env = DAPNEnvWrapper(
            base_env,
            encoder_path=dapn_encoder_path,
            use_dapn=True
        )
    else:
        env = base_env
    
    print("   ✓ Environment created")
    print(f"   Observation space: {env.observation_space}")
    print(f"   Action space: {env.action_space}")
    
    # Run one episode with detailed logging
    print("\n3. Running episode with detailed step-by-step logging...")
    print("-" * 60)
    
    obs, info = env.reset(seed=42)  # Use fixed seed for reproducibility
    total_reward = 0
    steps = 0
    done = False
    truncated = False
    
    step_rewards = []
    step_actions = []
    
    while not (done or truncated) and steps < 100:
        # Get action
        if isinstance(obs, dict):
            obs_for_pred = obs
        else:
            obs_for_pred = obs
        
        # Try both deterministic and non-deterministic
        action_det, _ = model.predict(obs_for_pred, deterministic=True)
        action_stoch, _ = model.predict(obs_for_pred, deterministic=False)
        
        # Check if actions are the same
        action_same = np.array_equal(action_det, action_stoch) if isinstance(action_det, np.ndarray) else action_det == action_stoch
        
        # Use deterministic action
        action = action_det
        
        # Convert to int if it's a numpy array
        if isinstance(action, np.ndarray):
            action = int(action.item()) if action.size == 1 else int(action[0])
        action = int(action)
        
        # Step environment
        obs_next, reward, done, truncated, info_step = env.step(action)
        
        total_reward += reward
        steps += 1
        step_rewards.append(reward)
        step_actions.append(action)
        
        # Log every 10 steps or on interesting events
        if steps <= 10 or steps % 10 == 0 or reward != 0 or done or truncated:
            action_names = ["noop", "ping_sweep", "port_scan", "discovery", "lateral_move", "privilege_escalation", "impact"]
            action_name = action_names[action] if 0 <= action < len(action_names) else f"unknown({action})"
            print(f"   Step {steps:3d}: reward={reward:6.2f}, action={action} ({action_name}), "
                  f"done={done}, truncated={truncated}, total={total_reward:6.2f}")
            if not action_same:
                stoch_action = int(action_stoch.item()) if isinstance(action_stoch, np.ndarray) else int(action_stoch)
                print(f"      ⚠ Deterministic={action} vs stochastic={stoch_action} differ!")
        
        obs = obs_next
    
    print("-" * 60)
    print(f"\nEpisode Summary:")
    print(f"   Total steps: {steps}")
    print(f"   Total reward: {total_reward:.2f}")
    print(f"   Average reward per step: {total_reward/steps:.4f}")
    print(f"   Reward statistics:")
    print(f"      Min: {min(step_rewards):.4f}")
    print(f"      Max: {max(step_rewards):.4f}")
    print(f"      Mean: {np.mean(step_rewards):.4f}")
    print(f"      Std: {np.std(step_rewards):.4f}")
    print(f"      Non-zero rewards: {sum(1 for r in step_rewards if r != 0)}/{len(step_rewards)}")
    
    # Check if all rewards are the same
    unique_rewards = set(step_rewards)
    if len(unique_rewards) == 1:
        print(f"\n   ⚠ WARNING: All rewards are identical: {unique_rewards.pop()}")
    else:
        print(f"\n   Reward distribution: {len(unique_rewards)} unique values")
        print(f"   Unique rewards: {sorted(unique_rewards)[:10]}")  # Show first 10
    
    # Check if all actions are the same
    if len(step_actions) > 1:
        first_action = step_actions[0]
        all_same = all(
            np.array_equal(a, first_action) if isinstance(a, np.ndarray) else a == first_action
            for a in step_actions[1:]
        )
        if all_same:
            print(f"\n   ⚠ WARNING: All actions are identical!")
        else:
            unique_actions = len(set(str(a) for a in step_actions))
            print(f"\n   Action diversity: {unique_actions} unique actions out of {len(step_actions)}")
    
    # Check observation changes
    print(f"\n4. Checking if observations change...")
    obs, info = env.reset(seed=42)  # Reset with same seed
    obs_initial = obs.copy() if isinstance(obs, dict) else obs
    
    obs, reward, done, truncated, info = env.step(step_actions[0] if step_actions else 0)
    obs_after_step = obs.copy() if isinstance(obs, dict) else obs
    
    if isinstance(obs_initial, dict):
        obs_changed = not all(
            np.array_equal(obs_initial[k], obs_after_step[k])
            for k in obs_initial.keys()
        )
    else:
        obs_changed = not np.array_equal(obs_initial, obs_after_step)
    
    if not obs_changed:
        print("   ⚠ WARNING: Observations don't change after taking action!")
    else:
        print("   ✓ Observations change after actions")
    
    print("\n" + "=" * 60)
    print("DIAGNOSIS COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    diagnose_reward()
