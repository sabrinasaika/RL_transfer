#!/usr/bin/env python3
"""
Collect more training data for the full observation encoder.
Runs more episodes to gather more transitions for better encoder training.
"""

import os
import sys
import argparse
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
# Add cyberwheel to path for imports
cyberwheel_path = project_root / "cyberwheel"
if cyberwheel_path.exists():
    sys.path.insert(0, str(cyberwheel_path))

import numpy as np
from adapters.unified_env import UnifiedSecEnv
from config.env_builders import make_cbs_env, make_cw_env


def collect_cbs_data(num_episodes: int = 10, max_steps: int = 200, output_file: str = None):
    """Collect CBS transitions and save to file"""
    print("COLLECTING CBS TRAINING DATA")
    
    env = UnifiedSecEnv("cbs", cbs_factory=make_cbs_env)
    transitions = []
    
    print(f"\nCollecting {num_episodes} episodes (max {max_steps} steps each)...")
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        step = 0
        
        while not (done or truncated) and step < max_steps:
            # Get raw CBS observation
            raw_obs = getattr(env, '_last_raw_cbs_obs', None)
            if raw_obs is None:
                # Fallback: try to get from env
                if hasattr(env, 'env') and hasattr(env.env, '_last_observation'):
                    raw_obs = env.env._last_observation
                else:
                    # Use translated obs as fallback
                    raw_obs = obs if isinstance(obs, dict) else {}
            
            # Random action
            if isinstance(obs, dict) and "mask" in obs:
                mask = obs["mask"]
                valid_actions = np.where(mask > 0.5)[0]
                if len(valid_actions) > 0:
                    action = np.random.choice(valid_actions)
                else:
                    action = env.action_space.sample()
            else:
                action = env.action_space.sample()
            
            next_obs, reward, done, truncated, info = env.step(action)
            
            # Get next raw obs
            raw_next_obs = getattr(env, '_last_raw_cbs_obs', None)
            if raw_next_obs is None:
                raw_next_obs = next_obs if isinstance(next_obs, dict) else {}
            
            # Store transition
            transitions.append({
                'obs': raw_obs,
                'action': int(action),
                'next_obs': raw_next_obs,
                'reward': float(reward),
                'done': bool(done or truncated)
            })
            
            obs = next_obs
            step += 1
        
        if (episode + 1) % 20 == 0:
            print(f"  Collected {episode + 1}/{num_episodes} episodes ({len(transitions)} transitions)")
    
    print(f"\n Collected {len(transitions)} CBS transitions")
    
    if output_file:
        import pickle
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'wb') as f:
            pickle.dump(transitions, f)
        print(f" Saved to {output_file}")
    
    return transitions


def collect_cw_data(num_episodes: int = 100, max_steps: int = 200, output_file: str = None, 
                    cw_checkpoint_path: str = None):
    """Collect Cyberwheel transitions and save to file
    
    Args:
        num_episodes: Number of episodes to collect
        max_steps: Max steps per episode
        output_file: Path to save transitions
        cw_checkpoint_path: Path to Cyberwheel checkpoint. If None, uses random actions.
    """
   
    
    # Load Cyberwheel agent if checkpoint provided
    policy = None
    use_direct_env = False
    if cw_checkpoint_path and os.path.exists(cw_checkpoint_path):
        try:
            import torch
            from cyberwheel.utils import RLPolicy
            from eval.eval_cw_checkpoints_on_cbs import infer_cyberwheel_config
            
            print(f"\n  Loading Cyberwheel agent from {cw_checkpoint_path}...")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            action_space_size, obs_space_shape = infer_cyberwheel_config(cw_checkpoint_path)
            policy = RLPolicy(action_space_shape=action_space_size, obs_space_shape=obs_space_shape).to(device)
            state_dict = torch.load(cw_checkpoint_path, map_location=device)
            policy.load_state_dict(state_dict)
            policy.eval()
            print(f"  ✓ Loaded Cyberwheel RLPolicy (action_space={action_space_size}, obs_shape={obs_space_shape})")
            use_direct_env = True  # Need direct env access for 701-dim obs
            print("  Using Cyberwheel environment directly to access 701-dim observations")
        except Exception as e:
            print(f"  Warning: Could not load Cyberwheel agent: {e}")
            print("  Falling back to random actions")
            policy = None
    
    # Create environment - use direct Cyberwheel env if using agent, otherwise UnifiedSecEnv
    if use_direct_env and policy is not None:
        try:
            # Create Cyberwheel environment directly (bypass UnifiedSecEnv)
            from config.env_builders import make_cw_env
            cw_env = make_cw_env()
            env = None  # We'll use cw_env directly
            
        except Exception as e:
            print(f"  Warning: Could not create direct Cyberwheel env: {e}")
            print("  Falling back to UnifiedSecEnv with random actions")
            use_direct_env = False
            policy = None
    
    if not use_direct_env:
        try:
            env = UnifiedSecEnv("cw", cw_factory=make_cw_env)
            print("\n  Using UnifiedSecEnv (random actions)")
        except Exception as e:
            import traceback
            print(f"  Warning: Could not create Cyberwheel environment: {e}")
            print(f"  Full traceback:")
            traceback.print_exc()
            print("  Skipping Cyberwheel data collection")
            return []
    
    transitions = []
    
    print(f"\nCollecting {num_episodes} episodes (max {max_steps} steps each)...")
    
    for episode in range(num_episodes):
        try:
            if use_direct_env:
                # Use Cyberwheel environment directly
                obs_dict, info = cw_env.reset()
                # Extract red agent observation (701-dim vector)
                # Check if red agent is RL-enabled (has observation)
                if isinstance(obs_dict, dict):
                    obs = obs_dict.get("red")
                    if obs is None:
                        # Red agent is not RL-enabled, can't use policy
                        print(f"  Warning: Red agent is not RL-enabled (obs is None). Cannot use Cyberwheel agent.")
                        policy = None  # Disable policy for this episode
                        obs = np.zeros(701, dtype=np.float32)  # Placeholder
                else:
                    obs = obs_dict
            else:
                obs, info = env.reset()
            
            done = False
            truncated = False
            step = 0
            
            while not (done or truncated) and step < max_steps:
                # Use Cyberwheel agent if available
                if policy is not None and use_direct_env:
                    import torch
                    # Get device from policy parameters
                    device = next(policy.parameters()).device
                    # Obs is 701-dim numpy array from Cyberwheel red agent
                    obs_array = np.array(obs, dtype=np.float32)
                    # Ensure it's the right shape (701,)
                    if obs_array.ndim == 0:
                        # Scalar - something wrong
                        print(f"  Warning: Obs is scalar, expected 701-dim vector. Shape: {obs_array.shape}")
                        sampled = cw_env.action_space.sample()
                        action = sampled.get("red", 0) if isinstance(sampled, dict) else 0
                        action_dict = {"red": action, "blue": None}
                    elif obs_array.shape[0] != 701:
                        # Wrong dimension
                        print(f"  Warning: Obs shape is {obs_array.shape}, expected (701,). Using random action.")
                        sampled = cw_env.action_space.sample()
                        action = sampled.get("red", 0) if isinstance(sampled, dict) else 0
                        action_dict = {"red": action, "blue": None}
                    else:
                        # Correct shape - use policy
                        obs_tensor = torch.from_numpy(obs_array).float().unsqueeze(0).to(device)
                        # Get action mask if available
                        action_mask = None
                        if hasattr(cw_env, 'action_mask') and hasattr(cw_env.action_mask, 'get'):
                            mask = cw_env.action_mask
                            if isinstance(mask, dict) and "red" in mask:
                                action_mask = torch.tensor(mask["red"], dtype=torch.bool, device=device).unsqueeze(0)
                        
                        with torch.no_grad():
                            action_tensor, _, _, _ = policy.get_action_and_value(obs_tensor, action_mask=action_mask)
                            action = int(action_tensor.cpu().numpy()[0])
                        
                        # Clip action to valid range if needed
                        if "red" in cw_env.action_space:
                            max_action = cw_env.action_space["red"].n - 1
                            if action > max_action:
                                action = max_action
                            if action < 0:
                                action = 0
                        
                        # Cyberwheel expects dict action - only provide red action, blue is inactive
                        action_dict = {"red": action, "blue": None}
                else:
                    # Random action
                    if use_direct_env:
                        action_dict = cw_env.action_space.sample()
                        action = action_dict.get("red", 0)  # Extract red action for storage
                    else:
                        action = env.action_space.sample()
                        action_dict = {"red": action, "blue": None}
                
                if use_direct_env:
                    next_obs_dict, reward, done, truncated, info = cw_env.step(action_dict)
                    # Extract red agent observation
                    next_obs = next_obs_dict.get("red") if isinstance(next_obs_dict, dict) else next_obs_dict
                else:
                    next_obs, reward, done, truncated, info = env.step(action)
                
                # Store transitions (obs is 701-dim numpy array from red agent)
                transitions.append({
                    'obs': obs if isinstance(obs, np.ndarray) else np.array(obs),
                    'action': int(action),
                    'next_obs': next_obs if isinstance(next_obs, np.ndarray) else np.array(next_obs),
                    'reward': float(reward),
                    'done': bool(done or truncated)
                })
                
                obs = next_obs
                step += 1
                
                obs = next_obs
                step += 1
                
                obs = next_obs
                step += 1
            
            if (episode + 1) % 20 == 0:
                print(f"  Collected {episode + 1}/{num_episodes} episodes ({len(transitions)} transitions)")
        except Exception as e:
            import traceback
            print(f"  Warning: Episode {episode + 1} failed: {e}")
            if episode == 0:  # Show full traceback for first error
                traceback.print_exc()
            continue
    
    print(f"\n Collected {len(transitions)} Cyberwheel transitions")
    
    if output_file and len(transitions) > 0:
        import pickle
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'wb') as f:
            pickle.dump(transitions, f)
        print(f"Saved to {output_file}")
    
    return transitions


def main():
    parser = argparse.ArgumentParser(description="Collect more training data for encoder")
    parser.add_argument("--cbs_episodes", type=int, default=100,
                       help="Number of CBS episodes to collect")
    parser.add_argument("--cw_episodes", type=int, default=100,
                       help="Number of Cyberwheel episodes to collect")
    parser.add_argument("--max_steps", type=int, default=200,
                       help="Max steps per episode")
    parser.add_argument("--cbs_output", type=str,
                       default="artifacts/training_data/cbs_transitions.pkl",
                       help="Output file for CBS transitions")
    parser.add_argument("--cw_output", type=str,
                       default="artifacts/training_data/cw_transitions.pkl",
                       help="Output file for Cyberwheel transitions")
    parser.add_argument("--skip_cw", action="store_true",
                       help="Skip Cyberwheel collection")
    parser.add_argument("--cw_checkpoint", type=str, default=None,
                       help="Path to Cyberwheel checkpoint to use for data collection")
    
    args = parser.parse_args()
    
    # Collect CBS data
    cbs_transitions = collect_cbs_data(
        num_episodes=args.cbs_episodes,
        max_steps=args.max_steps,
        output_file=args.cbs_output
    )
    
    # Collect Cyberwheel data
    cw_transitions = []
    if not args.skip_cw:
        cw_transitions = collect_cw_data(
            num_episodes=args.cw_episodes,
            max_steps=args.max_steps,
            output_file=args.cw_output,
            cw_checkpoint_path=args.cw_checkpoint
        )
    
    print(f"\nCollected:")
    print(f"  CBS transitions: {len(cbs_transitions)}")
    print(f"  Cyberwheel transitions: {len(cw_transitions)}")
    print(f"  Total: {len(cbs_transitions) + len(cw_transitions)}")
    
    if len(cbs_transitions) > 0:
        print(f"\nCBS data saved to: {args.cbs_output}")
    if len(cw_transitions) > 0:
        print(f"Cyberwheel data saved to: {args.cw_output}")
    
    print("\nTo retrain encoder with this data:")
    print(f"  python train_full_observation_transfer.py --cbs_episodes {args.cbs_episodes} --cw_episodes {args.cw_episodes}")


if __name__ == "__main__":
    main()

