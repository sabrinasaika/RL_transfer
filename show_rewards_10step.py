#!/usr/bin/env python3
"""
Show rewards at 10-step intervals during evaluation or training
"""

import os
import sys
from pathlib import Path
import numpy as np
from stable_baselines3 import PPO
from adapters.unified_env import UnifiedSecEnv
from adapters.dapn_env_wrapper import DAPNEnvWrapper
from config.env_builders import make_cw_env, make_cbs_env
import gymnasium as gym

# Add paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "cyberwheel"))

def show_rewards_10step(env_type="cw", num_episodes=1, model_path=None, interval=10):
    """
    Show rewards at specified step intervals
    
    Args:
        env_type: "cw" for Cyberwheel or "cbs" for CyberBattleSim
        num_episodes: Number of episodes to run
        model_path: Path to trained model (None for random actions)
        interval: Step interval for printing rewards (default: 10)
    """
    print("=" * 70)
    print(f"SHOWING REWARDS AT {interval}-STEP INTERVALS")
    print("=" * 70)
    
    # Create environment
    if env_type == "cw":
        print("\nCreating Cyberwheel environment...")
        os.environ["CW_ENV_YAML"] = "credential_preference_scenario.yaml"
        base_env = UnifiedSecEnv("cw", cw_factory=make_cw_env)
        
        # Wrap with DAPN if model exists and uses DAPN
        if model_path and os.path.exists(model_path):
            dapn_encoder_path = "artifacts/transfer_models/dapn_encoder.pt"
            if os.path.exists(dapn_encoder_path):
                print("Wrapping with DAPN encoder...")
                env = DAPNEnvWrapper(
                    base_env,
                    encoder_path=dapn_encoder_path,
                    use_dapn=True
                )
            else:
                env = base_env
        else:
            env = base_env
    else:  # cbs
        print("\nCreating CyberBattleSim environment...")
        os.environ["CBS_ENV"] = "CyberBattleFlat-v0"
        os.environ["CBS_FLAT_NODES"] = "20"
        os.environ["CBS_CRED_REUSE_PROB"] = "0.6"
        os.environ["CBS_EXPLOIT_PROB"] = "0.3"
        base_env = UnifiedSecEnv("cbs", cbs_factory=make_cbs_env)
        
        # Wrap with DAPN if model exists and uses DAPN
        if model_path and os.path.exists(model_path):
            dapn_encoder_path = "artifacts/transfer_models/dapn_encoder.pt"
            if os.path.exists(dapn_encoder_path):
                print("Wrapping with DAPN encoder...")
                env = DAPNEnvWrapper(
                    base_env,
                    encoder_path=dapn_encoder_path,
                    use_dapn=True
                )
            else:
                env = base_env
        else:
            env = base_env
    
    print(f"✓ Environment created: {env_type.upper()}")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")
    
    # Load model if provided
    model = None
    if model_path and os.path.exists(model_path):
        print(f"\nLoading model: {model_path}")
        model = PPO.load(model_path)
        print("✓ Model loaded")
    else:
        print("\n⚠ No model provided - using random actions")
    
    # Run episodes
    print(f"\n{'='*70}")
    print(f"Running {num_episodes} episode(s) with rewards shown every {interval} steps")
    print(f"{'='*70}\n")
    
    for episode in range(num_episodes):
        print(f"\n{'─'*70}")
        print(f"EPISODE {episode + 1}")
        print(f"{'─'*70}")
        
        obs, info = env.reset()
        total_reward = 0
        step_rewards = []
        steps = 0
        done = False
        truncated = False
        
        # Print header
        print(f"\n{'Step':<6} {'Reward':<10} {'Cumulative':<12} {'Action':<15} {'Status'}")
        print(f"{'-'*70}")
        
        while not (done or truncated) and steps < 100:
            # Get action
            if model is not None:
                if isinstance(obs, dict):
                    obs_for_pred = obs
                else:
                    obs_for_pred = obs
                action, _ = model.predict(obs_for_pred, deterministic=False)
            else:
                action = env.action_space.sample()
            
            # Step environment
            obs_next, reward, done, truncated, info = env.step(action)
            
            total_reward += reward
            step_rewards.append(reward)
            steps += 1
            
            # Print at intervals or on special events
            should_print = (
                steps <= 5 or  # First 5 steps
                steps % interval == 0 or  # Every N steps
                reward != 0 or  # Non-zero reward
                done or truncated  # Episode end
            )
            
            if should_print:
                action_str = str(action) if not isinstance(action, np.ndarray) else f"{action.item()}"
                status = "DONE" if done else ("TRUNC" if truncated else "CONT")
                reward_str = f"{reward:+.2f}" if reward != 0 else f"{reward:.2f}"
                print(f"{steps:<6} {reward_str:<10} {total_reward:<12.2f} {action_str:<15} {status}")
            
            obs = obs_next
        
        # Episode summary
        print(f"{'-'*70}")
        print(f"\nEpisode {episode + 1} Summary:")
        print(f"  Total steps: {steps}")
        print(f"  Total reward: {total_reward:.2f}")
        print(f"  Average reward per step: {total_reward/steps:.4f}")
        print(f"  Reward range: [{min(step_rewards):.2f}, {max(step_rewards):.2f}]")
        print(f"  Non-zero rewards: {sum(1 for r in step_rewards if r != 0)}/{len(step_rewards)}")
        
        # Show reward distribution
        unique_rewards = sorted(set(step_rewards))
        if len(unique_rewards) <= 10:
            print(f"  Reward values: {unique_rewards}")
        else:
            print(f"  Reward values (first 10): {unique_rewards[:10]}")
            print(f"  ... and {len(unique_rewards)-10} more unique values")
    
    print(f"\n{'='*70}")
    print("COMPLETE")
    print(f"{'='*70}")


def main():
    """Main function with command-line argument parsing"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Show rewards at 10-step intervals")
    parser.add_argument(
        "--env", 
        type=str, 
        choices=["cw", "cbs"], 
        default="cw",
        help="Environment type: 'cw' for Cyberwheel, 'cbs' for CyberBattleSim"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Number of episodes to run (default: 1)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to trained model (default: None, uses random actions)"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=10,
        help="Step interval for printing rewards (default: 10)"
    )
    
    args = parser.parse_args()
    
    # Default model paths
    if args.model is None:
        if args.env == "cw":
            args.model = "artifacts/policies/cw_ppo_dapn.zip"
        else:
            args.model = "artifacts/policies/cbs_ppo_minimal.zip"
    
    show_rewards_10step(
        env_type=args.env,
        num_episodes=args.episodes,
        model_path=args.model if os.path.exists(args.model) else None,
        interval=args.interval
    )


if __name__ == "__main__":
    main()
