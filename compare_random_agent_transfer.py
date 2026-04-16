#!/usr/bin/env python3
"""
Compare transfer learning approaches with random agent baseline:
1. Use random agent with encoder on CBS (baseline to test encoder benefit)
2. Train new policy on CBS using encoder trained on Cyberwheel data
"""

import os
import sys
import argparse

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch
from typing import Dict, List
from stable_baselines3 import PPO
import gymnasium as gym

from adapters.unified_env import UnifiedSecEnv
from adapters.observation_translator import ObservationTranslator
from adapters.full_obs_translator import FullObservationTranslator
from config.env_builders import make_cbs_env
from train_policy_full_obs import FullObsWrapper, make_full_obs_env


def evaluate_random_agent_with_encoder(
    encoder_path: str = None,
    episodes: int = 10,
    max_steps: int = 200,
) -> Dict:
    """
    Approach 1: Use random agent with encoder on CBS.
    This tests if the encoder itself provides any benefit over pure random.
    
    Args:
        encoder_path: Optional encoder for better features
        episodes: Number of evaluation episodes
        max_steps: Max steps per episode
    """
    
    print("=" * 60)
    print("APPROACH 1: RANDOM AGENT WITH ENCODER ON CBS")
    print("=" * 60)
  
    os.environ.setdefault("CBS_MULTI_INPUT", "0")
    os.environ.setdefault("CBS_REPAIR", "1")
    os.environ.setdefault("BOOST_STEPS", "0")
    os.environ.setdefault("EVAL_MASK_POLICY", "0")
    
    env = UnifiedSecEnv("cbs", cbs_factory=make_cbs_env)
    
    # Use full observation encoder if provided
    if encoder_path and os.path.exists(encoder_path):
        try:
            # Try loading as full observation encoder
            env.obs_t = FullObservationTranslator(use_transfer=True, encoder_path=encoder_path)
            print(f"  Using full observation encoder: {encoder_path}")
        except Exception as e1:
            try:
                # Fallback to 8-dim encoder
                env.obs_t = ObservationTranslator(use_transfer=True, encoder_path=encoder_path)
                print(f"  Using 8-dim transfer encoder: {encoder_path}")
            except Exception as e2:
                print(f"  Warning: Could not load encoder: {e1}, {e2}")
                print("  Using raw observations")
    else:
        print("  No encoder - using raw observations")
    
    cbs_action_space_size = len(env.act_t.unified_actions)
    print(f"  Using random agent (action space size: {cbs_action_space_size})")
    
    returns: List[float] = []
    steps_to_goal: List[int] = []
    successes: int = 0
    
    for episode_idx in range(episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        ep_ret = 0.0
        step = 0
        
        while not (done or truncated) and step < max_steps:
            # Sample random valid action
            try:
                mask_np = env._compute_unified_mask().astype(np.float32)
            except Exception:
                mask_np = np.ones((cbs_action_space_size,), dtype=np.float32)
            
            valid_indices = np.where(mask_np > 0.5)[0]
            if valid_indices.size == 0:
                action = int(np.random.randint(0, cbs_action_space_size))
            else:
                action = int(np.random.choice(valid_indices))
            
            next_obs, reward, done, truncated, info = env.step(action)
            ep_ret += float(reward)
            step += 1
            obs = next_obs
        
        returns.append(ep_ret)
        steps_to_goal.append(step)
        
        # Success is determined by done=True (goal reached), not by return value
        # because winning_reward can be 0.0 when CBS_ZERO_WIN_LOSE_REWARD=1
        success = (done and not truncated) or info.get("goal_reached", False) or info.get("success", False)
        if success:
            successes += 1
        
        # Print result for each episode
        print(f"    Episode {episode_idx + 1}/{episodes}: Return={ep_ret:.2f}, Steps={step}, Success={success}")
    
    avg_return = np.mean(returns)
    avg_steps = np.mean(steps_to_goal)
    success_rate = successes / episodes
    
    print(f"\n  Summary:")
    print(f"    Average Return: {avg_return:.3f}")
    print(f"    Average Steps to Goal: {avg_steps:.1f}")
    print(f"    Success Rate: {success_rate:.2%}")
    
    return {
        "avg_return": avg_return,
        "avg_steps_to_goal": avg_steps,
        "success_rate": success_rate,
        "episode_returns": returns,
        "episode_steps": steps_to_goal
    }


def evaluate_trained_policy(
    encoder_path: str,
    policy_path: str = None,
    episodes: int = 10,
    max_steps: int = 200,
    train_timesteps: int = 2000
) -> Dict:
    """
    Approach 2: Evaluate policy trained on CBS using encoder.
    
    Args:
        encoder_path: Path to encoder checkpoint
        policy_path: Path to trained policy (if None, will train one)
        episodes: Number of evaluation episodes
        max_steps: Max steps per episode
        train_timesteps: Timesteps for training if policy not found
    """
    print("\n" + "=" * 60)
    print("APPROACH 2: TRAINED POLICY ON CBS (WITH ENCODER)")
    print("=" * 60)
    
    # Create environment with encoder
    env_fn = make_full_obs_env(encoder_path)
    env = env_fn()
    
    # Check observation space - should be Dict with 64-dim obs
    obs_space = env.observation_space
    print(f"  Observation space: {obs_space}")
    
    # Verify the wrapper updated the space correctly
    if isinstance(obs_space, gym.spaces.Dict):
        obs_dim = obs_space['obs'].shape[0]
        print(f"  Encoded observation dimension: {obs_dim}")
        if obs_dim != 64:
            print(f"  Warning: Expected 64-dim, got {obs_dim}-dim")
    elif isinstance(obs_space, gym.spaces.Box):
        print(f"  Warning: Observation space is Box, not Dict. Encoder may not be working.")
        print(f"  Box shape: {obs_space.shape}")
    
    if policy_path and os.path.exists(policy_path):
        print(f"  Loading trained policy from {policy_path}...")
        model = PPO.load(policy_path, env=env)
    else:
        print(f"  No policy found. Training new policy ({train_timesteps} timesteps)...")
        print("  (Quick training for fast results...)")
        
        # Determine policy type based on observation space
        if isinstance(obs_space, gym.spaces.Dict):
            policy = "MultiInputPolicy"
        else:
            policy = "MlpPolicy"
        
        print(f"  Using policy: {policy}")
        # Use smaller batch sizes and fewer steps for faster training
        model = PPO(
            policy, 
            env, 
            verbose=1,
            n_steps=64,  # Smaller steps per update
            batch_size=16,  # Smaller batch
            n_epochs=4,  # Fewer epochs
            learning_rate=3e-4
        )
        model.learn(total_timesteps=train_timesteps)
        if policy_path:
            os.makedirs(os.path.dirname(policy_path), exist_ok=True)
            model.save(policy_path)
            print(f"  Saved policy to {policy_path}")
    
    print(f"\n  Evaluating on {episodes} episodes...")
    
    returns: List[float] = []
    episode_lengths: List[int] = []
    successes: int = 0
    
    for episode_idx in range(episodes):
        try:
            obs, info = env.reset()
            done = False
            truncated = False
            ep_ret = 0.0
            step = 0
            episode_success = False
            
            while not (done or truncated) and step < max_steps:
                try:
                    action, _ = model.predict(obs, deterministic=False)
                    next_obs, reward, done, truncated, info = env.step(action)
                    ep_ret += float(reward)
                    step += 1
                    obs = next_obs
                    
                    # Check for success (can happen during episode)
                    if info.get("goal_reached", False) or info.get("success", False):
                        episode_success = True
                        break
                except Exception as e:
                    print(f"    Error during step {step} of episode {episode_idx + 1}: {e}")
                    break
            
            returns.append(ep_ret)
            episode_lengths.append(step)
            
            # Success is determined by done=True (goal reached), not by return value
            # because winning_reward can be 0.0 when CBS_ZERO_WIN_LOSE_REWARD=1
            if done and not truncated:
                episode_success = True
            
            if episode_success:
                successes += 1
            
            print(f"    Episode {episode_idx + 1}/{episodes}: Return={ep_ret:.2f}, Steps={step}, Success={episode_success}")
        except Exception as e:
            print(f"    Error in episode {episode_idx + 1}: {e}")
            returns.append(0.0)
            episode_lengths.append(0)
    
    avg_return = np.mean(returns)
    avg_steps = np.mean(episode_lengths)
    success_rate = successes / len(returns) if returns else 0.0
    
    print(f"\n  Summary:")
    print(f"    Average Return: {avg_return:.3f}")
    print(f"    Average Steps: {avg_steps:.1f}")
    print(f"    Success Rate: {success_rate:.2%}")
    
    return {
        "avg_return": avg_return,
        "avg_steps_to_goal": avg_steps,
        "success_rate": success_rate,
        "episode_returns": returns,
        "episode_steps": episode_lengths
    }


def main():
    parser = argparse.ArgumentParser(description="Compare random agent vs trained policy with encoder")
    parser.add_argument("--encoder_path", type=str, required=True,
                       help="Path to trained encoder")
    parser.add_argument("--policy_path", type=str, default=None,
                       help="Path to trained CBS policy (will train if not provided)")
    parser.add_argument("--episodes", type=int, default=10,
                       help="Number of evaluation episodes")
    parser.add_argument("--max_steps", type=int, default=200,
                       help="Max steps per episode")
    parser.add_argument("--train_policy", action="store_true",
                       help="Train new policy if not found")
    parser.add_argument("--train_timesteps", type=int, default=1000,
                       help="Timesteps for policy training")
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("COMPARING RANDOM AGENT VS TRAINED POLICY (WITH ENCODER)")
    print("=" * 60)
    print(f"\nEncoder: {args.encoder_path}")
    print(f"Episodes: {args.episodes}")
    print(f"Max Steps: {args.max_steps}")
    
    # Approach 1: Random agent with encoder
    results1 = evaluate_random_agent_with_encoder(
        encoder_path=args.encoder_path,
        episodes=args.episodes,
        max_steps=args.max_steps
    )
    
    # Approach 2: Trained policy
    if args.policy_path or args.train_policy:
        results2 = evaluate_trained_policy(
            encoder_path=args.encoder_path,
            policy_path=args.policy_path,
            episodes=args.episodes,
            max_steps=args.max_steps,
            train_timesteps=args.train_timesteps
        )
    else:
        print("\n  Skipping Approach 2 (no policy path provided, use --train_policy to train)")
        results2 = None
    
    # Comparison
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    
    print("\nApproach 1: Random Agent with Encoder")
    print(f"  Average Return: {results1['avg_return']:.3f}")
    print(f"  Average Steps: {results1['avg_steps_to_goal']:.1f}")
    print(f"  Success Rate: {results1['success_rate']:.2%}")
    
    if results2:
        print("\nApproach 2: Trained Policy on CBS")
        print(f"  Average Return: {results2['avg_return']:.3f}")
        print(f"  Average Steps: {results2['avg_steps_to_goal']:.1f}")
        print(f"  Success Rate: {results2['success_rate']:.2%}")
        
        print("\nDifference (Approach 2 - Approach 1):")
        return_diff = results2['avg_return'] - results1['avg_return']
        steps_diff = results2['avg_steps_to_goal'] - results1['avg_steps_to_goal']
        success_diff = results2['success_rate'] - results1['success_rate']
        
        print(f"  Return: {return_diff:+.3f} ({'Better' if return_diff > 0 else 'Worse'})")
        print(f"  Steps: {steps_diff:+.1f} ({'Faster' if steps_diff < 0 else 'Slower'})")
        print(f"  Success Rate: {success_diff:+.2%}")
        
        # Calculate improvement percentage
        if results1['success_rate'] > 0:
            success_improvement_pct = (success_diff / results1['success_rate']) * 100
            print(f"  Success Rate Improvement: {success_improvement_pct:+.1f}%")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()

