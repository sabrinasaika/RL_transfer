#!/usr/bin/env python3
"""
Evaluate transfer learning using full observations from both environments.
"""

import os
import sys
import argparse
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import numpy as np
from adapters.unified_env import UnifiedSecEnv
from adapters.full_obs_translator import FullObservationTranslator
from config.env_builders import make_cbs_env


def evaluate_with_full_obs(encoder_path: str, episodes: int = 10, max_steps: int = 500):
    """Evaluate using full observation encoders"""
    print("=" * 60)
    print("EVALUATING WITH FULL OBSERVATION TRANSFER")
    print("=" * 60)
    
    # Create environment
    env = UnifiedSecEnv("cbs", cbs_factory=make_cbs_env)
    
    # Create translator with full observation encoder
    translator = FullObservationTranslator(
        use_transfer=True,
        encoder_path=encoder_path
    )
    
    # Replace env's translator
    env.obs_t = translator
    
    print(f"\nEnvironment: {env.observation_space}")
    print(f"Using full observation encoder from: {encoder_path}")
    
    # Evaluate
    returns = []
    steps_to_goal = []
    successes = 0
    
    print(f"\nRunning {episodes} episodes...")
    for episode in range(episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        ep_return = 0.0
        step = 0
        
        while not (done or truncated) and step < max_steps:
            # Get action mask if available
            if isinstance(obs, dict) and "mask" in obs:
                mask = obs["mask"]
                valid_actions = np.where(mask > 0.5)[0]
                if len(valid_actions) > 0:
                    action = np.random.choice(valid_actions)
                else:
                    action = env.action_space.sample()
            else:
                action = env.action_space.sample()
            
            obs, reward, done, truncated, info = env.step(action)
            ep_return += float(reward)
            step += 1
        
        returns.append(ep_return)
        steps_to_goal.append(step)
        
        if done and ep_return > 0:
            successes += 1
        
        print(f"  Episode {episode+1}/{episodes}: return={ep_return:.2f}, steps={step}")
    
    avg_return = np.mean(returns)
    std_return = np.std(returns)
    avg_steps = np.mean(steps_to_goal)
    success_rate = successes / episodes
    
  
    print("RESULTS")
   
    print(f"Average Return: {avg_return:.2f} ± {std_return:.2f}")
    print(f"Average Steps: {avg_steps:.1f}")
    print(f"Success Rate: {success_rate:.2%}")
    
    return {
        "avg_return": float(avg_return),
        "std_return": float(std_return),
        "avg_steps": float(avg_steps),
        "success_rate": float(success_rate),
        "returns": returns,
        "steps_to_goal": steps_to_goal
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate full observation transfer")
    parser.add_argument("--encoder_path", type=str, required=True,
                       help="Path to full observation encoder checkpoint")
    parser.add_argument("--episodes", type=int, default=10,
                       help="Number of evaluation episodes")
    parser.add_argument("--max_steps", type=int, default=500,
                       help="Maximum steps per episode")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.encoder_path):
        print(f"Error: Encoder path not found: {args.encoder_path}")
        print("Train an encoder first:")
        print("  python train_full_observation_transfer.py")
        return
    
    results = evaluate_with_full_obs(
        encoder_path=args.encoder_path,
        episodes=args.episodes,
        max_steps=args.max_steps
    )
    
    print("\n Evaluation complete!")


if __name__ == "__main__":
    main()

