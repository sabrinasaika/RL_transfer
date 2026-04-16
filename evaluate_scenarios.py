#!/usr/bin/env python3
"""
Evaluate trained models on both credential preference scenarios
Shows how well the agent learned to prefer credentials over exploits
"""

import os
import sys
from pathlib import Path
from stable_baselines3 import PPO
from adapters.unified_env import UnifiedSecEnv
from config.env_builders import make_cw_env, make_cbs_env

# Add paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "cyberwheel"))

def evaluate_cyberwheel(model_path="artifacts/policies/cw_ppo_very_short.zip", num_episodes=5):
    """Evaluate Cyberwheel scenario"""
    print("=" * 60)
    print("Evaluating Cyberwheel Credential Preference Scenario")
    print("=" * 60)
    
    if not os.path.exists(model_path):
        print(f"✗ Model not found: {model_path}")
        print("  Train a model first using: python train/train_cw_ppo_very_short.py")
        return None
    
    # Load model
    print(f"Loading model: {model_path}")
    model = PPO.load(model_path)
    print("✓ Model loaded")
    
    # Create environment
    os.environ["CW_ENV_YAML"] = "credential_preference_scenario.yaml"
    env = UnifiedSecEnv("cw", cw_factory=make_cw_env)
    print("✓ Environment created")
    
    # Run episodes
    print(f"\nRunning {num_episodes} evaluation episodes...")
    print("-" * 60)
    
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        total_reward = 0
        steps = 0
        done = False
        truncated = False
        
        while not (done or truncated) and steps < 100:  # Max 100 steps per episode
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
        
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        print(f"  Episode {episode+1}: Reward={total_reward:.2f}, Steps={steps}")
    
    # Statistics
    avg_reward = sum(episode_rewards) / len(episode_rewards)
    avg_length = sum(episode_lengths) / len(episode_lengths)
    max_reward = max(episode_rewards)
    
    print("-" * 60)
    print("Results:")
    print(f"  Average reward: {avg_reward:.2f}")
    print(f"  Max reward: {max_reward:.2f}")
    print(f"  Average episode length: {avg_length:.1f} steps")
    print("=" * 60)
    
    return {
        "avg_reward": avg_reward,
        "max_reward": max_reward,
        "avg_length": avg_length,
        "episode_rewards": episode_rewards
    }


def evaluate_cbs(model_path="artifacts/policies/cbs_ppo_very_short.zip", num_episodes=5):
    """Evaluate CyberBattleSim scenario"""
    print("\n" + "=" * 60)
    print("Evaluating CyberBattleSim Flat Network Scenario")
    print("=" * 60)
    
    if not os.path.exists(model_path):
        print(f"✗ Model not found: {model_path}")
        print("  Train a model first using: python train/train_cbs_ppo_very_short.py")
        return None
    
    # Load model
    print(f"Loading model: {model_path}")
    model = PPO.load(model_path)
    print("✓ Model loaded")
    
    # Create environment
    os.environ["CBS_ENV"] = "CyberBattleFlat-v0"
    os.environ["CBS_FLAT_NODES"] = "20"
    os.environ["CBS_CRED_REUSE_PROB"] = "0.6"
    os.environ["CBS_EXPLOIT_PROB"] = "0.3"
    env = UnifiedSecEnv("cbs", cbs_factory=make_cbs_env)
    print("✓ Environment created")
    
    # Run episodes
    print(f"\nRunning {num_episodes} evaluation episodes...")
    print("-" * 60)
    
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        total_reward = 0
        steps = 0
        done = False
        truncated = False
        
        # Handle dict observations
        while not (done or truncated) and steps < 100:  # Max 100 steps per episode
            obs_for_pred = obs['obs'] if isinstance(obs, dict) else obs
            action, _ = model.predict(obs_for_pred, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
        
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        print(f"  Episode {episode+1}: Reward={total_reward:.2f}, Steps={steps}")
    
    # Statistics
    avg_reward = sum(episode_rewards) / len(episode_rewards)
    avg_length = sum(episode_lengths) / len(episode_lengths)
    max_reward = max(episode_rewards)
    
    print("-" * 60)
    print("Results:")
    print(f"  Average reward: {avg_reward:.2f}")
    print(f"  Max reward: {max_reward:.2f}")
    print(f"  Average episode length: {avg_length:.1f} steps")
    print("=" * 60)
    
    return {
        "avg_reward": avg_reward,
        "max_reward": max_reward,
        "avg_length": avg_length,
        "episode_rewards": episode_rewards
    }


def main():
    """Run evaluation on both scenarios"""
    print("\n" + "=" * 60)
    print("SCENARIO EVALUATION")
    print("=" * 60)
    print("\nThis will test trained models on both scenarios")
    print("to see how well they learned credential preference.\n")
    
    # Check if models exist
    cw_model = "artifacts/policies/cw_ppo_very_short.zip"
    cbs_model = "artifacts/policies/cbs_ppo_very_short.zip"
    
    # Also check for other model names
    if not os.path.exists(cw_model):
        cw_model = "artifacts/policies/cw_ppo_minimal.zip"
    if not os.path.exists(cbs_model):
        cbs_model = "artifacts/policies/cbs_ppo_minimal.zip"
    
    results = {}
    
    # Evaluate Cyberwheel
    if os.path.exists(cw_model):
        results['cyberwheel'] = evaluate_cyberwheel(cw_model, num_episodes=5)
    else:
        print("⚠ Cyberwheel model not found. Train it first.")
    
    # Evaluate CyberBattleSim
    if os.path.exists(cbs_model):
        results['cyberbattlesim'] = evaluate_cbs(cbs_model, num_episodes=5)
    else:
        print("⚠ CyberBattleSim model not found. Train it first.")
    
    # Summary
    if results:
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)
        if 'cyberwheel' in results:
            cw = results['cyberwheel']
            print(f"Cyberwheel:      Avg Reward = {cw['avg_reward']:.2f}, Max = {cw['max_reward']:.2f}")
        if 'cyberbattlesim' in results:
            cbs = results['cyberbattlesim']
            print(f"CyberBattleSim:  Avg Reward = {cbs['avg_reward']:.2f}, Max = {cbs['max_reward']:.2f}")
        print("=" * 60)
    else:
        print("\n⚠ No models found. Please train models first:")
        print("  python train/train_cw_ppo_very_short.py")
        print("  python train/train_cbs_ppo_very_short.py")


if __name__ == "__main__":
    main()

