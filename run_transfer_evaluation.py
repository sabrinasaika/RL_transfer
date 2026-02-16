#!/usr/bin/env python3
"""
Simplified script to evaluate transfer learning on CBS.
Uses existing Cyberwheel checkpoints and applies transfer encoder.
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
from stable_baselines3.ppo import PPO

from adapters.unified_env import UnifiedSecEnv
from adapters.observation_translator import ObservationTranslator
from adapters.transfer_encoder import ObservationEncoder, DynamicsModel, save_transfer_models
from adapters.transfer_training import ReplayBuffer, train_dynamics_model
from config.env_builders import make_cbs_env


def create_and_train_encoder_on_cbs(num_episodes: int = 50, save_path: str = "artifacts/transfer_models/cbs_encoder.pt"):
    """
    Create and train encoder + dynamics model on CBS (as a proxy for source task).
    In practice, you'd train on Cyberwheel, but this works for testing.
    """
    print("=" * 60)
    print("TRAINING ENCODER AND DYNAMICS MODEL")
    print("=" * 60)
    
    # Create environment
    env = UnifiedSecEnv("cbs", cbs_factory=make_cbs_env)
    print(f"Environment: {env.observation_space}, {env.action_space}")
    
    # Initialize encoder and dynamics model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = ObservationEncoder(input_dim=8, feature_size=64).to(device)
    dynamics_model = DynamicsModel(feature_size=64, num_actions=env.action_space.n).to(device)
    
    # Replay buffer
    replay_buffer = ReplayBuffer(capacity=5000)
    
    # Collect transitions by running random policy
    print(f"\nCollecting {num_episodes} episodes of transitions...")
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        step = 0
        
        while not (done or truncated) and step < 200:
            # Get raw observation (before encoding)
            if isinstance(obs, dict):
                obs_raw = obs.get("obs", obs)
            else:
                obs_raw = obs
            
            # Random action
            action = env.action_space.sample()
            next_obs, reward, done, truncated, info = env.step(action)
            
            if isinstance(next_obs, dict):
                next_obs_raw = next_obs.get("obs", next_obs)
            else:
                next_obs_raw = next_obs
            
            # Store transition (using raw 8-dim observations)
            replay_buffer.push(obs_raw, action, next_obs_raw, float(reward), done or truncated)
            
            obs = next_obs
            step += 1
        
        if (episode + 1) % 10 == 0:
            print(f"  Collected {episode + 1}/{num_episodes} episodes")
    
    print(f"\nCollected {len(replay_buffer)} transitions")
    
    # Train dynamics model
    print("\nTraining dynamics model...")
    losses = train_dynamics_model(
        encoder, dynamics_model, replay_buffer,
        batch_size=64, num_epochs=20, device=device
    )
    
    if losses:
        print(f"  Final dynamics model loss: {losses[-1]:.4f}")
        print(f"  Initial loss: {losses[0]:.4f}, Final loss: {losses[-1]:.4f}")
    
    # Save models
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    save_transfer_models(
        encoder, dynamics_model, save_path,
        input_dim=8, feature_size=64, num_actions=env.action_space.n
    )
    print(f"\n✓ Saved encoder and dynamics model to {save_path}")
    
    return save_path


def evaluate_with_without_transfer(encoder_path: str, episodes: int = 10, max_steps: int = 5000):
    """
    Evaluate on CBS with and without transfer encoder.
    """
    print("\n" + "=" * 60)
    print("EVALUATING WITH/WITHOUT TRANSFER")
    print("=" * 60)
    
    # Create environment
    env = UnifiedSecEnv("cbs", cbs_factory=make_cbs_env)
    
    # Train a quick baseline model
    print("\nTraining quick baseline model on CBS...")
    # Use MultiInputPolicy for dict observation space
    policy = "MultiInputPolicy" if isinstance(env.observation_space, dict) else "MlpPolicy"
    model = PPO(policy, env, verbose=1)
    model.learn(total_timesteps=5000)
    
    # Evaluate WITHOUT transfer
    print("\n" + "-" * 60)
    print("Evaluating WITHOUT transfer...")
    print("-" * 60)
    results_no_transfer = evaluate_model(env, model, episodes, max_steps, use_transfer=False)
    print(f"  Average Return: {results_no_transfer['avg_return']:.2f} ± {results_no_transfer['std_return']:.2f}")
    print(f"  Average Steps: {results_no_transfer['avg_steps']:.1f}")
    print(f"  Success Rate: {results_no_transfer['success_rate']:.2%}")
    
    # Evaluate WITH transfer
    print("\n" + "-" * 60)
    print("Evaluating WITH transfer...")
    print("-" * 60)
    results_with_transfer = evaluate_model(env, model, episodes, max_steps, 
                                          use_transfer=True, encoder_path=encoder_path)
    print(f"  Average Return: {results_with_transfer['avg_return']:.2f} ± {results_with_transfer['std_return']:.2f}")
    print(f"  Average Steps: {results_with_transfer['avg_steps']:.1f}")
    print(f"  Success Rate: {results_with_transfer['success_rate']:.2%}")
    
    # Calculate improvement
    improvement = {
        "return": results_with_transfer['avg_return'] - results_no_transfer['avg_return'],
        "return_pct": ((results_with_transfer['avg_return'] - results_no_transfer['avg_return']) / 
                      abs(results_no_transfer['avg_return']) * 100) if results_no_transfer['avg_return'] != 0 else 0,
        "steps": results_with_transfer['avg_steps'] - results_no_transfer['avg_steps'],
        "success_rate": results_with_transfer['success_rate'] - results_no_transfer['success_rate'],
    }
    
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    print(f"Return Improvement: {improvement['return']:+.2f} ({improvement['return_pct']:+.1f}%)")
    print(f"Steps Improvement: {improvement['steps']:+.1f}")
    print(f"Success Rate Improvement: {improvement['success_rate']*100:+.1f}%")
    
    return {
        "no_transfer": results_no_transfer,
        "with_transfer": results_with_transfer,
        "improvement": improvement
    }


def evaluate_model(env, model, episodes: int, max_steps: int, 
                  use_transfer: bool = False, encoder_path: str = None):
    """Evaluate model on environment"""
    returns = []
    steps_to_goal = []
    successes = 0
    
    # Modify observation translator if using transfer
    if use_transfer and encoder_path and os.path.exists(encoder_path):
        env.obs_t = ObservationTranslator(use_transfer=True, encoder_path=encoder_path)
        print(f"  Using transfer encoder from {encoder_path}")
    else:
        env.obs_t = ObservationTranslator(use_transfer=False)
    
    for episode in range(episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        ep_return = 0.0
        step = 0
        
        while not (done or truncated) and step < max_steps:
            # Handle dict observations
            if isinstance(obs, dict):
                obs_input = obs.get("obs", obs)
            else:
                obs_input = obs
            
            action, _ = model.predict(obs_input, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            ep_return += float(reward)
            step += 1
        
        returns.append(ep_return)
        steps_to_goal.append(step)
        
        # Success is determined by done=True (goal reached), not by return value
        # because winning_reward can be 0.0 when CBS_ZERO_WIN_LOSE_REWARD=1
        if done and not truncated:
            successes += 1
        
        if (episode + 1) % 5 == 0:
            print(f"  Completed {episode + 1}/{episodes} episodes")
    
    return {
        "avg_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "avg_steps": float(np.mean(steps_to_goal)),
        "success_rate": float(successes) / episodes,
        "returns": returns,
        "steps_to_goal": steps_to_goal,
    }


def save_results(results, output_path: str = "artifacts/plots/transfer_evaluation_results.json"):
    """Save results to JSON file"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Saved results to {output_path}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Evaluate transfer learning on CBS")
    parser.add_argument("--train_encoder", action="store_true",
                       help="Train encoder and dynamics model first")
    parser.add_argument("--encoder_path", type=str, default="artifacts/transfer_models/cbs_encoder.pt",
                       help="Path to encoder checkpoint")
    parser.add_argument("--episodes", type=int, default=10,
                       help="Number of evaluation episodes")
    parser.add_argument("--max_steps", type=int, default=5000,
                       help="Maximum steps per episode")
    parser.add_argument("--collect_episodes", type=int, default=50,
                       help="Number of episodes to collect for encoder training")
    parser.add_argument("--output_dir", type=str, default="artifacts/plots",
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("TRANSFER LEARNING EVALUATION")
    print("=" * 60 + "\n")
    
    # Train encoder if requested
    if args.train_encoder:
        encoder_path = create_and_train_encoder_on_cbs(
            num_episodes=args.collect_episodes,
            save_path=args.encoder_path
        )
    else:
        encoder_path = args.encoder_path
        if not os.path.exists(encoder_path):
            print(f"Warning: Encoder path not found: {encoder_path}")
            print("Run with --train_encoder to create one first")
            print("Or provide path to existing encoder with --encoder_path")
            return
    
    # Evaluate with and without transfer
    results = evaluate_with_without_transfer(
        encoder_path=encoder_path,
        episodes=args.episodes,
        max_steps=args.max_steps
    )
    
    # Save results
    output_path = os.path.join(args.output_dir, "transfer_evaluation_results.json")
    save_results(results, output_path)
    
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE!")
    print("=" * 60)
    print(f"\nResults saved to: {output_path}")
    print("\nTo visualize results:")
    print(f"  python eval_transfer_results.py --model_path <model> --encoder_path {encoder_path} --plot")


if __name__ == "__main__":
    main()

