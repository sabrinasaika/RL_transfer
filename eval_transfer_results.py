#!/usr/bin/env python3
"""
Evaluate and visualize observation transfer results.
Compares performance with and without transfer learning.
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Optional

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from adapters.unified_env import UnifiedSecEnv
from adapters.observation_translator import ObservationTranslator
from config.env_builders import make_cbs_env
from stable_baselines3.ppo import PPO


def evaluate_model(env, model, episodes: int = 10, max_steps: int = 5000, 
                  use_transfer: bool = False, encoder_path: Optional[str] = None):
    """
    Evaluate a model on the environment.
    
    Args:
        env: Environment to evaluate on
        model: Model to evaluate (PPO or similar)
        episodes: Number of episodes to run
        max_steps: Maximum steps per episode
        use_transfer: Whether to use transfer encoder
        encoder_path: Path to encoder checkpoint
    
    Returns:
        Dictionary with evaluation metrics
    """
    returns = []
    steps_to_goal = []
    successes = 0
    
    # Modify observation translator if using transfer
    if use_transfer and encoder_path:
        env.obs_t = ObservationTranslator(use_transfer=True, encoder_path=encoder_path)
        print(f"  Using transfer encoder from {encoder_path}")
    
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
    
    return {
        "avg_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "avg_steps": float(np.mean(steps_to_goal)),
        "success_rate": float(successes) / episodes,
        "returns": returns,
        "steps_to_goal": steps_to_goal,
    }


def compare_with_without_transfer(model_path: str, encoder_path: Optional[str] = None,
                                  episodes: int = 10, max_steps: int = 5000):
    """
    Compare model performance with and without transfer learning.
    
    Args:
        model_path: Path to trained model
        encoder_path: Path to transfer encoder checkpoint
        episodes: Number of evaluation episodes
        max_steps: Maximum steps per episode
    
    Returns:
        Dictionary with comparison results
    """
    print("=" * 60)
    print("COMPARING WITH/WITHOUT TRANSFER LEARNING")
    print("=" * 60)
    
    # Load model
    print(f"\nLoading model from {model_path}")
    model = PPO.load(model_path)
    
    # Create environment
    env = UnifiedSecEnv("cbs", cbs_factory=make_cbs_env)
    
    # Evaluate WITHOUT transfer
    print("\n" + "-" * 60)
    print("Evaluating WITHOUT transfer learning...")
    print("-" * 60)
    results_no_transfer = evaluate_model(
        env, model, episodes=episodes, max_steps=max_steps,
        use_transfer=False
    )
    
    print(f"  Average Return: {results_no_transfer['avg_return']:.2f} ± {results_no_transfer['std_return']:.2f}")
    print(f"  Average Steps: {results_no_transfer['avg_steps']:.1f}")
    print(f"  Success Rate: {results_no_transfer['success_rate']:.2%}")
    
    # Evaluate WITH transfer
    if encoder_path and os.path.exists(encoder_path):
        print("\n" + "-" * 60)
        print("Evaluating WITH transfer learning...")
        print("-" * 60)
        results_with_transfer = evaluate_model(
            env, model, episodes=episodes, max_steps=max_steps,
            use_transfer=True, encoder_path=encoder_path
        )
        
        print(f"  Average Return: {results_with_transfer['avg_return']:.2f} ± {results_with_transfer['std_return']:.2f}")
        print(f"  Average Steps: {results_with_transfer['avg_steps']:.1f}")
        print(f"  Success Rate: {results_with_transfer['success_rate']:.2%}")
        
        # Calculate improvement
        return_improvement = results_with_transfer['avg_return'] - results_no_transfer['avg_return']
        return_improvement_pct = (return_improvement / abs(results_no_transfer['avg_return']) * 100) if results_no_transfer['avg_return'] != 0 else 0
        
        print("\n" + "=" * 60)
        print("COMPARISON RESULTS")
        print("=" * 60)
        print(f"Return Improvement: {return_improvement:+.2f} ({return_improvement_pct:+.1f}%)")
        print(f"Steps Improvement: {results_with_transfer['avg_steps'] - results_no_transfer['avg_steps']:+.1f}")
        print(f"Success Rate Improvement: {(results_with_transfer['success_rate'] - results_no_transfer['success_rate']) * 100:+.1f}%")
        
        return {
            "no_transfer": results_no_transfer,
            "with_transfer": results_with_transfer,
            "improvement": {
                "return": return_improvement,
                "return_pct": return_improvement_pct,
                "steps": results_with_transfer['avg_steps'] - results_no_transfer['avg_steps'],
                "success_rate": results_with_transfer['success_rate'] - results_no_transfer['success_rate'],
            }
        }
    else:
        print(f"\nWarning: Encoder path not found: {encoder_path}")
        print("Skipping transfer evaluation")
        return {"no_transfer": results_no_transfer, "with_transfer": None}


def plot_comparison(results: Dict, output_path: str = "artifacts/plots/transfer_comparison.png"):
    """
    Plot comparison of results with and without transfer.
    
    Args:
        results: Results dictionary from compare_with_without_transfer
        output_path: Path to save plot
    """
    if results.get("with_transfer") is None:
        print("No transfer results to plot")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    no_transfer = results["no_transfer"]
    with_transfer = results["with_transfer"]
    
    # Plot 1: Episode Returns
    axes[0].boxplot([no_transfer["returns"], with_transfer["returns"]], 
                    labels=["No Transfer", "With Transfer"])
    axes[0].set_ylabel("Episode Return", fontsize=12)
    axes[0].set_title("Episode Returns Comparison", fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Steps to Goal
    axes[1].boxplot([no_transfer["steps_to_goal"], with_transfer["steps_to_goal"]],
                    labels=["No Transfer", "With Transfer"])
    axes[1].set_ylabel("Steps to Goal", fontsize=12)
    axes[1].set_title("Steps to Goal Comparison", fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Metrics Summary
    metrics = ["Avg Return", "Success Rate"]
    no_transfer_vals = [no_transfer["avg_return"], no_transfer["success_rate"] * 100]
    with_transfer_vals = [with_transfer["avg_return"], with_transfer["success_rate"] * 100]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    axes[2].bar(x - width/2, no_transfer_vals, width, label="No Transfer", alpha=0.8)
    axes[2].bar(x + width/2, with_transfer_vals, width, label="With Transfer", alpha=0.8)
    axes[2].set_ylabel("Value", fontsize=12)
    axes[2].set_title("Metrics Summary", fontsize=14, fontweight='bold')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(metrics)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3, axis='y')
    
    # Add improvement text
    improvement = results["improvement"]
    textstr = f"Return: {improvement['return']:+.2f} ({improvement['return_pct']:+.1f}%)\n"
    textstr += f"Success Rate: {improvement['success_rate']*100:+.1f}%"
    axes[2].text(0.02, 0.98, textstr, transform=axes[2].transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved comparison plot to {output_path}")
    plt.close()


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate transfer learning results")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to trained model (.zip file)")
    parser.add_argument("--encoder_path", type=str, default=None,
                       help="Path to transfer encoder checkpoint (.pt file)")
    parser.add_argument("--episodes", type=int, default=10,
                       help="Number of evaluation episodes")
    parser.add_argument("--max_steps", type=int, default=5000,
                       help="Maximum steps per episode")
    parser.add_argument("--output_dir", type=str, default="artifacts/plots",
                       help="Output directory for plots and results")
    parser.add_argument("--plot", action="store_true",
                       help="Generate comparison plots")
    
    args = parser.parse_args()
    
    # Compare with/without transfer
    results = compare_with_without_transfer(
        args.model_path,
        args.encoder_path,
        episodes=args.episodes,
        max_steps=args.max_steps
    )
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_path = output_dir / "transfer_comparison_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Saved results to {results_path}")
    
    # Generate plots
    if args.plot and results.get("with_transfer") is not None:
        plot_path = output_dir / "transfer_comparison.png"
        plot_comparison(results, str(plot_path))
    
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()

