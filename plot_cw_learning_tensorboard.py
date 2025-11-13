#!/usr/bin/env python3
"""
Plot learning curves from tensorboard logs for CW training.
"""

import matplotlib.pyplot as plt
import numpy as np
import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import argparse


def plot_learning_curve_from_tensorboard(log_dir):
    """
    Plot learning curves from tensorboard logs.
    
    Args:
        log_dir: Path to tensorboard log directory
    """
    print(f"Loading data from: {log_dir}")
    
    # Load the event accumulator
    ea = EventAccumulator(log_dir)
    ea.Reload()
    
    # Get available scalar tags
    scalar_tags = ea.Tags()['scalars']
    print(f"Available scalar tags: {scalar_tags}")
    
    # Extract data
    data = {}
    
    # Common metrics we want to plot
    metrics_to_plot = {
        'rollout/ep_len_mean': 'Episode Length',
        'rollout/ep_rew_mean': 'Episode Reward',
        'train/policy_gradient_loss': 'Policy Loss',
        'train/value_loss': 'Value Loss',
        'train/entropy_loss': 'Entropy Loss',
        'train/approx_kl': 'Approximate KL Divergence',
        'train/explained_variance': 'Explained Variance'
    }
    
    for tag in scalar_tags:
        if tag in metrics_to_plot:
            scalar_events = ea.Scalars(tag)
            steps = [event.step for event in scalar_events]
            values = [event.value for event in scalar_events]
            data[tag] = {'steps': steps, 'values': values}
            print(f"Loaded {len(values)} data points for {tag}")
    
    if not data:
        print("No relevant metrics found!")
        return
    
    # Create subplots
    n_metrics = len(data)
    n_cols = 2
    n_rows = (n_metrics + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    if n_rows == 1:
        axes = [axes] if n_cols == 1 else axes
    else:
        axes = axes.flatten()
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink']
    
    for i, (tag, info) in enumerate(data.items()):
        ax = axes[i]
        steps = info['steps']
        values = info['values']
        
        ax.plot(steps, values, color=colors[i % len(colors)], linewidth=2, marker='o', markersize=3)
        ax.set_xlabel('Timesteps', fontsize=12)
        ax.set_ylabel(metrics_to_plot[tag], fontsize=12)
        ax.set_title(f'{metrics_to_plot[tag]} Over Time', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add horizontal line at zero for reward plots
        if 'reward' in tag.lower():
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Hide unused subplots
    for i in range(len(data), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    # Save plot
    output_file = 'cw_learning_curves.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Learning curve saved to: {output_file}")
    
    # Print summary statistics
    for tag, info in data.items():
        values = info['values']
        if values:
            print(f"\n{metrics_to_plot[tag]} Summary:")
            print(f"  Final: {values[-1]:.4f}")
            print(f"  Best:  {max(values):.4f}")
            print(f"  Mean:  {np.mean(values):.4f}")
            print(f"  Std:   {np.std(values):.4f}")
    
    # Try to show the plot if in interactive mode
    try:
        plt.show()
    except:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot learning curves from tensorboard logs')
    parser.add_argument('--log-dir', type=str, 
                       default='/home/ssaika/rl-transfer-sec-clean/runs/cw-ppo-training/PPO_2',
                       help='Path to tensorboard log directory')
    args = parser.parse_args()
    
    plot_learning_curve_from_tensorboard(args.log_dir)
