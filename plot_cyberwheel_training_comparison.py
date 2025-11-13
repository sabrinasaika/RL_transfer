#!/usr/bin/env python3
"""
Plot training metrics comparison across different training durations.
"""

import os
import sys
import argparse
from typing import List, Dict
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_tensorboard_data(log_dir: str) -> Dict[str, List]:
    """Load scalar metrics from TensorBoard log directory."""
    if not os.path.isdir(log_dir):
        return {}
    
    try:
        ea = EventAccumulator(log_dir)
        ea.Reload()
        tags = ea.Tags().get('scalars', [])
        
        data = {}
        # Map to actual metric names - use episodic return for total reward
        metrics_map = {
            'charts/red_episodic_return': 'charts/red_episodic_return',
            'charts/total_reward': 'charts/red_episodic_return',  # Use episodic return as total reward
            'losses/red_policy_loss': 'losses/red_policy_loss',
            'losses/red_value_loss': 'losses/red_value_loss',
        }
        
        for key, metric in metrics_map.items():
            if metric in tags:
                scalar_events = ea.Scalars(metric)
                steps = [e.step for e in scalar_events]
                values = [e.value for e in scalar_events]
                data[key] = {'steps': steps, 'values': values}
        
        return data
    except Exception as e:
        print(f"Warning: Could not load TensorBoard data from {log_dir}: {e}")
        return {}


def plot_comparison(run_dirs: Dict[int, str], output_file: str):
    """Create comparison plots for different training durations."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    metrics_config = [
        ('charts/red_episodic_return', 'Episodic Return', axes[0, 0], 'blue'),
        ('charts/total_reward', 'Total Reward', axes[0, 1], 'green'),
        ('losses/red_policy_loss', 'Policy Loss', axes[1, 0], 'red'),
        ('losses/red_value_loss', 'Value Loss', axes[1, 1], 'orange'),
    ]
    
    for metric_key, metric_title, ax, default_color in metrics_config:
        for steps, run_dir in sorted(run_dirs.items()):
            data = load_tensorboard_data(run_dir)
            if metric_key in data:
                ax.plot(data[metric_key]['steps'], data[metric_key]['values'], 
                       label=f'{steps} steps', linewidth=2, alpha=0.8)
        
        ax.set_title(metric_title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Training Step', fontsize=10)
        ax.set_ylabel(metric_title, fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        if 'Return' in metric_title or 'Reward' in metric_title:
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    
    plt.suptitle('Cyberwheel Training Metrics Comparison', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved comparison plot: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Plot training comparison across different durations')
    parser.add_argument('--runs-dir', type=str, 
                       default='/home/ssaika/rl-transfer-sec-clean/cyberwheel/cyberwheel/data/runs',
                       help='Directory containing training run directories')
    parser.add_argument('--out', type=str, 
                       default='/home/ssaika/rl-transfer-sec-clean/cyberwheel_training_comparison.png',
                       help='Output plot file')
    parser.add_argument('--improved', action='store_true',
                       help='Use improved configurations (CWRun_*_improved)')
    
    args = parser.parse_args()
    
    # Expected run directory names
    suffix = '_improved' if args.improved else ''
    run_dirs = {
        500: os.path.join(args.runs_dir, f'CWRun_500{suffix}'),
        1000: os.path.join(args.runs_dir, f'CWRun_1000{suffix}'),
        2000: os.path.join(args.runs_dir, f'CWRun_2000{suffix}'),
        5000: os.path.join(args.runs_dir, f'CWRun_5000{suffix}'),
    }
    
    # Filter out non-existent directories
    existing_runs = {k: v for k, v in run_dirs.items() if os.path.isdir(v)}
    
    if not existing_runs:
        print(f"❌ No training runs found in {args.runs_dir}")
        run_type = "improved" if args.improved else "original"
        print(f"   Expected {run_type} directories: CWRun_500{suffix}, CWRun_1000{suffix}, etc.")
        return
    
    run_type = "improved" if args.improved else "original"
    print(f"Found {len(existing_runs)} {run_type} training runs:")
    for steps, run_dir in sorted(existing_runs.items()):
        print(f"  {steps} steps: {run_dir}")
    
    plot_comparison(existing_runs, args.out)


if __name__ == '__main__':
    main()

