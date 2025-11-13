#!/usr/bin/env python3
"""
Compare original vs improved training configurations.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_metric(run_dir: str, metric_key: str):
    """Load a single metric from TensorBoard log."""
    if not os.path.isdir(run_dir):
        return None, None
    
    try:
        ea = EventAccumulator(run_dir)
        ea.Reload()
        tags = ea.Tags().get('scalars', [])
        
        if metric_key in tags:
            s = ea.Scalars(metric_key)
            steps = [e.step for e in s]
            values = [e.value for e in s]
            return steps, values
    except Exception:
        pass
    
    return None, None


def main():
    runs_dir = '/home/ssaika/rl-transfer-sec-clean/cyberwheel/cyberwheel/data/runs'
    
    metrics = [
        ('charts/red_episodic_return', 'Episodic Return'),
        ('losses/red_policy_loss', 'Policy Loss'),
        ('losses/red_value_loss', 'Value Loss'),
    ]
    
    durations = [500, 1000, 2000, 5000]
    
    fig, axes = plt.subplots(len(metrics), 1, figsize=(14, 12))
    if len(metrics) == 1:
        axes = [axes]
    
    for idx, (metric_key, metric_name) in enumerate(metrics):
        ax = axes[idx]
        
        for duration in durations:
            original_dir = os.path.join(runs_dir, f'CWRun_{duration}')
            improved_dir = os.path.join(runs_dir, f'CWRun_{duration}_improved')
            
            orig_steps, orig_vals = load_metric(original_dir, metric_key)
            impr_steps, impr_vals = load_metric(improved_dir, metric_key)
            
            if orig_steps and orig_vals:
                ax.plot(orig_steps, orig_vals, '--', label=f'{duration} steps (original)', 
                       linewidth=2, alpha=0.7)
            
            if impr_steps and impr_vals:
                ax.plot(impr_steps, impr_vals, '-', label=f'{duration} steps (improved)', 
                       linewidth=2, alpha=0.9)
        
        ax.set_title(metric_name, fontsize=12, fontweight='bold')
        ax.set_xlabel('Training Step', fontsize=10)
        ax.set_ylabel(metric_name, fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(ncol=2, fontsize=8)
        
        if 'Return' in metric_name:
            ax.axhline(y=0, color='black', linestyle=':', alpha=0.3)
    
    plt.suptitle('Original vs Improved Training Configuration Comparison', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    out_file = '/home/ssaika/rl-transfer-sec-clean/original_vs_improved_comparison.png'
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved comparison plot: {out_file}")
    
    # Print summary statistics
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    
    for duration in durations:
        original_dir = os.path.join(runs_dir, f'CWRun_{duration}')
        improved_dir = os.path.join(runs_dir, f'CWRun_{duration}_improved')
        
        orig_steps, orig_vals = load_metric(original_dir, 'charts/red_episodic_return')
        impr_steps, impr_vals = load_metric(improved_dir, 'charts/red_episodic_return')
        
        if orig_vals and impr_vals:
            orig_final = orig_vals[-1]
            impr_final = impr_vals[-1]
            improvement = impr_final - orig_final
            pct_change = (improvement / abs(orig_final) * 100) if orig_final != 0 else 0
            
            print(f"\n{duration} steps:")
            print(f"  Original final return: {orig_final:.2f}")
            print(f"  Improved final return: {impr_final:.2f}")
            print(f"  Improvement: {improvement:+.2f} ({pct_change:+.1f}%)")
        elif impr_vals:
            print(f"\n{duration} steps: Improved run only (no original for comparison)")


if __name__ == '__main__':
    main()

