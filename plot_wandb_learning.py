#!/usr/bin/env python3
"""
Plot learning curves directly from W&B offline runs.
"""

import argparse
import os
import glob
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def find_wandb_tensorboard_logs(wandb_dir: str, run_name: str = None) -> str:
    """Find TensorBoard logs in a W&B run directory."""
    if run_name:
        run_path = os.path.join(wandb_dir, run_name)
    else:
        # Find the latest run
        runs = [d for d in os.listdir(wandb_dir) if os.path.isdir(os.path.join(wandb_dir, d)) and d.startswith('offline-run-')]
        if not runs:
            raise ValueError(f"No W&B runs found in {wandb_dir}")
        run_path = os.path.join(wandb_dir, max(runs, key=lambda d: os.path.getmtime(os.path.join(wandb_dir, d))))
    
    # W&B syncs TensorBoard logs to files/ directory
    tb_logs = os.path.join(run_path, 'files', 'events.out.tfevents.*')
    matching = glob.glob(tb_logs)
    
    if matching:
        # Return parent directory containing the event files
        return os.path.dirname(matching[0])
    
    # Fallback: check if there are event files directly in the run directory
    tb_alt = os.path.join(run_path, 'events.out.tfevents.*')
    matching = glob.glob(tb_alt)
    if matching:
        return run_path
    
    raise ValueError(f"No TensorBoard logs found in W&B run: {run_path}")


def moving_average(values: List[float], window: int) -> List[float]:
    if window is None or window <= 1 or len(values) < 2:
        return values
    window = min(window, max(2, len(values)))
    kernel = np.ones(window) / window
    padded = np.pad(values, (window // 2, window - 1 - window // 2), mode='edge')
    smoothed = np.convolve(padded, kernel, mode='valid')
    return smoothed.tolist()


def load_scalars_from_wandb(wandb_dir: str, run_name: str = None, include_only: List[str] = None, mark_evals: bool = False) -> Dict[str, Dict[str, list]]:
    """Load scalar data from W&B TensorBoard logs."""
    log_dir = find_wandb_tensorboard_logs(wandb_dir, run_name)
    print(f"Loading TensorBoard logs from: {log_dir}")
    
    ea = EventAccumulator(log_dir)
    ea.Reload()
    tags = set(ea.Tags().get('scalars', []))
    
    # Default metrics for Cyberwheel
    default_metrics = [
        'charts/red_episodic_return',
        'losses/red_policy_loss',
        'losses/red_value_loss',
        'losses/red_entropy',
    ]
    wanted = include_only if include_only else default_metrics
    
    titles = {
        'charts/red_episodic_return': 'Red Episodic Return',
        'losses/red_policy_loss': 'Policy Loss',
        'losses/red_value_loss': 'Value Loss',
        'losses/red_entropy': 'Entropy',
        'charts/SPS': 'Steps Per Second',
        'charts/red_actor_lr': 'Actor Learning Rate',
        'charts/red_critic_lr': 'Critic Learning Rate',
    }
    
    data = {}
    eval_steps = []
    # Load eval_time separately just for checkpoint marking, don't plot it
    if 'charts/eval_time' in tags and mark_evals:
        scalar_events = ea.Scalars('charts/eval_time')
        eval_steps = [e.step for e in scalar_events]
    
    for tag in wanted:
        if tag in tags:
            scalar_events = ea.Scalars(tag)
            steps = [e.step for e in scalar_events]
            values = [e.value for e in scalar_events]
            data[tag] = {'title': titles.get(tag, tag), 'steps': steps, 'values': values}
            if mark_evals:
                data[tag]['eval_steps'] = eval_steps
            print(f"  Loaded {len(values)} points for {tag}")
    
    return data


def plot_wandb_metrics(data: Dict[str, Dict[str, list]], output_file: str, smooth: int, mark_evals: bool) -> None:
    if not data:
        print('No matching metrics found in W&B logs.')
        return
    
    n_metrics = len(data)
    n_cols = 2
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
    try:
        import numpy as _np
        axes = _np.array(axes).flatten().tolist()
    except Exception:
        axes = [axes] if not isinstance(axes, (list, tuple)) else axes
    
    for idx, (tag, info) in enumerate(data.items()):
        ax = axes[idx]
        y = moving_average(info['values'], smooth)
        ax.plot(info['steps'], y, linewidth=2)
        ax.set_title(info['title'])
        ax.set_xlabel('Global Step')
        ax.grid(True, alpha=0.3)
        if 'Return' in info['title']:
            ax.axhline(0.0, color='black', linestyle='--', alpha=0.25)
        if mark_evals and 'eval_steps' in info:
            for s in info['eval_steps']:
                ax.axvline(s, color='red', linestyle=':', alpha=0.25, linewidth=1)
    
    # Hide unused axes
    for j in range(len(data), len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f'Saved W&B plot: {output_file}')


def main():
    parser = argparse.ArgumentParser(description='Plot learning curves from W&B offline runs')
    parser.add_argument('--wandb-dir', type=str, default='/home/ssaika/rl-transfer-sec-clean/cyberwheel/wandb', help='W&B directory')
    parser.add_argument('--run', type=str, default=None, help='Specific W&B run name (default: latest)')
    parser.add_argument('--out', type=str, default='/home/ssaika/rl-transfer-sec-clean/cw_wandb_plot.png', help='Output PNG file')
    parser.add_argument('--smooth', type=int, default=5, help='Moving average window')
    parser.add_argument('--only', type=str, default='', help='Comma-separated scalar tags to include')
    parser.add_argument('--mark-evals', action='store_true', help='Mark evaluation checkpoints')
    args = parser.parse_args()
    
    if not os.path.isdir(args.wandb_dir):
        raise SystemExit(f'W&B directory not found: {args.wandb_dir}')
    
    include_only = [s.strip() for s in args.only.split(',') if s.strip()] if args.only else None
    data = load_scalars_from_wandb(args.wandb_dir, args.run, include_only, args.mark_evals)
    plot_wandb_metrics(data, args.out, args.smooth, args.mark_evals)


if __name__ == '__main__':
    main()
