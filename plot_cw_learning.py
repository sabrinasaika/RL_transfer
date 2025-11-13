#!/usr/bin/env python3
import argparse
import os
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def moving_average(values: List[float], window: int) -> List[float]:
    if window is None or window <= 1 or len(values) < 2:
        return values
    window = min(window, max(2, len(values)))
    kernel = np.ones(window) / window
    # 'same' keeps length, pad at ends to avoid shrink
    padded = np.pad(values, (window // 2, window - 1 - window // 2), mode='edge')
    smoothed = np.convolve(padded, kernel, mode='valid')
    return smoothed.tolist()


def load_scalars(log_dir: str, include_only: List[str], mark_evals: bool = False) -> Dict[str, Dict[str, list]]:
    ea = EventAccumulator(log_dir)
    ea.Reload()
    tags = set(ea.Tags().get('scalars', []))

    # Default minimal metrics
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
    return data


def plot_metrics(data: Dict[str, Dict[str, list]], output_file: str, smooth: int, mark_evals: bool) -> None:
    if not data:
        print('No matching metrics found in TensorBoard logs.')
        return

    n_metrics = len(data)
    n_cols = 2
    n_rows = (n_metrics + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3.6 * n_rows))
    # Normalize to a flat list of Axes
    try:
        import numpy as _np
        axes = _np.array(axes).flatten().tolist()
    except Exception:
        axes = [axes]

    for idx, (tag, info) in enumerate(data.items()):
        ax = axes[idx]
        y = moving_average(info['values'], smooth)
        ax.plot(info['steps'], y, linewidth=2)
        ax.set_title(info['title'])
        ax.set_xlabel('Global Step')
        ax.grid(True, alpha=0.3)
        if 'Return' in info['title']:
            ax.axhline(0.0, color='black', linestyle='--', alpha=0.25)
        # Mark evaluation checkpoints as vertical lines
        if mark_evals and 'eval_steps' in info:
            for s in info['eval_steps']:
                ax.axvline(s, color='red', linestyle=':', alpha=0.25, linewidth=1)

    # Hide any extra axes
    for j in range(len(data), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f'Saved plot: {output_file}')


def main():
    parser = argparse.ArgumentParser(description='Plot CW learning curves (clean) from TensorBoard logs')
    parser.add_argument('--log-dir', type=str, default='/home/ssaika/rl-transfer-sec-clean/cyberwheel/cyberwheel/data/runs/CWRun', help='Path to TensorBoard run directory')
    parser.add_argument('--out', type=str, default='/home/ssaika/rl-transfer-sec-clean/cw_learning_curves_clean.png', help='Output PNG file')
    parser.add_argument('--smooth', type=int, default=3, help='Moving average window (points)')
    parser.add_argument('--only', type=str, default='', help='Comma-separated scalar tags to include')
    parser.add_argument('--mark-evals', action='store_true', help='Mark evaluation checkpoints')
    args = parser.parse_args()

    if not os.path.isdir(args.log_dir):
        raise SystemExit(f'Log directory not found: {args.log_dir}')

    include_only = [s.strip() for s in args.only.split(',') if s.strip()] if args.only else []
    data = load_scalars(args.log_dir, include_only, args.mark_evals)
    plot_metrics(data, args.out, args.smooth, args.mark_evals)


if __name__ == '__main__':
    main()


