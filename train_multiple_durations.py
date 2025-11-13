#!/usr/bin/env python3
"""
Train Cyberwheel agents for different durations: 500, 1000, 2000, 5000 steps.
"""

import os
import sys
import subprocess
from pathlib import Path

# Add cyberwheel to path
sys.path.insert(0, str(Path(__file__).parent / 'cyberwheel'))

training_configs = {
    500: 'train_500_steps.yaml',
    1000: 'train_1000_steps.yaml',
    2000: 'train_2000_steps.yaml',
    5000: 'train_5000_steps.yaml',
}

def train_for_duration(steps: int, config_file: str):
    """Train agent for specified number of steps."""
    print(f"\n{'='*60}")
    print(f"Training for {steps} steps")
    print(f"{'='*60}")
    
    config_path = f'cyberwheel/data/configs/environment/{config_file}'
    
    cmd = [
        sys.executable, '-m', 'cyberwheel', 'train', config_path
    ]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        cwd=str(Path(__file__).parent / 'cyberwheel'),
        capture_output=False,
        text=True
    )
    
    if result.returncode == 0:
        print(f"\n✓ Training completed for {steps} steps")
    else:
        print(f"\n❌ Training failed for {steps} steps (exit code: {result.returncode})")
    
    return result.returncode == 0


def main():
    base_dir = Path(__file__).parent
    os.chdir(base_dir / 'cyberwheel')
    
    results = {}
    for steps, config_file in training_configs.items():
        success = train_for_duration(steps, config_file)
        results[steps] = success
    
    print(f"\n{'='*60}")
    print("TRAINING SUMMARY")
    print(f"{'='*60}")
    for steps, success in results.items():
        status = "✓" if success else "❌"
        print(f"{status} {steps} steps: {'Success' if success else 'Failed'}")
    
    print(f"\n✓ All training runs completed!")
    print(f"Run the plotting script to visualize results:")
    print(f"  python {base_dir}/plot_cyberwheel_training_comparison.py")


if __name__ == '__main__':
    main()

