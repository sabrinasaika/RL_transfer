#!/usr/bin/env python3
"""
Train a Stable-Baselines3 PPO agent on CyberBattleSim via UnifiedSecEnv.

Observation space is a Dict {'obs', 'mask'} → use MultiInputPolicy.

Example:
  PYTHONPATH=$PWD:$PWD/cyberwheel:$PWD/CyberBattleSim \\
    python train_cbs_agent.py --timesteps 50000 --out artifacts/cbs_agent.zip
"""

import argparse
import os
import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "cyberwheel"))

from stable_baselines3 import PPO

from adapters.unified_env import UnifiedSecEnv
from config.env_builders import make_cbs_env


def main():
    parser = argparse.ArgumentParser(description="Train PPO on CyberBattleSim (unified obs)")
    parser.add_argument("--timesteps", type=int, default=50_000, help="PPO learn() timesteps")
    parser.add_argument("--out", type=str, default="artifacts/cbs_agent.zip", help="Output .zip path")
    args = parser.parse_args()

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    env = UnifiedSecEnv("cbs", cbs_factory=make_cbs_env)
    model = PPO("MultiInputPolicy", env, verbose=1)
    model.learn(total_timesteps=args.timesteps)
    model.save(args.out)
    print(f"Saved PPO model to {args.out}")


if __name__ == "__main__":
    main()
