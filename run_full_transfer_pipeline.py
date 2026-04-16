#!/usr/bin/env python3
"""
Full DAPN transfer pipeline:
  Step 1 — Train DAPN encoder (domain alignment CW ↔ CBS)
  Step 2 — Train PPO on CyberWheel with DAPN-encoded observations
  Step 3 — Evaluate CW-trained policy on CyberBattleSim with same DAPN encoder
            and compare to a CBS-trained baseline (no transfer)
"""

import os
import sys
import json
import subprocess
import argparse
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "cyberwheel"))

import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import PPO

from adapters.unified_env import UnifiedSecEnv
from adapters.dapn_env_wrapper import DAPNEnvWrapper
from config.env_builders import make_cw_env, make_cbs_env


# ---------------------------------------------------------------------------
# Step 1: Train DAPN encoder
# ---------------------------------------------------------------------------
def train_encoder(encoder_samples: int, encoder_epochs: int, encoder_path: str,
                  lambda_pair: float = 0.1):
    print("\n" + "=" * 60)
    print("STEP 1: TRAIN DAPN ENCODER")
    print("=" * 60)

    cmd = [
        sys.executable, str(project_root / "train_dapn_encoder.py"),
        "--num-samples", str(encoder_samples),
        "--epochs", str(encoder_epochs),
        "--save-encoder", encoder_path,
        "--lambda-pair", str(lambda_pair),
    ]
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, check=True)
    if result.returncode != 0:
        raise RuntimeError("DAPN encoder training failed")
    print(f"\n✓ DAPN encoder saved to {encoder_path}")


# ---------------------------------------------------------------------------
# Step 2: Train PPO on CyberWheel with DAPN-encoded obs
# ---------------------------------------------------------------------------
def train_source_policy(encoder_path: str, timesteps: int, policy_path: str,
                        feature_size: int = 256):
    print("\n" + "=" * 60)
    print("STEP 2: TRAIN PPO ON CYBERWHEEL (DAPN-encoded obs)")
    print("=" * 60)

    base_env = UnifiedSecEnv("cw", cw_factory=make_cw_env)
    env = DAPNEnvWrapper(base_env, encoder_path=encoder_path, feature_size=feature_size)

    print(f"  Obs space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")

    # DAPNEnvWrapper always returns Dict obs → use MultiInputPolicy
    model = PPO("MultiInputPolicy", env, verbose=1,
                tensorboard_log="artifacts/tb/cw_dapn")
    model.learn(total_timesteps=timesteps)

    os.makedirs(os.path.dirname(policy_path), exist_ok=True)
    model.save(policy_path)
    print(f"\n✓ CW policy saved to {policy_path}")
    return model


# ---------------------------------------------------------------------------
# Step 3: Evaluate on CBS
# ---------------------------------------------------------------------------
def _run_episodes(model, env, n_episodes: int, max_steps: int) -> dict:
    """Run evaluation episodes; handles both MultiInputPolicy and MlpPolicy."""
    _model_is_multi = hasattr(model, "observation_space") and isinstance(
        model.observation_space, gym.spaces.Dict
    )
    returns, steps_list, successes = [], [], 0

    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = truncated = False
        ep_return = 0.0
        step = 0
        while not (done or truncated) and step < max_steps:
            obs_input = obs if _model_is_multi else (obs["obs"] if isinstance(obs, dict) else obs)
            action, _ = model.predict(obs_input, deterministic=True)
            obs, reward, done, truncated, _ = env.step(action)
            ep_return += float(reward)
            step += 1
        returns.append(ep_return)
        steps_list.append(step)
        if done and not truncated:
            successes += 1

    return {
        "avg_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "avg_steps": float(np.mean(steps_list)),
        "success_rate": float(successes) / n_episodes,
    }


def evaluate_transfer(encoder_path: str, cw_policy_path: str,
                      n_episodes: int = 10, max_steps: int = 5000,
                      feature_size: int = 256):
    print("\n" + "=" * 60)
    print("STEP 3: EVALUATE ON CYBERBATTLESIM")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Without transfer: train a fresh PPO directly on CBS ---
    print("\nTraining CBS baseline (no transfer)...")
    cbs_base = UnifiedSecEnv("cbs", cbs_factory=make_cbs_env)
    baseline = PPO("MultiInputPolicy", cbs_base, verbose=0,
                   tensorboard_log="artifacts/tb/cbs_baseline")
    baseline.learn(total_timesteps=5000)

    cbs_eval_base = UnifiedSecEnv("cbs", cbs_factory=make_cbs_env)
    results_no_transfer = _run_episodes(baseline, cbs_eval_base, n_episodes, max_steps)
    print(f"  [No Transfer]  avg_return={results_no_transfer['avg_return']:.2f}  "
          f"success={results_no_transfer['success_rate']:.0%}")

    # --- With transfer: load CW policy, run on CBS with DAPN wrapper ---
    print("\nEvaluating CW policy on CBS via DAPN encoder (with transfer)...")
    cw_policy = PPO.load(cw_policy_path, device=device)

    cbs_base2 = UnifiedSecEnv("cbs", cbs_factory=make_cbs_env)
    cbs_dapn = DAPNEnvWrapper(cbs_base2, encoder_path=encoder_path, feature_size=feature_size)
    results_transfer = _run_episodes(cw_policy, cbs_dapn, n_episodes, max_steps)
    print(f"  [With Transfer] avg_return={results_transfer['avg_return']:.2f}  "
          f"success={results_transfer['success_rate']:.0%}")

    improvement = {
        "return_delta": results_transfer["avg_return"] - results_no_transfer["avg_return"],
        "success_rate_delta": results_transfer["success_rate"] - results_no_transfer["success_rate"],
    }
    print(f"\n  Return Δ: {improvement['return_delta']:+.2f}")
    print(f"  Success Rate Δ: {improvement['success_rate_delta']*100:+.1f}%")

    return {
        "no_transfer": results_no_transfer,
        "with_transfer": results_transfer,
        "improvement": improvement,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Full DAPN transfer learning pipeline")
    parser.add_argument("--encoder_samples", type=int, default=1000,
                        help="Obs samples per domain for DAPN encoder training")
    parser.add_argument("--encoder_epochs", type=int, default=50)
    parser.add_argument("--lambda_pair", type=float, default=0.1,
                        help="Pair alignment loss weight in DAPN training")
    parser.add_argument("--source_timesteps", type=int, default=10000,
                        help="PPO timesteps on CyberWheel")
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--feature_size", type=int, default=256)
    parser.add_argument("--encoder_path", type=str,
                        default="artifacts/transfer_models/dapn_encoder.pt")
    parser.add_argument("--policy_path", type=str,
                        default="artifacts/transfer_models/cw_dapn_policy")
    parser.add_argument("--output_dir", type=str, default="artifacts/plots")
    parser.add_argument("--skip_encoder", action="store_true",
                        help="Skip Step 1 (use existing encoder checkpoint)")
    parser.add_argument("--skip_source_train", action="store_true",
                        help="Skip Step 2 (use existing CW policy checkpoint)")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("DAPN TRANSFER LEARNING — FULL PIPELINE")
    print("=" * 60)

    # Step 1
    if not args.skip_encoder:
        train_encoder(args.encoder_samples, args.encoder_epochs,
                      args.encoder_path, lambda_pair=args.lambda_pair)
    else:
        print(f"\nSkipping encoder training — using {args.encoder_path}")

    # Step 2
    if not args.skip_source_train:
        train_source_policy(args.encoder_path, args.source_timesteps,
                            args.policy_path, feature_size=args.feature_size)
    else:
        print(f"\nSkipping source training — using {args.policy_path}")

    # Step 3
    results = evaluate_transfer(
        encoder_path=args.encoder_path,
        cw_policy_path=args.policy_path,
        n_episodes=args.eval_episodes,
        max_steps=args.max_steps,
        feature_size=args.feature_size,
    )

    # Save
    out_path = os.path.join(args.output_dir, "transfer_results.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to {out_path}")

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
