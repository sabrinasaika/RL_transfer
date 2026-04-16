"""
Train a CW policy on top of frozen DAPN-encoded observations (256D).

Pipeline:
  CW env → raw obs → 512D (UnifiedFullObsPreprocessor) → DAPN encoder (frozen) → 256D → PPO

After training, the policy can be evaluated zero-shot on CBS because the DAPN encoder
maps CBS obs to the same 256D feature space.

Usage:
  PYTHONPATH=$PWD:$PWD/cyberwheel:$PWD/CyberBattleSim python3.10 train_cw_dapn_policy.py \
    --encoder artifacts/transfer_models/dapn_encoder.pt \
    --timesteps 200000 \
    --out artifacts/policies/cw_dapn_policy.zip
"""

import os, sys, argparse
from pathlib import Path
import numpy as np
import gymnasium as gym
from gymnasium import spaces

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "cyberwheel"))

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from adapters.unified_env import UnifiedSecEnv
from adapters.dapn_env_wrapper import DAPNEnvWrapper
from config.env_builders import make_cw_env, make_cbs_env


# ── Env factory helpers ──────────────────────────────────────────────────────

def make_cw_dapn_env(encoder_path: str, feature_size: int = 256):
    """Create CW env wrapped with DAPN encoder → returns 256D obs."""
    def _factory():
        base = UnifiedSecEnv("cw", cw_factory=make_cw_env)
        wrapped = DAPNEnvWrapper(
            base,
            encoder_path=encoder_path,
            feature_size=feature_size,
            use_dapn=True,
        )
        return wrapped
    return _factory


def make_cbs_dapn_env(encoder_path: str, feature_size: int = 256):
    """Create CBS env wrapped with DAPN encoder → returns 256D obs."""
    def _factory():
        base = UnifiedSecEnv("cbs", cbs_factory=make_cbs_env)
        wrapped = DAPNEnvWrapper(
            base,
            encoder_path=encoder_path,
            feature_size=feature_size,
            use_dapn=True,
        )
        return wrapped
    return _factory


# ── Sanity check: verify encoder returns 256D obs ────────────────────────────

def sanity_check(encoder_path: str, feature_size: int):
    print("Sanity check: CW env + DAPN encoder...")
    env = make_cw_dapn_env(encoder_path, feature_size)()
    obs, _ = env.reset()
    obs_arr = obs["obs"] if isinstance(obs, dict) else obs
    obs_arr = np.asarray(obs_arr).ravel()
    print(f"  CW obs shape after encoding: {obs_arr.shape}  "
          f"(expected {feature_size}D)")
    nonzero = int((obs_arr != 0).sum())
    print(f"  Nonzero elements: {nonzero}/{feature_size}")
    env.close()
    assert obs_arr.shape[0] == feature_size, (
        f"Expected {feature_size}D obs, got {obs_arr.shape[0]}D")
    print("  ✓ Encoder output matches expected feature size")


# ── Main training ─────────────────────────────────────────────────────────────

def train(encoder_path: str,
          timesteps: int = 200_000,
          feature_size: int = 256,
          out_path: str = "artifacts/policies/cw_dapn_policy.zip",
          n_envs: int = 1,
          seed: int = 42):

    sanity_check(encoder_path, feature_size)

    print(f"\nTraining CW PPO policy on {feature_size}D DAPN-encoded observations")
    print(f"  Encoder : {encoder_path}")
    print(f"  Steps   : {timesteps}")
    print(f"  Output  : {out_path}\n")

    # Vectorised CW envs
    env_fns = [make_cw_dapn_env(encoder_path, feature_size) for _ in range(n_envs)]
    venv = VecMonitor(DummyVecEnv(env_fns))

    model = PPO(
        "MultiInputPolicy",
        venv,
        verbose=1,
        seed=seed,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        learning_rate=3e-4,
        ent_coef=0.01,
        clip_range=0.2,
    )

    model.learn(total_timesteps=timesteps)

    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    model.save(out_path.replace(".zip", ""))
    print(f"\n✓ Saved CW DAPN policy to {out_path}")
    venv.close()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", default="artifacts/transfer_models/dapn_encoder.pt")
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--feature-size", type=int, default=256)
    parser.add_argument("--out", default="artifacts/policies/cw_dapn_policy.zip")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train(
        encoder_path=args.encoder,
        timesteps=args.timesteps,
        feature_size=args.feature_size,
        out_path=args.out,
        seed=args.seed,
    )
