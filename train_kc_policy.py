"""
Train a PPO policy on CyberWheel using the kill-chain intent interface.

Two modes:
  --encoder   (default) : 512-D obs → frozen DAPN encoder → 256-D → PPO
  --no-encoder          :            512-D obs directly → PPO  (ablation baseline)

Action space (both modes): Discrete(9) kill-chain intent
  action i  → advance the kill-chain on host slot i
  action 8  → no-op

Usage:
  # With DAPN encoder (for transfer):
  PYTHONPATH=$PWD:$PWD/cyberwheel:$PWD/CyberBattleSim python3.10 train_kc_policy.py \\
      --encoder artifacts/transfer_models/dapn_encoder.pt \\
      --timesteps 50000 \\
      --out artifacts/policies/cw_kc_dapn_policy.zip

  # Without encoder (ablation baseline):
  PYTHONPATH=$PWD:$PWD/cyberwheel:$PWD/CyberBattleSim python3.10 train_kc_policy.py \\
      --no-encoder \\
      --timesteps 50000 \\
      --out artifacts/policies/cw_kc_raw_policy.zip
"""

import os, sys, argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "cyberwheel"))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback

try:
    import tqdm  # noqa: F401
    import rich  # noqa: F401
    _SB3_PROGRESS_BAR = True
except ImportError:
    _SB3_PROGRESS_BAR = False

from adapters.unified_env import UnifiedSecEnv
from config.env_builders import make_cw_env


def make_env(encoder_path: str | None, seed: int = 0):
    """Factory that returns a callable creating one training env."""
    def _factory():
        base = UnifiedSecEnv("cw", cw_factory=make_cw_env)
        if encoder_path:
            from adapters.kc_dapn_wrapper import KCDAPNWrapper
            return KCDAPNWrapper(base, encoder_path=encoder_path)
        return base
    return _factory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder",    default="artifacts/transfer_models/dapn_encoder.pt",
                        help="Path to frozen DAPN encoder checkpoint (512→256)")
    parser.add_argument("--no-encoder", action="store_true",
                        help="Train directly on 512-D obs without DAPN (ablation)")
    parser.add_argument("--timesteps",  type=int, default=50_000)
    parser.add_argument("--out",        default=None,
                        help="Output policy path (auto-named if omitted)")
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    encoder_path = None if args.no_encoder else args.encoder
    if encoder_path and not Path(encoder_path).exists():
        print(f"[warn] Encoder not found at {encoder_path} — running without DAPN")
        encoder_path = None

    # Auto-name output
    if args.out is None:
        tag  = "raw" if encoder_path is None else "dapn"
        args.out = f"artifacts/policies/cw_kc_{tag}_policy.zip"
    os.makedirs(Path(args.out).parent, exist_ok=True)

    mode = "DAPN (512→256)" if encoder_path else "raw 512-D"
    print(f"Training PPO on CyberWheel — kill-chain intent  [{mode}]")
    print(f"  timesteps : {args.timesteps:,}")
    print(f"  encoder   : {encoder_path or 'none'}")
    print(f"  output    : {args.out}")

    train_env = VecMonitor(DummyVecEnv([make_env(encoder_path, args.seed)]))
    eval_env  = VecMonitor(DummyVecEnv([make_env(encoder_path, args.seed + 1)]))

    obs_dim = train_env.observation_space.shape[0]
    print(f"  obs dim   : {obs_dim}  |  action space: {train_env.action_space}")

    # DAPN encoder compresses 512-D → 256-D domain-invariant features.
    # The compressed representation is harder to read than raw 512-D obs,
    # so DAPN training needs stronger exploration (higher ent_coef) to prevent
    # the policy from prematurely committing to the no-op action.
    ent_coef = 0.05 if encoder_path else 0.01

    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        seed=args.seed,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=ent_coef,
    )

    eval_cb = EvalCallback(
        eval_env,
        n_eval_episodes=5,
        eval_freq=max(1, args.timesteps // 10),
        best_model_save_path=str(Path(args.out).parent / f"best_kc_{'dapn' if encoder_path else 'raw'}"),
        verbose=1,
    )

    model.learn(
        total_timesteps=args.timesteps,
        callback=eval_cb,
        progress_bar=_SB3_PROGRESS_BAR,
    )
    model.save(args.out)
    print(f"\nSaved → {args.out}")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
