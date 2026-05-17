"""
Few-shot fine-tuning of the CW-trained kill-chain policy on CyberBattleSim.

Takes the best raw CW policy (trained on 512-D CW obs) and fine-tunes it on
CBS using DAPN-translated observations.  DAPN bridges the domain gap; a small
number of CBS steps then adapts the policy to actual CBS rewards without
catastrophic forgetting.

Usage:
  PYTHONPATH=$PWD:$PWD/cyberwheel:$PWD/CyberBattleSim python3.10 finetune_kc_cbs.py
  PYTHONPATH=$PWD:$PWD/cyberwheel:$PWD/CyberBattleSim python3.10 finetune_kc_cbs.py \\
      --raw-policy   artifacts/policies/best_kc_raw/best_model.zip \\
      --encoder      artifacts/transfer_models/dapn_encoder_v2.pt \\
      --timesteps    10000
"""

import os, sys, argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "cyberwheel"))
sys.path.insert(0, str(Path(__file__).parent / "CyberBattleSim"))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback

from gymnasium.wrappers import TimeLimit
from adapters.unified_env import UnifiedSecEnv
from adapters.kc_dapn_translate_wrapper import KCDAPNTranslateWrapper
from config.env_builders import make_cbs_env

# Cap episode length to prevent CBS credential cache from growing unbounded.
# The connect action mask is [n_nodes × n_nodes × n_ports × n_creds]; once
# enough credentials accumulate in a long episode it becomes extremely slow.
MAX_EPISODE_STEPS = 100


def make_translated_env(encoder_path: str):
    """Returns a callable that creates a CBS env with DAPN obs-translation."""
    def _factory():
        base = UnifiedSecEnv("cbs", cbs_factory=make_cbs_env)
        base = TimeLimit(base, max_episode_steps=MAX_EPISODE_STEPS)
        return KCDAPNTranslateWrapper(base, encoder_path=encoder_path, device="cpu")
    return _factory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-policy",  default="artifacts/policies/best_kc_raw_12slot/best_kc_raw/best_model.zip")
    parser.add_argument("--encoder",     default="artifacts/transfer_models/dapn_encoder_phase_aware.pt")
    parser.add_argument("--timesteps",   type=int, default=10_000,
                        help="Fine-tuning steps on CBS (few-shot)")
    parser.add_argument("--lr",          type=float, default=1e-4,
                        help="Fine-tuning learning rate (lower than CW training to avoid forgetting)")
    parser.add_argument("--ent-coef",    type=float, default=0.05)
    parser.add_argument("--cbs-size",    type=int, default=12,
                        help="CBS chain size (default 12)")
    parser.add_argument("--win-nodes",   type=int, default=8,
                        help="Nodes to own for a win (default 8). Set 0 to use percent threshold.")
    parser.add_argument("--out",         default="artifacts/policies/kc_finetuned/finetuned_model.zip")
    parser.add_argument("--seed",        type=int, default=42)
    args = parser.parse_args()

    os.environ["CBS_SIZE"] = str(args.cbs_size)
    if args.win_nodes > 0:
        os.environ["CBS_WIN_NODES"] = str(args.win_nodes)
    else:
        os.environ.pop("CBS_WIN_NODES", None)

    if not Path(args.raw_policy).exists():
        print(f"[error] raw policy not found: {args.raw_policy}")
        sys.exit(1)
    if not Path(args.encoder).exists():
        print(f"[error] encoder not found: {args.encoder}")
        sys.exit(1)

    os.makedirs(Path(args.out).parent, exist_ok=True)
    best_dir = str(Path(args.out).parent / "best_finetuned")

    print(f"\nFew-shot CBS fine-tuning")
    print(f"  base policy : {args.raw_policy}")
    print(f"  encoder     : {args.encoder}")
    print(f"  timesteps   : {args.timesteps:,}")
    print(f"  lr          : {args.lr}  ent_coef: {args.ent_coef}")
    print(f"  output      : {args.out}")

    # Load the CW-trained policy
    model = PPO.load(args.raw_policy)

    # Set fine-tuning hypers (lower lr to avoid catastrophic forgetting)
    model.learning_rate = args.lr
    model.ent_coef      = args.ent_coef
    model.seed          = args.seed

    # Wrap CBS env with DAPN translation so the policy sees 512-D CW-like obs
    factory = make_translated_env(args.encoder)
    train_env = VecMonitor(DummyVecEnv([factory]))
    eval_env  = VecMonitor(DummyVecEnv([factory]))

    # Reset optimizers with new lr (SB3 stores the optimizer inside the policy)
    import torch
    for opt in model.policy.optimizer.param_groups:
        opt["lr"] = args.lr

    print(f"  obs space   : {train_env.observation_space.shape}")
    print(f"  action space: {train_env.action_space}")

    # Swap in the CBS environments
    model.set_env(train_env)

    eval_cb = EvalCallback(
        eval_env,
        n_eval_episodes=10,
        eval_freq=max(1, args.timesteps // 10),
        best_model_save_path=best_dir,
        verbose=1,
    )

    # n_steps=256: collect 256 CBS steps per rollout instead of SB3's default 2048.
    # CBS is slow (~200ms/step); smaller rollouts give faster feedback and more
    # frequent gradient updates within the same timestep budget.
    from stable_baselines3.common.buffers import RolloutBuffer
    model.n_steps    = 256
    model.batch_size = 64
    model.rollout_buffer = RolloutBuffer(
        model.n_steps,
        model.observation_space,
        model.action_space,
        device=model.device,
        gamma=model.gamma,
        gae_lambda=model.gae_lambda,
        n_envs=1,
    )

    model.learn(total_timesteps=args.timesteps, callback=eval_cb, reset_num_timesteps=True)
    model.save(args.out)

    train_env.close()
    eval_env.close()

    print(f"\n✓ Fine-tuned policy saved → {args.out}")
    print(f"  Best checkpoint  → {best_dir}/best_model.zip")


if __name__ == "__main__":
    main()
