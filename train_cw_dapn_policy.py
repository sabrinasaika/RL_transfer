"""
Train a CW policy on either DAPN-encoded (256D) or raw unified (8D) observations.

DAPN mode (default):
  CW env → raw obs → 512D (UnifiedFullObsPreprocessor) → DAPN encoder (frozen) → 256D → PPO
  After training, policy transfers zero-shot to CBS via the shared encoder.

Raw / no-DAPN mode (--no-dapn):
  CW env → ObservationTranslator → 8D unified obs → PPO
  Both CW and CBS map to the same 8-dimensional hand-crafted feature vector,
  so the policy can be evaluated on CBS via UnifiedSecEnv with no encoder.
  This is the ablation baseline: DAPN vs no domain-adaptation.

Usage:
  # DAPN policy (for transfer with domain adaptation):
  PYTHONPATH=$PWD:$PWD/cyberwheel:$PWD/CyberBattleSim python3.10 train_cw_dapn_policy.py \
    --encoder artifacts/transfer_models/dapn_encoder.pt \
    --timesteps 200000 \
    --out artifacts/policies/cw_dapn_policy.zip

  # Raw 8D policy (ablation baseline — no domain adaptation):
  PYTHONPATH=$PWD:$PWD/cyberwheel:$PWD/CyberBattleSim python3.10 train_cw_dapn_policy.py \
    --no-dapn \
    --timesteps 200000 \
    --out artifacts/policies/cw_raw_policy.zip
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
    """Create CW env wrapped with DAPN encoder → returns {"obs": 256D, "mask": 7D}."""
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


def make_cw_raw_env():
    """
    Create CW env with NO encoder — ObservationTranslator only → {"obs": 8D, "mask": 7D}.

    The 8D unified vector uses identical field semantics for both CW and CBS:
      [discovered_nodes, compromised_hosts, total_hosts, known_vulns,
       credentials, steps_elapsed, dist_to_goal, alerts]
    This allows zero-shot transfer to CBS without any learned domain adaptation.
    """
    from gymnasium import spaces
    from adapters.action_translator import ActionTranslator

    def _factory():
        base = UnifiedSecEnv("cw", cw_factory=make_cw_env)

        # Wrap in a thin gym.Wrapper that promotes the flat 8D obs to a Dict
        # ({"obs": 8D, "mask": 7D}) so the policy architecture matches the
        # DAPN condition and the eval script can use a single rollout function.
        class RawDictWrapper(gym.Wrapper):
            def __init__(self, env):
                super().__init__(env)
                n_act = len(ActionTranslator().unified_actions)
                from adapters.observation_translator import OBS_DIM
                self.observation_space = spaces.Dict({
                    "obs": spaces.Box(low=0.0, high=1.0,
                                      shape=(OBS_DIM,), dtype=np.float32),
                    "mask": spaces.Box(low=0.0, high=1.0,
                                       shape=(n_act,), dtype=np.float32),
                })

            def _wrap(self, obs):
                flat = obs["obs"] if isinstance(obs, dict) else np.asarray(obs, dtype=np.float32)
                mask = self.env._compute_unified_mask()
                return {"obs": np.asarray(flat, dtype=np.float32), "mask": mask}

            def reset(self, **kwargs):
                obs, info = self.env.reset(**kwargs)
                return self._wrap(obs), info

            def step(self, action):
                obs, r, done, trunc, info = self.env.step(action)
                return self._wrap(obs), r, done, trunc, info

        return RawDictWrapper(base)
    return _factory


def make_cbs_dapn_env(encoder_path: str, feature_size: int = 256):
    """Create CBS env wrapped with DAPN encoder → returns {"obs": 256D, "mask": 7D}."""
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


# ── Sanity check ─────────────────────────────────────────────────────────────

def sanity_check(use_dapn: bool, encoder_path: str = "", feature_size: int = 256):
    from adapters.observation_translator import OBS_DIM
    if use_dapn:
        print("Sanity check: CW env + DAPN encoder...")
        env = make_cw_dapn_env(encoder_path, feature_size)()
        expected = feature_size
    else:
        print("Sanity check: CW env + raw 8D obs (no encoder)...")
        env = make_cw_raw_env()()
        expected = OBS_DIM

    obs, _ = env.reset()
    obs_arr = obs["obs"] if isinstance(obs, dict) else np.asarray(obs, dtype=np.float32)
    obs_arr = np.asarray(obs_arr).ravel()
    print(f"  CW obs shape: {obs_arr.shape}  (expected {expected}D)")
    nonzero = int((obs_arr != 0).sum())
    print(f"  Nonzero elements: {nonzero}/{expected}")
    env.close()
    assert obs_arr.shape[0] == expected, (
        f"Expected {expected}D obs, got {obs_arr.shape[0]}D")
    print("  ✓ Observation shape correct")


# ── Main training ─────────────────────────────────────────────────────────────

def train(use_dapn: bool = True,
          encoder_path: str = "",
          timesteps: int = 200_000,
          feature_size: int = 256,
          out_path: str = "artifacts/policies/cw_dapn_policy.zip",
          n_envs: int = 1,
          seed: int = 42):

    sanity_check(use_dapn, encoder_path, feature_size)

    if use_dapn:
        print(f"\nTraining CW PPO policy on {feature_size}D DAPN-encoded observations")
        print(f"  Encoder : {encoder_path}")
        env_fns = [make_cw_dapn_env(encoder_path, feature_size) for _ in range(n_envs)]
    else:
        from adapters.observation_translator import OBS_DIM
        print(f"\nTraining CW PPO policy on raw {OBS_DIM}D unified observations (no DAPN)")
        print(f"  No encoder — ObservationTranslator only")
        env_fns = [make_cw_raw_env() for _ in range(n_envs)]

    print(f"  Steps   : {timesteps}")
    print(f"  Output  : {out_path}\n")

    venv = VecMonitor(DummyVecEnv(env_fns))

    model = PPO(
        "MultiInputPolicy",   # works for both: Dict{obs, mask} in both modes
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
    label = "DAPN" if use_dapn else "raw-8D"
    print(f"\n✓ Saved CW {label} policy to {out_path}")
    venv.close()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", default="artifacts/transfer_models/dapn_encoder.pt",
                        help="Path to DAPN encoder checkpoint (ignored with --no-dapn)")
    parser.add_argument("--no-dapn", action="store_true",
                        help="Train on raw 8D unified obs (no encoder). "
                             "Use this to produce the ablation baseline for "
                             "eval_cw_dapn_on_cbs.py --raw-policy.")
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--feature-size", type=int, default=256)
    parser.add_argument("--out", default=None,
                        help="Output path. Defaults to cw_dapn_policy.zip or "
                             "cw_raw_policy.zip depending on mode.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    use_dapn = not args.no_dapn
    out_path = args.out or (
        "artifacts/policies/cw_dapn_policy.zip" if use_dapn
        else "artifacts/policies/cw_raw_policy.zip"
    )

    train(
        use_dapn=use_dapn,
        encoder_path=args.encoder,
        timesteps=args.timesteps,
        feature_size=args.feature_size,
        out_path=out_path,
        seed=args.seed,
    )
