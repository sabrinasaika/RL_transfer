"""
Evaluate a CW policy (trained on 256D DAPN features) zero-shot on CBS.

Pipeline:
  CBS env → raw obs → 512D (UnifiedFullObsPreprocessor) → DAPN encoder (frozen) → 256D → CW policy

Compares three baselines:
  1. Random policy on CBS
  2. CW policy (raw 8D) on CBS via unified adapter  [8D baseline]
  3. CW DAPN policy (256D) on CBS via DAPN encoder  [DAPN transfer]

Usage:
  PYTHONPATH=$PWD:$PWD/cyberwheel:$PWD/CyberBattleSim python3.10 eval/eval_cw_dapn_on_cbs.py \
    --dapn-policy artifacts/policies/cw_dapn_policy.zip \
    --encoder artifacts/transfer_models/dapn_encoder.pt \
    --episodes 20
"""

import os, sys, argparse
from pathlib import Path
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "cyberwheel"))

from adapters.unified_env import UnifiedSecEnv
from adapters.dapn_env_wrapper import DAPNEnvWrapper
from adapters.kill_chain import stage_from_cbs
from config.env_builders import make_cbs_env


# ── Rollout helper ────────────────────────────────────────────────────────────

def rollout(env, policy, n_episodes: int, max_steps: int = 500,
            deterministic: bool = True, label: str = ""):
    returns, steps_list, final_stages = [], [], []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep)
        obs_in = obs["obs"] if isinstance(obs, dict) and "obs" in obs else obs

        ret, steps = 0.0, 0
        done = truncated = False
        while not (done or truncated) and steps < max_steps:
            if policy is None:
                action = env.action_space.sample()
            else:
                action, _ = policy.predict(obs_in, deterministic=deterministic)
                action = int(np.asarray(action).squeeze())

            obs, r, done, truncated, info = env.step(action)
            obs_in = obs["obs"] if isinstance(obs, dict) and "obs" in obs else obs
            ret += float(r)
            steps += 1

        # Kill-chain stage at episode end
        raw = getattr(env, "_last_raw_obs", None)
        # unwrap DAPNEnvWrapper → UnifiedSecEnv → raw CBS obs
        inner = env
        while hasattr(inner, "env"):
            inner = inner.env
        raw = getattr(inner, "_last_raw_obs", raw)
        stage = stage_from_cbs(raw) if isinstance(raw, dict) else -1

        returns.append(ret)
        steps_list.append(steps)
        final_stages.append(stage)

        print(f"  [{label}] ep={ep+1:>2}  return={ret:>7.2f}  steps={steps:>4}  "
              f"final_stage={stage}")

    from collections import Counter
    stage_dist = Counter(final_stages)
    mean_ret = float(np.mean(returns))
    mean_steps = float(np.mean(steps_list))
    print(f"\n  [{label}] mean_return={mean_ret:.3f}  mean_steps={mean_steps:.1f}")
    print(f"  [{label}] final stage distribution: "
          + "  ".join(f"s{s}={stage_dist[s]}" for s in sorted(stage_dist)))
    return {
        "label": label,
        "returns": returns,
        "mean_return": mean_ret,
        "mean_steps": mean_steps,
        "stage_dist": dict(stage_dist),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dapn-policy",
                        default="artifacts/policies/cw_dapn_policy.zip",
                        help="CW policy trained on 256D DAPN features")
    parser.add_argument("--encoder",
                        default="artifacts/transfer_models/dapn_encoder.pt")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--feature-size", type=int, default=256)
    parser.add_argument("--skip-random", action="store_true",
                        help="Skip random baseline (saves time)")
    args = parser.parse_args()

    results = []

    # ── 1. Random baseline ───────────────────────────────────────────────────
    if not args.skip_random:
        print("\n" + "="*60)
        print("BASELINE 1: Random policy on CBS")
        print("="*60)
        env_rand = UnifiedSecEnv("cbs", cbs_factory=make_cbs_env)
        r = rollout(env_rand, policy=None,
                    n_episodes=args.episodes, max_steps=args.max_steps,
                    label="random")
        results.append(r)
        env_rand.close()

    # ── 2. CW DAPN policy on CBS ─────────────────────────────────────────────
    dapn_policy_path = args.dapn_policy
    if not dapn_policy_path.endswith(".zip"):
        dapn_policy_path += ".zip"

    if not os.path.exists(dapn_policy_path):
        print(f"\n⚠  DAPN policy not found at {dapn_policy_path}")
        print("   Run train_cw_dapn_policy.py first.")
    else:
        from stable_baselines3 import PPO
        print("\n" + "="*60)
        print("TRANSFER: CW DAPN policy (256D) → zero-shot on CBS")
        print("="*60)
        cw_dapn_policy = PPO.load(dapn_policy_path)
        base_cbs = UnifiedSecEnv("cbs", cbs_factory=make_cbs_env)
        env_dapn = DAPNEnvWrapper(
            base_cbs,
            encoder_path=args.encoder,
            feature_size=args.feature_size,
            use_dapn=True,
        )
        r = rollout(env_dapn, policy=cw_dapn_policy,
                    n_episodes=args.episodes, max_steps=args.max_steps,
                    label="cw_dapn→cbs")
        results.append(r)
        env_dapn.close()

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    STAGE_NAMES = ["nothing", "recon", "foothold", "lateral", "impact"]
    for r in results:
        print(f"\n  {r['label']}")
        print(f"    mean_return : {r['mean_return']:.3f}")
        print(f"    mean_steps  : {r['mean_steps']:.1f}")
        print("    final stages: " + "  ".join(
            f"s{s}({STAGE_NAMES[s]})={r['stage_dist'].get(s,0)}"
            for s in range(5)))


if __name__ == "__main__":
    main()
