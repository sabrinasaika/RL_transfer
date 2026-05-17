"""
Evaluate CW policies zero-shot on CBS. Supports three conditions for ablation:

  Condition 1 — Random baseline:
    Random policy on CBS (lower bound).

  Condition 2 — No domain adaptation (8D raw transfer):
    CW policy trained on raw 8D unified obs (ObservationTranslator only, no encoder).
    Evaluated on CBS using the same 8D translator via UnifiedSecEnv.
    Both domains share identical hand-crafted feature semantics; no learned alignment.
    Train with:  python train_cw_dapn_policy.py --no-dapn --out artifacts/policies/cw_raw_policy.zip

  Condition 3 — DAPN transfer (256D):
    CW policy trained on 256D DAPN-encoded obs (domain-adversarial encoder).
    Evaluated on CBS via the same frozen encoder.
    Train with:  python train_cw_dapn_policy.py --encoder <path> --out artifacts/policies/cw_dapn_policy.zip

Usage:
  # All three conditions:
  PYTHONPATH=$PWD:$PWD/cyberwheel:$PWD/CyberBattleSim python3.10 eval/eval_cw_dapn_on_cbs.py \
    --dapn-policy artifacts/policies/cw_dapn_policy.zip \
    --encoder     artifacts/transfer_models/dapn_encoder.pt \
    --raw-policy  artifacts/policies/cw_raw_policy.zip \
    --episodes 20

  # Skip random baseline:
  ... --skip-random
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

    # Detect whether policy expects dict obs (MultiInputPolicy) or flat array
    use_dict_obs = (policy is not None and
                    hasattr(policy, "observation_space") and
                    hasattr(policy.observation_space, "spaces"))

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep)

        # For MultiInputPolicy pass full dict; for random/MlpPolicy pass flat array
        def _obs_for_policy(o):
            if use_dict_obs:
                # Ensure all dict values are numpy arrays
                if isinstance(o, dict):
                    return {k: np.asarray(v, dtype=np.float32) for k, v in o.items()}
                return o
            return o["obs"] if isinstance(o, dict) and "obs" in o else o

        obs_in = _obs_for_policy(obs)

        ret, steps = 0.0, 0
        done = truncated = False
        while not (done or truncated) and steps < max_steps:
            if policy is None:
                action = env.action_space.sample()
            else:
                action, _ = policy.predict(obs_in, deterministic=deterministic)
                action = int(np.asarray(action).squeeze())

            obs, r, done, truncated, info = env.step(action)
            obs_in = _obs_for_policy(obs)
            ret += float(r)
            steps += 1

        # Kill-chain stage at episode end.
        #
        # CBS stage 4 = attacker achieved the game objective (own_atleast_percent met).
        # This is signalled by terminated=True (not truncated).  CyberBattleChain has no
        # active defender, so terminated=True always means the attacker won.
        #
        # For stages 0-3 we fall back to obs-based stage_from_cbs.
        if done and not truncated:
            # Game ended because the attacker reached the CBS win condition → stage 4
            stage = 4
        else:
            # Timeout / max-steps — read stage from last raw obs
            raw = None
            inner = env
            while inner is not None:
                if hasattr(inner, "_last_raw_obs"):
                    raw = inner._last_raw_obs
                    break
                inner = getattr(inner, "env", None)
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

def _make_cbs_raw_env():
    """
    CBS env with the plain 8D ObservationTranslator wrapped in a Dict format
    matching the raw CW policy's observation space: {"obs": 8D, "mask": 7D}.
    No DAPN encoder — just the hand-crafted unified feature vector.
    """
    import gymnasium as gym
    from gymnasium import spaces
    from adapters.observation_translator import OBS_DIM
    from adapters.action_translator import ActionTranslator

    class RawDictWrapper(gym.Wrapper):
        def __init__(self, env):
            super().__init__(env)
            n_act = len(ActionTranslator().unified_actions)
            self.observation_space = spaces.Dict({
                "obs":  spaces.Box(low=0.0, high=1.0,
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

    base = UnifiedSecEnv("cbs", cbs_factory=make_cbs_env)
    return RawDictWrapper(base)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dapn-policy",
                        default="artifacts/policies/cw_dapn_policy.zip",
                        help="CW policy trained on 256D DAPN features (condition 3)")
    parser.add_argument("--encoder",
                        default="artifacts/transfer_models/dapn_encoder.pt",
                        help="DAPN encoder checkpoint (used with --dapn-policy)")
    parser.add_argument("--raw-policy", default=None,
                        help="CW policy trained on raw 8D obs, no encoder (condition 2). "
                             "Train with: python train_cw_dapn_policy.py --no-dapn")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--feature-size", type=int, default=256)
    parser.add_argument("--deterministic", action="store_true",
                        help="Use deterministic policy (default: stochastic)")
    parser.add_argument("--skip-random", action="store_true",
                        help="Skip random baseline (saves time)")
    parser.add_argument("--save-json", default=None,
                        help="Save all results to this JSON file for later plotting. "
                             "Example: --save-json results/eval_results.json")
    args = parser.parse_args()

    results = []

    # ── Condition 1: Random baseline ─────────────────────────────────────────
    if not args.skip_random:
        print("\n" + "="*60)
        print("CONDITION 1: Random policy on CBS")
        print("="*60)
        env_rand = UnifiedSecEnv("cbs", cbs_factory=make_cbs_env)
        r = rollout(env_rand, policy=None,
                    n_episodes=args.episodes, max_steps=args.max_steps,
                    label="random")
        results.append(r)
        env_rand.close()

    # ── Condition 2: Raw 8D policy — no domain adaptation ────────────────────
    raw_policy_path = args.raw_policy
    if raw_policy_path is not None:
        if not raw_policy_path.endswith(".zip"):
            raw_policy_path += ".zip"
        if not os.path.exists(raw_policy_path):
            print(f"\n⚠  Raw policy not found at {raw_policy_path}")
            print("   Train with: python train_cw_dapn_policy.py --no-dapn "
                  "--out artifacts/policies/cw_raw_policy.zip")
        else:
            from stable_baselines3 import PPO
            print("\n" + "="*60)
            print("CONDITION 2: CW raw-8D policy → zero-shot on CBS (no domain adaptation)")
            print("="*60)
            cw_raw_policy = PPO.load(raw_policy_path)
            env_raw = _make_cbs_raw_env()
            r = rollout(env_raw, policy=cw_raw_policy,
                        n_episodes=args.episodes, max_steps=args.max_steps,
                        deterministic=args.deterministic,
                        label="cw_raw→cbs")
            results.append(r)
            env_raw.close()
    else:
        print("\n(Condition 2 skipped — pass --raw-policy to include the no-DAPN ablation)")

    # ── Condition 3: DAPN policy — with domain adaptation ────────────────────
    dapn_policy_path = args.dapn_policy
    if not dapn_policy_path.endswith(".zip"):
        dapn_policy_path += ".zip"

    if not os.path.exists(dapn_policy_path):
        print(f"\n⚠  DAPN policy not found at {dapn_policy_path}")
        print("   Train with: python train_cw_dapn_policy.py "
              "--encoder <path> --out artifacts/policies/cw_dapn_policy.zip")
    else:
        from stable_baselines3 import PPO
        print("\n" + "="*60)
        print("CONDITION 3: CW DAPN policy (256D) → zero-shot on CBS (with domain adaptation)")
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
                    deterministic=args.deterministic,
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

    # Delta table (if both ablation conditions ran)
    raw_r   = next((r for r in results if r["label"] == "cw_raw→cbs"),  None)
    dapn_r  = next((r for r in results if r["label"] == "cw_dapn→cbs"), None)
    rand_r  = next((r for r in results if r["label"] == "random"),       None)
    if raw_r and dapn_r:
        print("\n" + "="*60)
        print("DAPN vs NO-DAPN DELTA")
        print("="*60)
        delta_ret   = dapn_r["mean_return"] - raw_r["mean_return"]
        delta_steps = dapn_r["mean_steps"]  - raw_r["mean_steps"]
        dapn_s4  = dapn_r["stage_dist"].get(4, 0)
        raw_s4   = raw_r["stage_dist"].get(4, 0)
        print(f"  mean_return : {raw_r['mean_return']:+.3f} (raw)  →  "
              f"{dapn_r['mean_return']:+.3f} (DAPN)   Δ={delta_ret:+.3f}")
        print(f"  mean_steps  : {raw_r['mean_steps']:.1f} (raw)  →  "
              f"{dapn_r['mean_steps']:.1f} (DAPN)   Δ={delta_steps:+.1f}")
        print(f"  stage-4 eps : {raw_s4} (raw)  →  {dapn_s4} (DAPN)  "
              f"out of {args.episodes}")
        if rand_r:
            rand_baseline = rand_r["mean_return"]
            if abs(raw_r["mean_return"] - rand_baseline) > 1e-6:
                gain_raw  = (raw_r["mean_return"]  - rand_baseline)
                gain_dapn = (dapn_r["mean_return"] - rand_baseline)
                print(f"\n  Gain over random:  raw={gain_raw:+.3f}   DAPN={gain_dapn:+.3f}")

    # ── Save JSON for plotting ────────────────────────────────────────────────
    if args.save_json:
        import json
        os.makedirs(os.path.dirname(args.save_json) if os.path.dirname(args.save_json) else ".", exist_ok=True)
        with open(args.save_json, "w") as f:
            json.dump({"n_episodes": args.episodes, "conditions": results}, f, indent=2)
        print(f"\nResults saved to {args.save_json}")
        print(f"  Plot with: python3 plot_results.py --results {args.save_json}")


if __name__ == "__main__":
    main()
