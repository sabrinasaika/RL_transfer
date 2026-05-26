"""
Evaluate CW-trained kill-chain policies zero-shot on CyberBattleSim.

Four conditions:
  1. true_random   — random valid action sampled directly from raw CBS action mask
                     (real lower bound — no kill-chain translation)
  2. random        — random slot chosen, action resolved via kill-chain translation
                     (shows translation layer value without a learned policy)
  3. cw_kc_raw     — CW policy trained on 512-D obs, no DAPN adaptation
  4. dapn          — CW policy + phase-aware DAPN obs-translation (stochastic)

Usage:
  PYTHONPATH=$PWD:$PWD/cyberwheel:$PWD/CyberBattleSim python3.10 eval_kc_transfer.py
  PYTHONPATH=$PWD:$PWD/cyberwheel:$PWD/CyberBattleSim python3.10 eval_kc_transfer.py \\
      --raw-policy  artifacts/policies/best_kc_raw_12slot/best_kc_raw/best_model.zip \\
      --encoder     artifacts/transfer_models/dapn_encoder_phase_aware.pt \\
      --episodes 20 --win-nodes 8
"""

import os, sys, argparse, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "CyberBattleSim"))

import numpy as np
from stable_baselines3 import PPO
from gymnasium.wrappers import TimeLimit

from adapters.unified_env import UnifiedSecEnv
from adapters.kc_dapn_translate_wrapper import KCDAPNTranslateWrapper
from adapters.kill_chain import stage_from_cbs
from config.env_builders import make_cbs_env

# Cap episodes to prevent CBS credential cache from growing unbounded
MAX_EPISODE_STEPS = 500


def make_base_cbs_env():
    base = UnifiedSecEnv("cbs", cbs_factory=make_cbs_env)
    return TimeLimit(base, max_episode_steps=MAX_EPISODE_STEPS)


def make_true_random_cbs_env():
    """Raw CBS env without UnifiedSecEnv — action space is the real CBS action space."""
    raw = make_cbs_env()
    return TimeLimit(raw, max_episode_steps=MAX_EPISODE_STEPS)


def make_dapn_cbs_env(encoder_path):
    base = UnifiedSecEnv("cbs", cbs_factory=make_cbs_env)
    base = TimeLimit(base, max_episode_steps=MAX_EPISODE_STEPS)
    return KCDAPNTranslateWrapper(base, encoder_path=encoder_path, device="cpu")


def _get_raw_obs(env):
    """Unwrap env layers to find the UnifiedSecEnv with _raw_obs."""
    e = env
    while e is not None:
        raw = getattr(e, "_raw_obs", None)
        if raw is not None:
            return raw
        e = getattr(e, "env", None)
    return None


def _nodes_owned(raw_obs) -> int:
    """Count target nodes with priv >= 1 (excludes node 0 = attacker start)."""
    if not isinstance(raw_obs, dict):
        return 0
    priv = raw_obs.get("nodes_privilegelevel")
    if priv is None:
        return 0
    priv = np.asarray(priv, dtype=np.int32)
    priv_targets = priv[1:] if priv.size > 1 else np.array([], dtype=np.int32)
    return int((priv_targets >= 1).sum())


def _sample_valid_cbs_action(env):
    """Sample a random valid action from the CBS action mask.

    CBS throws on invalid actions by default.  We use compute_action_mask()
    to restrict sampling to only valid (src, tgt, port, cred / node, vuln) combos.
    Falls back to a safe local_vulnerability(0,0) if the mask is unavailable.
    """
    # Unwrap to find the CBS env that has compute_action_mask
    inner = env
    while inner is not None:
        if hasattr(inner, "compute_action_mask"):
            break
        inner = getattr(inner, "env", None)

    if inner is None:
        return {"local_vulnerability": (0, 0)}

    try:
        am = inner.compute_action_mask()
    except Exception:
        return {"local_vulnerability": (0, 0)}

    # Collect all valid actions across all action types
    valid = []
    for key, length in [("connect", 4), ("local_vulnerability", 2), ("remote_vulnerability", 3)]:
        arr = am.get(key)
        if arr is None:
            continue
        indices = np.argwhere(np.asarray(arr))
        for idx in indices.tolist():
            valid.append({key: tuple(int(x) for x in idx[:length])})

    if not valid:
        return {"local_vulnerability": (0, 0)}

    return valid[np.random.randint(len(valid))]


def rollout_true_random(env, n_episodes, max_steps=200, label=""):
    """True random baseline: sample uniformly from valid raw CBS actions.

    Unlike the 'random' condition which picks a random SLOT and then uses the
    kill-chain translation layer, this samples directly from the CBS action mask
    with no kill-chain logic — a genuine lower bound.
    """
    returns, steps_list, final_stages = [], [], []
    nodes_owned_list, win_list, steps_to_first_owned_list = [], [], []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep)
        total_r, n_steps = 0.0, 0
        won = False
        first_owned_step = None

        for step in range(max_steps):
            action = _sample_valid_cbs_action(env)
            try:
                obs, r, terminated, truncated, _ = env.step(action)
            except Exception:
                # Invalid action slipped through — treat as no-op
                obs, r, terminated, truncated = obs, 0.0, False, False
            total_r += r
            n_steps += 1

            if first_owned_step is None and _nodes_owned(obs) > 0:
                first_owned_step = n_steps

            if terminated:
                won = True
                break
            if truncated:
                break

        final_stage = stage_from_cbs(obs) if isinstance(obs, dict) else 0
        n_owned = _nodes_owned(obs)

        returns.append(total_r)
        steps_list.append(n_steps)
        final_stages.append(final_stage)
        nodes_owned_list.append(n_owned)
        win_list.append(int(won))
        steps_to_first_owned_list.append(first_owned_step if first_owned_step else max_steps)

    mean_r     = float(np.mean(returns))
    std_r      = float(np.std(returns))
    mean_st    = float(np.mean(steps_list))
    mean_owned = float(np.mean(nodes_owned_list))
    win_rate   = float(np.mean(win_list))
    mean_first = float(np.mean(steps_to_first_owned_list))
    stage_dist = {int(s): int(final_stages.count(s)) for s in sorted(set(final_stages))}

    print(f"\n{label}")
    print(f"  mean return       : {mean_r:.2f} ± {std_r:.2f}")
    print(f"  win rate          : {win_rate*100:.0f}%  ({sum(win_list)}/{n_episodes} episodes)")
    print(f"  nodes owned (avg) : {mean_owned:.2f} / {_total_target_nodes(env)}")
    print(f"  steps to 1st own  : {mean_first:.1f}")
    print(f"  mean steps        : {mean_st:.1f}")
    print(f"  final stages      : {stage_dist}")
    return {
        "returns": returns, "steps": steps_list, "final_stages": final_stages,
        "nodes_owned": nodes_owned_list, "wins": win_list,
        "steps_to_first_owned": steps_to_first_owned_list,
        "mean_return": mean_r, "std_return": std_r,
        "win_rate": win_rate, "mean_nodes_owned": mean_owned,
        "mean_steps_to_first_owned": mean_first,
        "stage_dist": stage_dist,
    }


def rollout(env, policy, n_episodes, max_steps=200, deterministic=True, label=""):
    returns, steps_list, final_stages = [], [], []
    nodes_owned_list, win_list, steps_to_first_owned_list = [], [], []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep)
        total_r, n_steps = 0.0, 0
        won = False
        first_owned_step = None

        for step in range(max_steps):
            if policy == "random":
                action = env.action_space.sample()
            else:
                action, _ = policy.predict(obs, deterministic=deterministic)
                action = int(action)
            obs, r, terminated, truncated, _ = env.step(action)
            total_r += r
            n_steps += 1

            # Track first node owned
            if first_owned_step is None:
                raw_mid = _get_raw_obs(env)
                if _nodes_owned(raw_mid) > 0:
                    first_owned_step = n_steps

            # True win = CBS objective met (not just time limit)
            if terminated:
                won = True
                break
            if truncated:
                break

        raw = _get_raw_obs(env)
        final_stage  = stage_from_cbs(raw) if isinstance(raw, dict) else 0
        n_owned      = _nodes_owned(raw)

        returns.append(total_r)
        steps_list.append(n_steps)
        final_stages.append(final_stage)
        nodes_owned_list.append(n_owned)
        win_list.append(int(won))
        steps_to_first_owned_list.append(first_owned_step if first_owned_step else max_steps)

    mean_r      = float(np.mean(returns))
    std_r       = float(np.std(returns))
    mean_st     = float(np.mean(steps_list))
    mean_owned  = float(np.mean(nodes_owned_list))
    win_rate    = float(np.mean(win_list))
    mean_first  = float(np.mean(steps_to_first_owned_list))
    stage_dist  = {int(s): int(final_stages.count(s)) for s in sorted(set(final_stages))}

    print(f"\n{label}")
    print(f"  mean return       : {mean_r:.2f} ± {std_r:.2f}")
    print(f"  win rate          : {win_rate*100:.0f}%  ({sum(win_list)}/{n_episodes} episodes)")
    print(f"  nodes owned (avg) : {mean_owned:.2f} / {_total_target_nodes(env)}")
    print(f"  steps to 1st own  : {mean_first:.1f}")
    print(f"  mean steps        : {mean_st:.1f}")
    print(f"  final stages      : {stage_dist}")
    return {
        "returns": returns, "steps": steps_list, "final_stages": final_stages,
        "nodes_owned": nodes_owned_list, "wins": win_list,
        "steps_to_first_owned": steps_to_first_owned_list,
        "mean_return": mean_r, "std_return": std_r,
        "win_rate": win_rate, "mean_nodes_owned": mean_owned,
        "mean_steps_to_first_owned": mean_first,
        "stage_dist": stage_dist,
    }


def _total_target_nodes(env) -> int:
    """Number of target nodes (all nodes except start node 0).

    CyberBattleChain(size=N) creates N+2 nodes:
      node 0         — attacker entry (pre-owned, not a target)
      nodes 1..N     — chain nodes  (alternating Linux/Windows pairs)
      node N+1       — flag/final node
    So there are N+1 target nodes in total.

    CBS observations pad nodes_privilegelevel to 100 elements regardless of
    chain size, so we use CBS_SIZE (set via CLI) rather than the obs length.
    """
    cbs_size = int(os.environ.get("CBS_SIZE", "6"))
    return cbs_size + 1  # chain creates size+2 nodes total; all except node 0 are targets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-policy",
                        default="artifacts/policies/best_kc_raw/best_model.zip")
    parser.add_argument("--encoder",
                        default="artifacts/transfer_models/dapn_encoder_v2.pt")
    parser.add_argument("--episodes",  type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--cbs-size",  type=int, default=12,
                        help="CBS chain length (harder = higher, default 12)")
    parser.add_argument("--win-nodes", type=int, default=8,
                        help="Number of chain nodes the attacker must own to win (default 8)."
                             " Set 0 to use the CBS_GOAL_OWN_PCT percentage threshold.")
    parser.add_argument("--out",       default="results/kc_eval.json")
    args = parser.parse_args()

    # Set CBS chain size and win condition via env vars
    # (read by make_cbs_env in config/env_builders.py)
    os.environ["CBS_SIZE"] = str(args.cbs_size)
    if args.win_nodes > 0:
        os.environ["CBS_WIN_NODES"] = str(args.win_nodes)
        print(f"\nCBS chain size : {args.cbs_size}  (total nodes: {args.cbs_size + 2}, "
              f"need {args.win_nodes} chain nodes owned to win)")
    else:
        os.environ.pop("CBS_WIN_NODES", None)
        print(f"\nCBS chain size : {args.cbs_size}  (need {int(args.cbs_size * 0.5)} nodes owned to win)")

    results = {}

    # ── Condition 1: True random (raw CBS action mask) ────────────────────────
    print("\nRunning condition 1/4 — True random (raw CBS actions)...")
    env_true_rand = make_true_random_cbs_env()
    results["true_random"] = rollout_true_random(
        env_true_rand, args.episodes, args.max_steps,
        label="Condition 1 — True Random (raw CBS)"
    )
    env_true_rand.close()

    # ── Condition 2: Kill-chain random (random slot + translation) ────────────
    print("\nRunning condition 2/4 — KC-Random (random slot, kill-chain translation)...")
    env_rand = make_base_cbs_env()
    results["random"] = rollout(env_rand, "random", args.episodes, args.max_steps,
                                label="Condition 2 — KC-Random (slot random, action translated)")
    env_rand.close()

    # ── Condition 3: Raw CW policy, no adaptation ─────────────────────────────
    print("\nRunning condition 3/4 — No DAPN...")
    if Path(args.raw_policy).exists():
        policy_raw = PPO.load(args.raw_policy)
        env_raw = make_base_cbs_env()
        results["cw_kc_raw"] = rollout(
            env_raw, policy_raw, args.episodes, args.max_steps, deterministic=True,
            label="Condition 3 — No DAPN"
        )
        env_raw.close()
    else:
        print(f"  [skip] not found: {args.raw_policy}")

    # ── Condition 4: DAPN (stochastic) ───────────────────────────────────────
    print("\nRunning condition 4/4 — DAPN...")
    if Path(args.raw_policy).exists() and Path(args.encoder).exists():
        policy_dapn = PPO.load(args.raw_policy)
        env_dapn = make_dapn_cbs_env(args.encoder)
        results["dapn"] = rollout(
            env_dapn, policy_dapn, args.episodes, args.max_steps, deterministic=False,
            label="Condition 4 — DAPN"
        )
        env_dapn.close()
    else:
        print(f"  [skip] missing policy or encoder")

    # ── Save ─────────────────────────────────────────────────────────────────
    os.makedirs(Path(args.out).parent, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {args.out}")

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n── Summary ─────────────────────────────────────────────────────────────────")
    print(f"  {'Condition':<20}  {'WinRate':>7}  {'NodesOwned':>10}  {'StepsTo1st':>10}  {'MeanReturn':>10}")
    print(f"  {'-'*20}  {'-'*7}  {'-'*10}  {'-'*10}  {'-'*10}")
    conditions = [
        ("true_random", "True Random         "),
        ("random",      "KC-Random           "),
        ("cw_kc_raw",   "No DAPN             "),
        ("dapn",        "DAPN                "),
    ]
    for key, label in conditions:
        if key in results:
            r   = results[key]
            wr  = r.get("win_rate", 0) * 100
            no  = r.get("mean_nodes_owned", 0)
            sf  = r.get("mean_steps_to_first_owned", 0)
            mr  = r["mean_return"]
            print(f"  {label}  {wr:6.0f}%  {no:10.2f}  {sf:10.1f}  {mr:10.2f}")


if __name__ == "__main__":
    main()
