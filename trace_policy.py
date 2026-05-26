"""
Trace the trained policy step-by-step on CyberBattleSim.

Shows every step: which slot the policy targeted, what CBS action was
executed, which nodes are now owned, and the cumulative reward.

Usage:
  PYTHONPATH=$PWD:$PWD/cyberwheel:$PWD/CyberBattleSim python3.10 trace_policy.py
  PYTHONPATH=$PWD:$PWD/cyberwheel:$PWD/CyberBattleSim python3.10 trace_policy.py \\
      --policy   artifacts/policies/best_kc_raw_12slot/best_kc_raw/best_model.zip \\
      --encoder  artifacts/transfer_models/dapn_encoder_phase_aware.pt \\
      --episodes 3 \\
      --max-steps 200 \\
      --win-nodes 10
"""

import os, sys, argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "CyberBattleSim"))

import numpy as np
from stable_baselines3 import PPO
from gymnasium.wrappers import TimeLimit

from adapters.unified_env import UnifiedSecEnv, MAX_SLOTS
from adapters.kc_dapn_translate_wrapper import KCDAPNTranslateWrapper
from config.env_builders import make_cbs_env

# Kill-chain phase names (0-6)
PHASE_NAMES = {
    0: "ping_sweep",
    1: "port_scan",
    2: "svc_disc",
    3: "exploit",
    4: "owned",
    5: "escalated",
    6: "impacted",
}

def phase_name(p):
    return PHASE_NAMES.get(int(p), f"phase{p}")


def _fmt_cbs_action(action: dict) -> str:
    """Format a CBS backend action dict into a readable string.

    Examples:
      {"local_vulnerability":  (3, 1)}        → "local_vuln  (node=3, vuln=1)"
      {"remote_vulnerability": (0, 4, 2)}      → "remote_vuln (src=0, tgt=4, vuln=2)"
      {"connect":              (2, 3, 0, 1)}   → "connect     (src=2 → tgt=3, port=0, cred=1)"
    """
    if not isinstance(action, dict) or not action:
        return "(no action)"
    key, val = next(iter(action.items()))
    if key == "local_vulnerability":
        node, vuln = val
        return f"local_vuln  (node={node}, vuln_idx={vuln})"
    if key == "remote_vulnerability":
        src, tgt, vuln = val
        return f"remote_vuln (src={src} → tgt={tgt}, vuln_idx={vuln})"
    if key == "connect":
        src, tgt, port, cred = val
        return f"connect     (src={src} → tgt={tgt}, port={port}, cred={cred})"
    return f"{key}{val}"


def _get_unified_env(env):
    """Unwrap layers to find the UnifiedSecEnv."""
    e = env
    while e is not None:
        if isinstance(e, UnifiedSecEnv):
            return e
        e = getattr(e, "env", None)
    return None


def _priv_levels(unified):
    """Return current privilege levels array from the raw CBS obs."""
    raw = getattr(unified, "_raw_obs", None)
    if not isinstance(raw, dict):
        return None
    priv = raw.get("nodes_privilegelevel")
    if priv is None:
        return None
    return np.asarray(priv, dtype=np.int32)


def _owned_nodes(unified, cbs_size):
    """List of target nodes that are currently owned (priv >= 1)."""
    priv = _priv_levels(unified)
    if priv is None:
        return []
    total = cbs_size + 2          # chain creates size+2 nodes
    return [i for i in range(1, total) if i < priv.size and priv[i] >= 1]


def _bar(owned, total, width=20):
    """Simple text progress bar."""
    filled = int(width * owned / max(total, 1))
    return f"[{'█' * filled}{'░' * (width - filled)}] {owned}/{total}"


def run_trace(env, policy, episode, max_steps, win_nodes, cbs_size, deterministic):
    unified = _get_unified_env(env)
    obs, _ = env.reset(seed=episode)

    total_r = 0.0
    won = False

    print(f"\n{'═'*70}")
    print(f"  Episode {episode + 1}   |  win target: {win_nodes} nodes  |  max steps: {max_steps}")
    print(f"{'═'*70}")
    print(f"  {'Step':>4}  {'Slot':>4}  {'Phase':<12}  {'CBS Action':<38}  {'Owned':>5}  {'Reward':>8}")
    print(f"  {'----':>4}  {'----':>4}  {'------------':<12}  {'-------------------------------':<38}  {'-----':>5}  {'--------':>8}")

    prev_owned = set()

    for step in range(max_steps):
        # Policy decision
        action, _ = policy.predict(obs, deterministic=deterministic)
        slot = int(action)

        # Read sim phase BEFORE step
        sim_phases = {}
        if unified is not None:
            sim_phases = dict(getattr(unified, "_slot_cw_phases", {}))

        phase = sim_phases.get(slot, 0)
        phase_str = phase_name(phase)

        # Step
        obs, r, terminated, truncated, info = env.step(slot)
        total_r += r

        # Format the actual CBS action that was sent to the environment
        cbs_action_str = _fmt_cbs_action(info.get("cbs_action", {}))

        # Current owned nodes
        owned_now = set(_owned_nodes(unified, cbs_size)) if unified else set()
        newly_owned = owned_now - prev_owned

        # Format owned bar
        owned_bar = _bar(len(owned_now), cbs_size + 1)

        # Mark newly owned nodes
        new_tag = ""
        if newly_owned:
            new_tag = f"  ◀ node(s) {sorted(newly_owned)} OWNED"

        # Truncate CBS action string for display
        disp_action = cbs_action_str[:36] if len(cbs_action_str) > 36 else cbs_action_str

        print(f"  {step+1:>4}  {slot:>4}  {phase_str:<12}  {disp_action:<38}  "
              f"{len(owned_now):>5}  {r:>8.1f}{new_tag}")

        prev_owned = owned_now

        if terminated:
            won = True
            break
        if truncated:
            break

    status = "✓ WIN" if won else "✗ timeout"
    print(f"\n  {status}  |  total reward: {total_r:.1f}  |  steps: {step+1}"
          f"  |  nodes owned: {_bar(len(prev_owned), cbs_size + 1)}")

    return won, total_r, step + 1, len(prev_owned)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy",     default="artifacts/policies/best_kc_raw_12slot/best_kc_raw/best_model.zip")
    parser.add_argument("--encoder",    default="artifacts/transfer_models/dapn_encoder_phase_aware.pt")
    parser.add_argument("--episodes",   type=int,   default=3)
    parser.add_argument("--max-steps",  type=int,   default=200)
    parser.add_argument("--cbs-size",   type=int,   default=12)
    parser.add_argument("--win-nodes",  type=int,   default=8)
    parser.add_argument("--no-dapn",    action="store_true",
                        help="Trace raw policy without DAPN (shows what happens without translation)")
    parser.add_argument("--random",     action="store_true",
                        help="Trace a random policy (baseline)")
    parser.add_argument("--deterministic", action="store_true", default=False,
                        help="Use deterministic policy (default: stochastic)")
    args = parser.parse_args()

    os.environ["CBS_SIZE"] = str(args.cbs_size)
    if args.win_nodes > 0:
        os.environ["CBS_WIN_NODES"] = str(args.win_nodes)
    else:
        os.environ.pop("CBS_WIN_NODES", None)

    total_nodes = args.cbs_size + 2
    print(f"\nCBS chain:   size={args.cbs_size}  total_nodes={total_nodes}"
          f"  target_nodes={total_nodes - 1}")
    print(f"Win target:  {args.win_nodes} chain nodes owned (= {args.win_nodes + 1} total)")

    # Build env
    def _make_base():
        base = UnifiedSecEnv("cbs", cbs_factory=make_cbs_env)
        return TimeLimit(base, max_episode_steps=args.max_steps)

    if args.random:
        print(f"Mode:        RANDOM baseline\n")
        env = _make_base()
        policy = None
        label = "Random"
    elif args.no_dapn:
        print(f"Mode:        No-DAPN  (raw CW policy on raw CBS obs)\n")
        if not Path(args.policy).exists():
            print(f"[error] policy not found: {args.policy}")
            sys.exit(1)
        env = _make_base()
        policy = PPO.load(args.policy)
        label = "No-DAPN"
    else:
        print(f"Mode:        DAPN  (CW policy + phase-aware encoder)\n")
        if not Path(args.policy).exists():
            print(f"[error] policy not found: {args.policy}")
            sys.exit(1)
        if not Path(args.encoder).exists():
            print(f"[error] encoder not found: {args.encoder}")
            sys.exit(1)
        base = _make_base()
        env = KCDAPNTranslateWrapper(base, encoder_path=args.encoder, device="cpu")
        policy = PPO.load(args.policy)
        label = "DAPN"

    wins, returns, steps_list, nodes_list = [], [], [], []

    for ep in range(args.episodes):
        if args.random:
            # Random trace — manually step
            unified = _get_unified_env(env)
            obs, _ = env.reset(seed=ep)
            total_r, won = 0.0, False
            prev_owned = set()

            print(f"\n{'═'*70}")
            print(f"  Episode {ep+1} [Random]  |  win target: {args.win_nodes} nodes")
            print(f"{'═'*70}")
            print(f"  {'Step':>4}  {'Slot':>4}  {'CBS Action':<40}  {'Owned':>5}  {'Reward':>8}")
            print(f"  {'----':>4}  {'----':>4}  {'----------------------------------------':<40}  {'-----':>5}  {'--------':>8}")

            for step in range(args.max_steps):
                action = env.action_space.sample()
                obs, r, terminated, truncated, info = env.step(action)
                total_r += r
                owned_now = set(_owned_nodes(unified, args.cbs_size)) if unified else set()
                newly_owned = owned_now - prev_owned
                new_tag = f"  ◀ node(s) {sorted(newly_owned)} OWNED" if newly_owned else ""
                cbs_act = _fmt_cbs_action(info.get("cbs_action", {}))
                print(f"  {step+1:>4}  {action:>4}  {cbs_act:<40}  "
                      f"{len(owned_now):>5}  {r:>8.1f}{new_tag}")
                prev_owned = owned_now
                if terminated:
                    won = True
                    break
                if truncated:
                    break

            status = "✓ WIN" if won else "✗ timeout"
            print(f"\n  {status}  |  total reward: {total_r:.1f}  |  steps: {step+1}"
                  f"  |  nodes owned: {_bar(len(prev_owned), args.cbs_size + 1)}")
            wins.append(int(won)); returns.append(total_r)
            steps_list.append(step + 1); nodes_list.append(len(prev_owned))
        else:
            w, r, s, n = run_trace(env, policy, ep, args.max_steps,
                                   args.win_nodes, args.cbs_size,
                                   deterministic=args.deterministic)
            wins.append(int(w)); returns.append(r)
            steps_list.append(s); nodes_list.append(n)

    env.close()

    print(f"\n{'═'*70}")
    print(f"  SUMMARY  [{label}]  ({args.episodes} episodes)")
    print(f"{'═'*70}")
    print(f"  Win rate      : {np.mean(wins)*100:.0f}%  ({sum(wins)}/{args.episodes})")
    print(f"  Mean return   : {np.mean(returns):.1f} ± {np.std(returns):.1f}")
    print(f"  Mean steps    : {np.mean(steps_list):.1f}")
    print(f"  Nodes owned   : {np.mean(nodes_list):.1f} / {args.cbs_size + 1}")


if __name__ == "__main__":
    main()
