#!/usr/bin/env python3
"""
Collect raw observations from CyberWheel (CW) and CyberBattleSim (CBS)
and save them to a .npz file for later DAPN training.

Usage:
    python collect_observations.py --num-samples 1000 --out data/obs.npz

    # CBS only (if CyberWheel not available):
    python collect_observations.py --num-samples 1000 --out data/obs.npz --cbs-only

    # Use a heuristic policy on BOTH CW and CBS to force all kill-chain
    # stages (0-4) including "impact" — strongly recommended for DAPN training:
    python collect_observations.py --num-samples 1000 --out data/obs.npz --directed

    # Use trained PPO on CBS instead of random/heuristic:
    python collect_observations.py --num-samples 1000 --out data/obs.npz \\
      --cbs-agent artifacts/cbs_agent.zip
"""

import os
import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "cyberwheel"))

import argparse
import gc
import numpy as np
from tqdm import tqdm

from adapters.unified_env import UnifiedSecEnv
from config.env_builders import make_cbs_env, make_cw_env
from train_dapn_encoder import _collect_full_episodes
from adapters.kill_chain import stage_from_cbs, stage_from_cw, KILL_CHAIN_STAGES
from collections import Counter


# Keys to keep from CBS obs.
# Drops: action_mask (huge 3D/4D arrays → OOM), _discovered_nodes, _explored_network (graph objects).
# Keeps: credential_cache_matrix — shape (1000,2), ~16KB per obs, fine to save.
_CBS_KEEP_KEYS = {
    "newly_discovered_nodes_count",
    "leaked_credentials",
    "lateral_move",
    "customer_data_found",
    "escalation",
    "probe_result",
    "credential_cache_length",
    "credential_cache_matrix",   # (1000,2) node+port pairs — needed by preprocessor
    "discovered_node_count",
    "discovered_nodes_properties",
    "nodes_privilegelevel",
}


def _strip_cbs_obs(obs: dict) -> dict:
    """Remove large action_mask arrays from CBS obs before saving."""
    return {k: v for k, v in obs.items() if k in _CBS_KEEP_KEYS}


def _pick_cbs_action(obs: dict, rng: np.random.Generator) -> dict:
    """
    Heuristic action selector operating directly on raw CBS obs + action mask.

    Actual CBS kill-chain mechanics (confirmed empirically):
      - local_vulnerability on owned node  → escalates priv, discovers neighbors
      - remote_vulnerability(src, tgt, *)  → compromises tgt (requires src != tgt)
        Note: remote_vulnerability(0, 0, *) is self-targeting and does nothing
      - connect(src, tgt, ...)             → lateral move using cached credentials

    Priority:
      Stage 0    → local_vulnerability (discovers adjacent nodes + leaks creds)
      Stage 1    → connect (if creds cached), else remote_vulnerability to probe
      Stage 2+   → connect (if fresh creds), then local_vulnerability on owned nodes
                   (random sampling eventually hits the credential-leaking vuln),
                   then remote_vulnerability as fallback probe
    """
    stage = stage_from_cbs(obs)
    mask = obs.get("action_mask", {})

    def _sample_lv():
        """Sample a valid local_vulnerability action."""
        m = mask.get("local_vulnerability")
        if m is None:
            return None
        indices = np.argwhere(np.asarray(m) == 1)
        if len(indices) == 0:
            return None
        return tuple(int(x) for x in indices[rng.integers(len(indices))])

    def _sample_rv_cross():
        """Sample a valid remote_vulnerability where source node != target node."""
        m = mask.get("remote_vulnerability")
        if m is None:
            return None
        arr = np.asarray(m)
        # source is dim 0, target is dim 1 — keep only cross-node entries
        cross = arr.copy()
        for i in range(cross.shape[0]):
            cross[i, i, :] = 0   # zero out self-targeting
        indices = np.argwhere(cross == 1)
        if len(indices) == 0:
            return None
        return tuple(int(x) for x in indices[rng.integers(len(indices))])

    def _sample_connect():
        """Sample a valid cross-node connect action (source != target)."""
        m = mask.get("connect")
        if m is None:
            return None
        arr = np.asarray(m)
        cross = arr.copy()
        for i in range(cross.shape[0]):
            cross[i, i, :, :] = 0   # zero out self-targeting
        indices = np.argwhere(cross == 1)
        if len(indices) == 0:
            return None
        return tuple(int(x) for x in indices[rng.integers(len(indices))])

    if stage == 0:
        # No nodes discovered yet — local_vulnerability on start node reveals neighbors
        lv = _sample_lv()
        if lv is not None:
            return {"local_vulnerability": lv}
        rv = _sample_rv_cross()
        if rv is not None:
            return {"remote_vulnerability": rv}

    elif stage == 1:
        # Nodes discovered but not owned.
        # connect (uses cached credentials) is the definitive way to get a foothold.
        # remote_vulnerability leaks credentials needed for connect.
        # Order: connect cross-node first (if credentials cached), else leak creds first.
        cred_len = int(obs.get("credential_cache_length", 0) or 0)
        if cred_len > 0:
            co = _sample_connect()
            if co is not None:
                return {"connect": co}
        rv = _sample_rv_cross()
        if rv is not None:
            return {"remote_vulnerability": rv}
        co = _sample_connect()
        if co is not None:
            return {"connect": co}
        lv = _sample_lv()
        if lv is not None:
            return {"local_vulnerability": lv}

    else:
        # Stage 2+ — CBS chain mechanics (confirmed from chainpattern.py source):
        #   Credentials for the NEXT node are leaked by LOCAL vulnerabilities on owned nodes:
        #     start → ScanExplorerRecentFiles → SSH cred for 1_LinuxNode
        #     1_LinuxNode → CrackKeepPassX    → RDP cred for 2_WindowsNode
        #     2_WindowsNode → CrackKeepPass   → SSH cred for 3_LinuxNode  (etc.)
        #   Remote vulnerabilities on chain nodes are probes only — no credential leaks.
        #
        #   Critical bug fixed: after ScanBashHistory discovers a node without leaking creds,
        #   we must keep calling local_vulnerability (not rv) to eventually hit the
        #   credential-leaking vuln (CrackKeepPassX / CrackKeepPass).
        #
        #   Priority:
        #     1. Fresh cached credentials available → connect to claim the node
        #     2. local_vulnerability on owned nodes → discovers next node OR leaks creds
        #        (random sampling will eventually hit the credential-leaking vulnerability)
        #     3. remote_vulnerability → fallback probe only
        cred_len = int(obs.get("credential_cache_length", 0) or 0)
        comp_now = int((np.asarray(obs.get("nodes_privilegelevel", [])) >= 1).sum()) - 1  # exclude start
        has_fresh_creds = cred_len > max(comp_now, 0)

        if has_fresh_creds:
            co = _sample_connect()
            if co is not None:
                return {"connect": co}

        # local_vulnerability on owned nodes: discovers new nodes OR leaks credentials.
        # Sampling will eventually hit the credential-leaking vuln (CrackKeepPassX etc.).
        lv = _sample_lv()
        if lv is not None:
            return {"local_vulnerability": lv}

        # Fallback: remote probe or speculative connect
        rv = _sample_rv_cross()
        if rv is not None:
            return {"remote_vulnerability": rv}
        co = _sample_connect()
        if co is not None:
            return {"connect": co}

    # Nothing valid — CBS will handle gracefully
    return {"local_vulnerability": (0, 0)}


def _fmt_cbs_action(action: dict) -> str:
    """Format a CBS action dict into a short readable string."""
    if not isinstance(action, dict):
        return str(action)
    (k, v), = action.items()
    return f"{k}({', '.join(str(x) for x in v)})" if isinstance(v, tuple) else f"{k}({v})"


def _cbs_state_snapshot(obs: dict):
    """Extract key CBS state metrics for debug printing."""
    disc  = int(obs.get("discovered_node_count", 0) or 0)
    creds = int(obs.get("credential_cache_length", 0) or 0)
    priv  = obs.get("nodes_privilegelevel", np.array([], dtype=np.int32))
    if not isinstance(priv, np.ndarray):
        priv = np.array(priv, dtype=np.int32) if priv is not None else np.array([], dtype=np.int32)
    owned = int((priv[1:] >= 1).sum()) if priv.size > 1 else 0  # skip start node
    return disc, owned, creds


def _collect_directed_cbs(raw_cbs_env, n_samples, seed=None, debug=False):
    """
    Collect CBS observations using a stage-aware heuristic policy.
    Operates on the raw CBS env (not UnifiedSecEnv) for direct action control.

    raw_cbs_env: the object returned by make_cbs_env()
    debug: if True, print per-step state transitions.
    """
    rng = np.random.default_rng(seed)
    obs_list = []
    pbar = tqdm(total=n_samples, desc="Collecting CBS-directed")

    ep = 0
    while len(obs_list) < n_samples:
        s = (seed + ep) if seed is not None else None
        obs, _ = raw_cbs_env.reset(seed=s)
        done = truncated = False
        step = 0

        while not (done or truncated) and len(obs_list) < n_samples:
            obs_list.append(obs)  # raw CBS obs dict

            stage_before = stage_from_cbs(obs)
            disc_b, owned_b, creds_b = _cbs_state_snapshot(obs)

            action = _pick_cbs_action(obs, rng)
            obs, _, done, truncated, _ = raw_cbs_env.step(action)
            step += 1
            pbar.update(1)

            # Capture the terminal/winning observation (done=True means the attacker
            # reached the CBS win condition).  The loop exits before appending it,
            # so we do it here explicitly — this is the only way to get stage-4 obs.
            if done and len(obs_list) < n_samples:
                obs_list.append(obs)
                pbar.update(1)

            if debug:
                stage_after = stage_from_cbs(obs)
                disc_a, owned_a, creds_a = _cbs_state_snapshot(obs)
                transition = f"s{stage_before}→s{stage_after}"
                marker = " *** STAGE UP" if stage_after > stage_before else ""
                print(
                    f"  [CBS ep={ep+1:>3} step={step:>4}]  {_fmt_cbs_action(action):<40}"
                    f"  {transition}  disc={disc_a}  owned={owned_a}  creds={creds_a}"
                    f"{marker}",
                    flush=True
                )

        if debug:
            print(f"  [CBS ep={ep+1:>3}] episode end  done={done}  truncated={truncated}"
                  f"  total_steps={step}  obs_so_far={len(obs_list)}", flush=True)
        ep += 1

    pbar.close()
    return obs_list[:n_samples]


def _collect_directed_cw(cw_env, n_samples, seed=None, debug=False):
    """
    Collect CW observations using a stage-aware heuristic policy.

    Operates on a UnifiedSecEnv("cw", ...) instance so it can call
    _compute_unified_mask() for valid-action filtering and access
    _last_raw_obs for stage detection (same format as _collect_full_episodes).

    Stage → action priority (left = highest priority):
      Stage 0 (nothing done):   ping_sweep → port_scan → service_disc → move → escalate → impact
      Stage 1 (recon done):     move → service_disc → port_scan → ping_sweep → escalate → impact
      Stage 2 (foothold):       escalate → move → service_disc → port_scan → ping_sweep → impact
      Stage 3 (lateral spread): impact → escalate → move → service_disc → port_scan → ping_sweep
      Stage 4 (impacted):       nothing  (episode is about to end)

    Indices: 0=ping_sweep, 1=port_scan, 2=service_disc, 3=move,
             4=escalate, 5=impact, 6=nothing
    """
    STAGE_PRIORITY = [
        [0, 1, 2, 3, 4, 5, 6],  # stage 0: recon first — sweep → scan → discover
        [1, 2, 3, 0, 4, 5, 6],  # stage 1: port_scan → discover → move; ping_sweep to fill gaps
        [4, 3, 2, 1, 0, 5, 6],  # stage 2: escalate on current host, spread laterally
        [5, 4, 3, 2, 1, 0, 6],  # stage 3: impact! then keep escalating / moving
        [6, 5, 4, 3, 2, 1, 0],  # stage 4: already impacted — no-op until episode ends
    ]

    obs_list = []
    pbar = tqdm(total=n_samples, desc="Collecting CW-directed")

    ep = 0
    while len(obs_list) < n_samples:
        s = (seed + ep) if seed is not None else None
        cw_env.reset(seed=s)
        done = truncated = False
        step = 0

        while not (done or truncated) and len(obs_list) < n_samples:
            raw_obs = cw_env._last_raw_obs
            obs_list.append(raw_obs)

            # Kill-chain stage and valid-action mask for this step
            stage = stage_from_cw(raw_obs) if raw_obs is not None else 0
            try:
                mask = cw_env._compute_unified_mask()
            except Exception:
                mask = np.ones(len(cw_env.act_t.unified_actions), dtype=np.float32)

            # Pick highest-priority valid action for this stage
            priority = STAGE_PRIORITY[min(stage, len(STAGE_PRIORITY) - 1)]
            action = None
            for idx in priority:
                if idx < len(mask) and mask[idx] > 0.5:
                    action = idx
                    break
            if action is None:
                # Mask says nothing is valid (shouldn't happen) — safe no-op
                action = len(cw_env.act_t.unified_actions) - 1  # "nothing"

            stage_before = stage
            _, _, done, truncated, _ = cw_env.step(action)
            step += 1
            pbar.update(1)

            if debug:
                raw_after = cw_env._last_raw_obs
                stage_after = stage_from_cw(raw_after) if raw_after is not None else 0
                action_name = cw_env.act_t.unified_actions[action]
                marker = " *** STAGE UP" if stage_after > stage_before else ""
                print(
                    f"  [CW  ep={ep+1:>3} step={step:>4}]  {action_name:<15}"
                    f"  s{stage_before}→s{stage_after}{marker}",
                    flush=True,
                )

        if debug:
            print(
                f"  [CW  ep={ep+1:>3}] episode end  done={done}  truncated={truncated}"
                f"  total_steps={step}  obs_so_far={len(obs_list)}",
                flush=True,
            )
        ep += 1

    pbar.close()
    return obs_list[:n_samples]


def _load_agents(cw_agent_path, cbs_agent_path):
    cw_agent = None
    cbs_agent = None

    if cw_agent_path and os.path.isfile(cw_agent_path):
        try:
            print(f"Loading CyberWheel agent from {cw_agent_path}...")
            if cw_agent_path.endswith(".zip"):
                # SB3 PPO checkpoint (trained on unified 8D obs via UnifiedSecEnv)
                from stable_baselines3 import PPO
                cw_agent = PPO.load(cw_agent_path)
                print(f"  Loaded SB3 CW agent  obs={cw_agent.observation_space}  act={cw_agent.action_space}")
            else:
                # Raw RLPolicy .pt checkpoint (trained on native CW obs)
                import torch
                from cyberwheel.utils import RLPolicy
                from eval.eval_cw_checkpoints_on_cbs import infer_cyberwheel_config
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                action_space_size, obs_space_shape = infer_cyberwheel_config(cw_agent_path)
                cw_agent = RLPolicy(action_space_shape=action_space_size, obs_space_shape=obs_space_shape).to(device)
                state_dict = torch.load(cw_agent_path, map_location=device)
                cw_agent.load_state_dict(state_dict)
                cw_agent.eval()
                print(f"  Loaded RLPolicy CW agent")
        except Exception as e:
            print(f"  Warning: could not load CyberWheel agent: {e}")

    if cbs_agent_path and os.path.isfile(cbs_agent_path):
        try:
            from stable_baselines3 import PPO
            print(f"Loading CBS agent from {cbs_agent_path}...")
            cbs_agent = PPO.load(cbs_agent_path)
            print("  Loaded CBS agent")
        except Exception as e:
            print(f"  Warning: could not load CBS agent: {e}")

    return cw_agent, cbs_agent


def print_stage_distribution(label, obs_list, stage_fn):
    counts = Counter(stage_fn(o) for o in obs_list)
    total = len(obs_list)
    stage_names = ["nothing done", "recon done", "foothold", "escalated", "impact"]
    print(f"\n  {label} — {total} observations")
    print("  " + "-" * 52)
    for s in range(KILL_CHAIN_STAGES):
        n = counts[s]
        bar = "█" * int(20 * n / max(total, 1))
        print(f"  Stage {s} ({stage_names[s]:<14})  {n:>5}  ({100*n/max(total,1):5.1f}%)  {bar}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=1000,
                        help="Number of observations to collect per domain (fallback if --cw-samples/--cbs-samples not set)")
    parser.add_argument("--cw-samples", type=int, default=None,
                        help="Number of CW observations to collect (overrides --num-samples for CW)")
    parser.add_argument("--cbs-samples", type=int, default=None,
                        help="Number of CBS observations to collect (overrides --num-samples for CBS)")
    parser.add_argument("--out", type=str, default="data/obs.npz",
                        help="Output .npz file path")
    parser.add_argument("--cbs-only", action="store_true",
                        help="Skip CyberWheel (use if Python < 3.10)")
    parser.add_argument("--directed", action="store_true",
                        help="Use stage-aware heuristic policy on BOTH CW and CBS "
                             "to force all kill-chain stages (0-4) including 'impact'. "
                             "Strongly recommended for DAPN encoder training.")
    parser.add_argument("--cbs-agent", type=str, default=None,
                        help="Path to trained SB3 PPO .zip for CBS rollouts")
    parser.add_argument("--cw-agent", type=str, default=None,
                        help="Path to CyberWheel policy checkpoint (.pt) for rollouts")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug-steps", action="store_true",
                        help="Print per-step state transitions (stage before/after, "
                             "discovered nodes, owned nodes, credentials) to verify "
                             "the kill-chain progression during collection.")
    args = parser.parse_args()

    cw_agent, cbs_agent = _load_agents(args.cw_agent, args.cbs_agent)

    # Per-domain sample counts (--cw-samples / --cbs-samples override --num-samples)
    n_cw  = args.cw_samples  if args.cw_samples  is not None else args.num_samples
    n_cbs = args.cbs_samples if args.cbs_samples is not None else args.num_samples

    cw_obs = []
    cbs_obs = []

    # ----------------------------------------------------------------
    # Collect CW observations
    # ----------------------------------------------------------------
    if not args.cbs_only:
        print(f"\nCollecting CyberWheel observations ({n_cw}) ...")
        cw_env = None
        try:
            cw_env = UnifiedSecEnv("cw", cw_factory=make_cw_env)
            if args.directed:
                print("  Using directed heuristic policy (stage-aware kill-chain progression)")
                cw_obs = _collect_directed_cw(cw_env, n_cw, seed=args.seed,
                                              debug=args.debug_steps)
            else:
                if cw_agent is not None:
                    print("  Using trained CW agent")
                else:
                    print("  Using random policy  (tip: --directed for full stage coverage)")
                cw_obs = _collect_full_episodes(
                    cw_env, n_cw, agent=cw_agent, is_cbs=False, label="CW",
                    debug=args.debug_steps
                )
            print(f"  Collected {len(cw_obs)} CW observations")
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  ERROR: Could not collect CW observations — {e}")
            sys.exit(1)
        finally:
            if cw_env is not None:
                del cw_env
            gc.collect()
    else:
        print("\nSkipping CyberWheel (--cbs-only)")

    # ----------------------------------------------------------------
    # Collect CBS observations
    # ----------------------------------------------------------------
    print(f"\nCollecting CyberBattleSim observations ({n_cbs}) ...")
    cbs_env = None
    try:
        if args.directed:
            print("  Using directed heuristic policy (raw CBS action space)")
            cbs_env = make_cbs_env()   # raw CBS env, not UnifiedSecEnv
            cbs_raw = _collect_directed_cbs(cbs_env, n_cbs, seed=args.seed,
                                            debug=args.debug_steps)
        else:
            cbs_env = UnifiedSecEnv("cbs", cbs_factory=make_cbs_env)
            if cbs_agent is not None:
                print("  Using trained CBS agent")
            else:
                print("  Using random policy  (tip: --directed for better stage coverage)")
            cbs_raw = _collect_full_episodes(
                cbs_env, n_cbs, agent=cbs_agent, is_cbs=True, label="CBS",
                debug=args.debug_steps
            )

        # Strip action_mask to avoid OOM when saving
        cbs_obs = [_strip_cbs_obs(o) if isinstance(o, dict) else o for o in cbs_raw]
        print(f"  Collected {len(cbs_obs)} CBS observations")

    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"  ERROR: Could not collect CBS observations — {e}")
        sys.exit(1)
    finally:
        if cbs_env is not None:
            del cbs_env
        gc.collect()

    # ----------------------------------------------------------------
    # Stage distribution summary
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("KILL-CHAIN STAGE DISTRIBUTION")
    print("=" * 60)
    if cw_obs:
        print_stage_distribution("CyberWheel (source)", cw_obs, stage_from_cw)
    print_stage_distribution("CyberBattleSim (target)", cbs_obs, stage_from_cbs)

    # ----------------------------------------------------------------
    # Save
    # ----------------------------------------------------------------
    out_path = args.out
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    np.savez(
        out_path,
        source_obs=np.array(cw_obs, dtype=object),
        target_obs=np.array(cbs_obs, dtype=object),
    )
    print(f"\nSaved to {out_path}")
    print(f"  source_obs (CW):  {len(cw_obs)}")
    print(f"  target_obs (CBS): {len(cbs_obs)}")
    print(f"\nTo train DAPN encoder from this file:")
    print(f"  python train_dapn_encoder.py --load-data {out_path} --epochs 50")


if __name__ == "__main__":
    main()
