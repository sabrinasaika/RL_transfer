#!/usr/bin/env python3
"""
Show paired CW ↔ CBS observations grouped by kill-chain stage.
Usage:
    python show_pairs.py --data data/obs.npz --n-per-stage 2
"""
import sys, os
from pathlib import Path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "cyberwheel"))

import argparse
import numpy as np
from adapters.kill_chain import stage_from_cbs, stage_from_cw, KILL_CHAIN_STAGES
from train_dapn_encoder import build_stage_pairs
from adapters.unified_full_obs_preprocessor import UnifiedFullObsPreprocessor

STAGE_NAMES = ["nothing done", "recon done", "foothold", "lateral/escalated", "impact"]

def fmt_cw(obs):
    """Summarise a raw CW obs vector."""
    if isinstance(obs, dict):
        obs = obs.get("red", obs.get("obs", np.array([])))
    obs = np.asarray(obs, dtype=np.float32).ravel()
    HOST_ATTRS = 7
    n = obs.size
    standalone = n % HOST_ATTRS
    n_hosts = (n - standalone) // HOST_ATTRS if n >= HOST_ATTRS else 0
    discovered = escalated = on_host = impacted = 0
    for i in range(n_hosts):
        b = i * HOST_ATTRS
        c = obs[b:b+HOST_ATTRS]
        if np.all(c == -1): continue
        discovered  += int(c[3] == 1)
        on_host     += int(c[4] == 1)
        escalated   += int(c[5] == 1)
        impacted    += int(c[6] == 1)
    return (f"hosts={n_hosts}  disc={discovered}  on_host={on_host}  "
            f"esc={escalated}  impact={impacted}  obs_len={n}")

def fmt_cbs(obs):
    """Summarise a raw CBS obs dict."""
    if not isinstance(obs, dict):
        return str(obs)[:80]
    priv = np.asarray(obs.get("nodes_privilegelevel", []), dtype=np.float32)
    priv_t = priv[1:] if priv.size > 1 else priv
    return (
        f"disc_nodes={int(obs.get('discovered_node_count',0) or 0)}"
        f"  priv_targets={priv_t.tolist()[:8]}"
        f"  cred_len={int(obs.get('credential_cache_length',0) or 0)}"
        f"  esc={int(obs.get('escalation',0) or 0)}"
        f"  lat={int(obs.get('lateral_move',0) or 0)}"
        f"  cust_data={int(obs.get('customer_data_found',0) or 0)}"
        f"  probe={int(obs.get('probe_result',0) or 0)}"
    )

def fmt_vec(v, label):
    """Show nonzero count and first few nonzero entries of a 512D vector."""
    nz = int((v != 0).sum())
    top = np.argsort(np.abs(v))[::-1][:5]
    top_vals = [(int(i), round(float(v[i]), 3)) for i in top if v[i] != 0]
    return f"{label:>3}D  nonzero={nz:>3}  top5={top_vals}"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/obs.npz")
    parser.add_argument("--n-per-stage", type=int, default=2,
                        help="Number of example pairs to show per stage")
    args = parser.parse_args()

    data = np.load(args.data, allow_pickle=True)
    src_obs = data["source_obs"].tolist()
    tgt_obs = data["target_obs"].tolist()

    print(f"\nLoaded {len(src_obs)} CW obs, {len(tgt_obs)} CBS obs from {args.data}")

    # ── Stage distributions ────────────────────────────────────────────────────
    cw_stages  = [stage_from_cw(o)  for o in src_obs]
    cbs_stages = [stage_from_cbs(o) for o in tgt_obs]
    from collections import Counter
    print("\n── CW stage distribution ──────────────────────────")
    for s in range(KILL_CHAIN_STAGES):
        n = Counter(cw_stages)[s]
        print(f"  Stage {s} ({STAGE_NAMES[s]:<18})  {n:>5}  ({100*n/max(len(src_obs),1):5.1f}%)")
    print("\n── CBS stage distribution ─────────────────────────")
    for s in range(KILL_CHAIN_STAGES):
        n = Counter(cbs_stages)[s]
        print(f"  Stage {s} ({STAGE_NAMES[s]:<18})  {n:>5}  ({100*n/max(len(tgt_obs),1):5.1f}%)")

    # ── Build pairs ────────────────────────────────────────────────────────────
    print("\n── Building stage-matched pairs ───────────────────")
    paired_src, paired_tgt, stage_counts = build_stage_pairs(src_obs, tgt_obs)
    print(f"Total pairs: {len(paired_src)}")

    # ── 512D preprocessed vectors ──────────────────────────────────────────────
    prep = UnifiedFullObsPreprocessor(unified_dim=512)

    # ── Per-stage sample display ───────────────────────────────────────────────
    # Rebuild per-stage buckets from the pairs
    paired_stages = [stage_from_cw(s) for s in paired_src]
    buckets = {s: [] for s in range(KILL_CHAIN_STAGES)}
    for i, (s, ps, pt) in enumerate(zip(paired_stages, paired_src, paired_tgt)):
        buckets[s].append((ps, pt))

    rng = np.random.default_rng(0)
    for stage in range(KILL_CHAIN_STAGES):
        items = buckets[stage]
        if not items:
            print(f"\n{'='*70}")
            print(f"  STAGE {stage}  ({STAGE_NAMES[stage]})  — NO PAIRS")
            continue
        print(f"\n{'='*70}")
        print(f"  STAGE {stage}  ({STAGE_NAMES[stage]})  — {len(items)} pairs total")
        print(f"{'='*70}")
        idxs = rng.choice(len(items), min(args.n_per_stage, len(items)), replace=False)
        for rank, idx in enumerate(idxs, 1):
            ps, pt = items[idx]
            cbs_stage = stage_from_cbs(pt)
            print(f"\n  Pair {rank}:")
            print(f"    CW  (stage={stage_from_cw(ps)})  {fmt_cw(ps)}")
            print(f"    CBS (stage={cbs_stage})  {fmt_cbs(pt)}")
            # 512D vectors
            cw_vec  = prep.preprocess_cw(ps)
            cbs_vec = prep.preprocess_cbs(pt)
            print(f"    {fmt_vec(cw_vec, 'CW ')}")
            print(f"    {fmt_vec(cbs_vec, 'CBS')}")

    print(f"\n{'='*70}")
    print("Done.")

if __name__ == "__main__":
    main()
