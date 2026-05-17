#!/usr/bin/env python3
"""
Show paired CW ↔ CBS observations side-by-side, including:
  - Raw field values (per-host CW table, CBS dict fields)
  - 512D preprocessed vector segments (key slices + cosine similarity)

Usage:
    PYTHONPATH=$PWD:$PWD/cyberwheel python3.10 show_pairs.py --data data/obs.npz
    PYTHONPATH=$PWD:$PWD/cyberwheel python3.10 show_pairs.py --data data/obs.npz --n-per-stage 3 --stage 2
"""
import sys, os
from pathlib import Path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "cyberwheel"))

import argparse
import numpy as np
from collections import Counter
from adapters.kill_chain import stage_from_cbs, stage_from_cw, KILL_CHAIN_STAGES
from train_dapn_encoder import build_stage_pairs
from adapters.unified_full_obs_preprocessor import UnifiedFullObsPreprocessor

STAGE_NAMES = ["nothing", "recon", "foothold", "lateral", "impact"]
CW_HOST_FIELDS = ["type", "sweeped", "scanned", "disc", "on_host", "esc", "impacted"]
HOST_ATTRS = 7

SEP  = "=" * 72
SEP2 = "-" * 72


# ── Raw state printers ────────────────────────────────────────────────────────

def print_cw_raw(obs, indent="    "):
    """Print per-host breakdown of a raw CW observation vector."""
    if isinstance(obs, dict):
        obs = obs.get("red", obs.get("obs", np.array([])))
    vec = np.asarray(obs, dtype=np.float32).ravel()
    n_hosts = vec.size // HOST_ATTRS
    standalone = vec.size % HOST_ATTRS

    print(f"{indent}CW raw obs  ({vec.size}D, {n_hosts} hosts{', +%d standalone' % standalone if standalone else ''})")
    header = ("  ".join(f"{f:>8}" for f in CW_HOST_FIELDS))
    print(f"{indent}  host  {header}")
    print(f"{indent}  {'─'*60}")
    any_active = False
    for i in range(n_hosts):
        chunk = vec[i * HOST_ATTRS: i * HOST_ATTRS + HOST_ATTRS]
        if np.all(chunk == -1):
            continue          # padding slot — skip
        any_active = True
        vals = "  ".join(f"{float(v):>8.3f}" for v in chunk)
        # Highlight active flags
        flags = ""
        if chunk[3] == 1: flags += " DISC"
        if chunk[4] == 1: flags += " ON"
        if chunk[5] == 1: flags += " ESC"
        if chunk[6] == 1: flags += " IMPACT"
        print(f"{indent}  [{i:>2}]  {vals}  {flags}")
    if not any_active:
        print(f"{indent}  (all hosts are padding / -1)")
    if standalone:
        tail = vec[n_hosts * HOST_ATTRS:]
        print(f"{indent}  standalone tail: {tail.tolist()}")


def print_cbs_raw(obs, indent="    "):
    """Print field-by-field CBS observation dict."""
    if not isinstance(obs, dict):
        print(f"{indent}CBS obs is not a dict: {type(obs)}")
        return

    priv = np.asarray(obs.get("nodes_privilegelevel", []), dtype=np.int32)
    priv_t = priv[1:] if priv.size > 1 else priv   # skip start node

    print(f"{indent}CBS raw obs  ({len(obs)} fields)")
    print(f"{indent}  discovered_node_count    : {int(obs.get('discovered_node_count', 0) or 0)}")
    print(f"{indent}  credential_cache_length  : {int(obs.get('credential_cache_length', 0) or 0)}")
    print(f"{indent}  escalation               : {int(obs.get('escalation', 0) or 0)}")
    print(f"{indent}  lateral_move             : {int(obs.get('lateral_move', 0) or 0)}")
    print(f"{indent}  newly_discovered_nodes   : {int(obs.get('newly_discovered_nodes_count', 0) or 0)}")
    print(f"{indent}  customer_data_found      : {int(obs.get('customer_data_found', 0) or 0)}")
    print(f"{indent}  probe_result             : {int(obs.get('probe_result', 0) or 0)}")

    # nodes_privilegelevel — show all nodes (skip zeros for clarity)
    if priv.size > 0:
        priv_str = "  ".join(
            f"n{i}={int(p)}" for i, p in enumerate(priv) if int(p) > 0
        ) or "none owned"
        print(f"{indent}  nodes with priv>0        : {priv_str}")
        print(f"{indent}  priv_levels (all)        : {priv.tolist()}")

    # Credential cache matrix — first 5 cached credentials
    ccm = obs.get("credential_cache_matrix", ())
    if isinstance(ccm, (tuple, list, np.ndarray)) and len(ccm) > 0:
        arr = np.asarray(ccm, dtype=np.float32)
        nonzero_rows = arr[np.any(arr != 0, axis=-1)] if arr.ndim == 2 else arr
        shown = nonzero_rows[:5].tolist() if len(nonzero_rows) > 0 else []
        print(f"{indent}  credential_cache (first 5 non-zero): {shown}")

    # discovered_nodes_properties — shape summary
    props = obs.get("discovered_nodes_properties", None)
    if props is not None:
        arr = np.asarray(props, dtype=np.float32)
        nz = int((arr != 0).sum())
        print(f"{indent}  discovered_nodes_properties shape={arr.shape}  nonzero={nz}")


# ── 512D vector printer ───────────────────────────────────────────────────────

def print_vec_comparison(cw_vec, cbs_vec, indent="    "):
    """
    Print key segments of the two 512D vectors side-by-side and cosine similarity.

    CBS layout  (from UnifiedFullObsPreprocessor):
      [  0:100]  nodes_privilegelevel (100D)
      [100:106]  scalars: disc_count, cred_len, esc, lat, new_disc, cust_data (6D)
      [106:206]  credential_cache_matrix encoded (100D)
      [206:512]  node_properties flattened (306D)

    CW layout:
      [  0:N]    raw red-agent vector (7D × n_hosts), zero-padded to 512D
      First host starts at [0], host k at [7k].
    """
    cw_vec  = np.asarray(cw_vec,  dtype=np.float32)
    cbs_vec = np.asarray(cbs_vec, dtype=np.float32)

    # Cosine similarity
    cw_norm  = np.linalg.norm(cw_vec)
    cbs_norm = np.linalg.norm(cbs_vec)
    cosine   = float(np.dot(cw_vec, cbs_vec) / (cw_norm * cbs_norm + 1e-8))

    nz_cw  = int((cw_vec  != 0).sum())
    nz_cbs = int((cbs_vec != 0).sum())

    print(f"{indent}512D vector stats:")
    print(f"{indent}  CW  nonzero={nz_cw:>3}  norm={cw_norm:.3f}")
    print(f"{indent}  CBS nonzero={nz_cbs:>3}  norm={cbs_norm:.3f}")
    print(f"{indent}  cosine_similarity = {cosine:.4f}  "
          f"({'high ✓' if cosine > 0.5 else 'low — domains differ here'})")

    # ── Segment: CBS priv / CW host escalation flags ──────────────────────────
    print(f"{indent}  Segment [0:10]  (CBS=priv_levels nodes 0-9 | CW=host_0 attrs):")
    cw_seg  = cw_vec[:10].tolist()
    cbs_seg = cbs_vec[:10].tolist()
    print(f"{indent}    CW : {[round(v,3) for v in cw_seg]}")
    print(f"{indent}    CBS: {[round(v,3) for v in cbs_seg]}")

    # ── Segment: CBS scalars [100:106] ────────────────────────────────────────
    cbs_scalar_names = ["disc_count", "cred_len", "esc", "lat", "new_disc", "cust_data"]
    cbs_scalars = cbs_vec[100:106].tolist()
    # Corresponding CW positions: host 14 chunk [98:104] — show for reference
    cw_around = cw_vec[98:106].tolist()
    print(f"{indent}  Segment [100:106]  (CBS=scalars | CW=hosts 14-15 region):")
    print(f"{indent}    CBS scalars: " +
          "  ".join(f"{n}={round(v,3)}" for n, v in zip(cbs_scalar_names, cbs_scalars)))
    print(f"{indent}    CW [98:106]: {[round(v,3) for v in cw_around]}")

    # ── Top active dims ───────────────────────────────────────────────────────
    top_cw  = np.argsort(np.abs(cw_vec))[::-1][:8]
    top_cbs = np.argsort(np.abs(cbs_vec))[::-1][:8]
    print(f"{indent}  Top-8 active dims:")
    print(f"{indent}    CW : " + "  ".join(f"[{i}]={round(float(cw_vec[i]),3)}" for i in top_cw))
    print(f"{indent}    CBS: " + "  ".join(f"[{i}]={round(float(cbs_vec[i]),3)}" for i in top_cbs))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/obs.npz")
    parser.add_argument("--n-per-stage", type=int, default=2,
                        help="Number of example pairs to show per stage")
    parser.add_argument("--stage", type=int, default=None,
                        help="Show only this stage (0-4). Default: all stages.")
    args = parser.parse_args()

    data = np.load(args.data, allow_pickle=True)
    src_obs = data["source_obs"].tolist()
    tgt_obs = data["target_obs"].tolist()
    print(f"\nLoaded {len(src_obs)} CW obs, {len(tgt_obs)} CBS obs from {args.data}")

    # ── Stage distributions ───────────────────────────────────────────────────
    cw_stages  = [stage_from_cw(o)  for o in src_obs]
    cbs_stages = [stage_from_cbs(o) for o in tgt_obs]

    print("\n── CW stage distribution ──────────────────────────")
    for s in range(KILL_CHAIN_STAGES):
        n = Counter(cw_stages)[s]
        bar = "█" * int(20 * n / max(len(src_obs), 1))
        print(f"  Stage {s} ({STAGE_NAMES[s]:<10})  {n:>5}  ({100*n/max(len(src_obs),1):5.1f}%)  {bar}")

    print("\n── CBS stage distribution ─────────────────────────")
    for s in range(KILL_CHAIN_STAGES):
        n = Counter(cbs_stages)[s]
        bar = "█" * int(20 * n / max(len(tgt_obs), 1))
        print(f"  Stage {s} ({STAGE_NAMES[s]:<10})  {n:>5}  ({100*n/max(len(tgt_obs),1):5.1f}%)  {bar}")

    # ── Build stage-matched pairs ─────────────────────────────────────────────
    print("\n── Building stage-matched pairs ───────────────────")
    paired_src, paired_tgt, stage_counts = build_stage_pairs(src_obs, tgt_obs)
    print(f"Total pairs: {len(paired_src)}")
    for s, c in stage_counts.items():
        if c > 0:
            print(f"  Stage {s} ({STAGE_NAMES[s]:<10}): {c} pairs")

    # ── 512D preprocessor ────────────────────────────────────────────────────
    prep = UnifiedFullObsPreprocessor(unified_dim=512)

    # ── Per-stage buckets ─────────────────────────────────────────────────────
    paired_stages = [stage_from_cw(s) for s in paired_src]
    buckets = {s: [] for s in range(KILL_CHAIN_STAGES)}
    for i, (s, ps, pt) in enumerate(zip(paired_stages, paired_src, paired_tgt)):
        buckets[s].append((ps, pt))

    stages_to_show = [args.stage] if args.stage is not None else range(KILL_CHAIN_STAGES)
    rng = np.random.default_rng(0)

    for stage in stages_to_show:
        items = buckets[stage]
        print(f"\n{SEP}")
        print(f"  STAGE {stage}  ({STAGE_NAMES[stage].upper()})  —  {len(items)} pairs total")
        print(SEP)

        if not items:
            print("  No pairs at this stage.")
            continue

        idxs = rng.choice(len(items), min(args.n_per_stage, len(items)), replace=False)
        for rank, idx in enumerate(idxs, 1):
            cw_raw, cbs_raw = items[idx]
            cw_stage  = stage_from_cw(cw_raw)
            cbs_stage = stage_from_cbs(cbs_raw)

            print(f"\n  ── Pair {rank}  (CW stage={cw_stage}, CBS stage={cbs_stage}) ──")
            print(SEP2)

            # Raw values
            print_cw_raw(cw_raw,  indent="  ")
            print()
            print_cbs_raw(cbs_raw, indent="  ")
            print()

            # 512D vectors
            cw_vec  = prep.preprocess_cw(cw_raw)
            cbs_vec = prep.preprocess_cbs(cbs_raw)
            print_vec_comparison(cw_vec, cbs_vec, indent="  ")
            print(SEP2)

    print(f"\n{SEP}")
    print("Done.")


if __name__ == "__main__":
    main()
