#!/usr/bin/env python3
"""
Diagnose kill-chain stage distribution and pairing counts.

Shows:
  - How many CW obs fall into each stage (0-4)
  - How many CBS obs fall into each stage (0-4)
  - How many pairs were formed per stage after build_stage_pairs()

Usage:
    python diagnose_stage_pairs.py --num-samples 500
"""

import os
import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "cyberwheel"))

import argparse
import numpy as np
from collections import Counter

from adapters.kill_chain import stage_from_cbs, stage_from_cw, KILL_CHAIN_STAGES
from train_dapn_encoder import (
    _collect_full_episodes,
    build_stage_pairs,
)
from adapters.unified_env import UnifiedSecEnv
from config.env_builders import make_cbs_env, make_cw_env


STAGE_NAMES = {
    0: "Stage 0 — nothing done",
    1: "Stage 1 — recon done, no foothold",
    2: "Stage 2 — foothold (on_host / node owned)",
    3: "Stage 3 — privilege escalated",
    4: "Stage 4 — impact / exfiltration",
}


def print_stage_distribution(label, obs_list, stage_fn):
    counts = Counter()
    for obs in obs_list:
        counts[stage_fn(obs)] += 1
    total = len(obs_list)

    print(f"\n{label} — {total} observations")
    print("-" * 50)
    for s in range(KILL_CHAIN_STAGES):
        n = counts[s]
        bar = "█" * int(30 * n / max(total, 1))
        print(f"  {STAGE_NAMES[s]:<40}  {n:>5}  ({100*n/max(total,1):5.1f}%)  {bar}")
    print(f"  {'TOTAL':<40}  {total:>5}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=300,
                        help="Obs to collect per domain")
    parser.add_argument("--load-data", type=str, default=None,
                        help="Load existing .npz instead of collecting")
    parser.add_argument("--cbs-only", action="store_true",
                        help="Skip CyberWheel collection")
    args = parser.parse_args()

    # ----------------------------------------------------------------
    # Collect or load observations
    # ----------------------------------------------------------------
    if args.load_data and os.path.exists(args.load_data):
        print(f"Loading observations from {args.load_data} ...")
        data = np.load(args.load_data, allow_pickle=True)
        cw_obs  = data["source_obs"].tolist() if "source_obs" in data else []
        cbs_obs = data["target_obs"].tolist() if "target_obs" in data else []
        print(f"  Loaded {len(cw_obs)} CW obs, {len(cbs_obs)} CBS obs")
    else:
        cw_obs = []
        if not args.cbs_only:
            print(f"\nCollecting CW observations ({args.num_samples}) ...")
            try:
                cw_env = UnifiedSecEnv("cw", cw_factory=make_cw_env)
                cw_obs = _collect_full_episodes(
                    cw_env, args.num_samples, is_cbs=False, label="CW"
                )
            except Exception as e:
                print(f"  Warning: CW unavailable — {e}")

        print(f"\nCollecting CBS observations ({args.num_samples}) ...")
        cbs_env = UnifiedSecEnv("cbs", cbs_factory=make_cbs_env)
        cbs_obs = _collect_full_episodes(
            cbs_env, args.num_samples, is_cbs=True, label="CBS"
        )

    if len(cbs_obs) == 0:
        print("No CBS observations collected. Exiting.")
        sys.exit(1)

    # ----------------------------------------------------------------
    # Stage distribution BEFORE pairing
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("KILL-CHAIN STAGE DISTRIBUTION (before pairing)")
    print("=" * 60)

    if cw_obs:
        print_stage_distribution("CyberWheel (source)", cw_obs, stage_from_cw)
    print_stage_distribution("CyberBattleSim (target)", cbs_obs, stage_from_cbs)

    # ----------------------------------------------------------------
    # Build pairs and show counts
    # ----------------------------------------------------------------
    if cw_obs:
        print("\n" + "=" * 60)
        print("PAIRING RESULTS (kill-chain stage matching)")
        print("=" * 60)

        paired_src, paired_tgt, stage_counts = build_stage_pairs(cw_obs, cbs_obs)

        print(f"\n  Total CW obs:     {len(cw_obs)}")
        print(f"  Total CBS obs:    {len(cbs_obs)}")
        print(f"  Total pairs made: {len(paired_src)}")

        print(f"\n  Pairs formed per stage:")
        print("  " + "-" * 46)
        total_pairs = sum(stage_counts.values())
        for s in range(KILL_CHAIN_STAGES):
            n = stage_counts[s]
            bar = "█" * int(30 * n / max(total_pairs, 1))
            print(f"  {STAGE_NAMES[s]:<40}  {n:>5}  {bar}")
        print(f"  {'TOTAL':<40}  {total_pairs:>5}")

        # ----------------------------------------------------------------
        # Warn about stage imbalance
        # ----------------------------------------------------------------
        print("\n  Pairing quality:")
        cw_counts   = Counter(stage_from_cw(o)  for o in cw_obs)
        cbs_counts  = Counter(stage_from_cbs(o) for o in cbs_obs)
        for s in range(KILL_CHAIN_STAGES):
            cw_n  = cw_counts[s]
            cbs_n = cbs_counts[s]
            paired_n = stage_counts[s]
            if cw_n == 0 and cbs_n == 0:
                continue
            if cbs_n == 0:
                status = "⚠  No CBS obs at this stage — fell back to nearest stage"
            elif cw_n == 0:
                status = "—  No CW obs at this stage"
            else:
                ratio = min(cw_n, cbs_n) / max(cw_n, cbs_n)
                if ratio > 0.5:
                    status = "✓  Good balance"
                else:
                    status = "⚠  Imbalanced — consider collecting more episodes"
            print(f"    Stage {s}: CW={cw_n:>4}  CBS={cbs_n:>4}  pairs={paired_n:>4}  {status}")
    else:
        print("\nSkipping pairing (no CW obs available)")


if __name__ == "__main__":
    main()
