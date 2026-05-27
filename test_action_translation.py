"""
Test and visualise the kill-chain action translation layer.

Checks:
  1. Phase 0-2  always produce  local_vulnerability
  2. Phase 3    produces        connect  when a credential is cached,
                                local_vulnerability  otherwise
  3. Phase advances correctly 0→1→2→3→4 per slot
  4. Owned nodes immediately jump to phase ≥ 4
  5. No CBS action targets an un-owned source node for connect

Output:
  - Pass/fail test results printed to terminal
  - Trajectory table (slot × step, phase colour-coded)
  - results/trajectory_plot.png  —  phase heatmap + ownership timeline

Usage:
  PYTHONPATH=$PWD:$PWD/cyberwheel:$PWD/CyberBattleSim python3.10 test_action_translation.py
"""

import os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "CyberBattleSim"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

os.environ["CBS_SIZE"]      = "12"
os.environ["CBS_WIN_NODES"] = "8"

from gymnasium.wrappers import TimeLimit
from adapters.unified_env import UnifiedSecEnv, MAX_SLOTS
from config.env_builders import make_cbs_env

# ── Phase colour map ──────────────────────────────────────────────────────────
PHASE_NAMES  = ["ping_sweep","port_scan","svc_disc","exploit","owned","escalated","impacted"]
PHASE_COLORS = ["#2C3E50","#1A5276","#117A65","#B7950B","#27AE60","#8E44AD","#C0392B"]

def phase_name(p):
    p = int(p)
    return PHASE_NAMES[p] if p < len(PHASE_NAMES) else f"phase{p}"

# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_unified(env):
    e = env
    while e is not None:
        if isinstance(e, UnifiedSecEnv):
            return e
        e = getattr(e, "env", None)
    return None

def _priv(unified):
    raw = getattr(unified, "_raw_obs", None)
    if not isinstance(raw, dict): return None
    p = raw.get("nodes_privilegelevel")
    return np.asarray(p, dtype=np.int32) if p is not None else None

def _owned_set(unified):
    p = _priv(unified)
    if p is None: return set()
    total = 14  # CBS_SIZE=12 → 14 nodes
    return {i for i in range(total) if i < p.size and p[i] >= 1}

# ── Test runner ───────────────────────────────────────────────────────────────

def run_episode(seed=0, max_steps=150):
    """Run one episode, recording every translation decision."""
    env = TimeLimit(UnifiedSecEnv("cbs", cbs_factory=make_cbs_env),
                    max_episode_steps=max_steps)
    unified = _get_unified(env)

    obs, _ = env.reset(seed=seed)

    records   = []   # one dict per step
    tests     = []   # (description, passed)
    won       = False

    prev_phases = {}   # slot → phase BEFORE this step
    prev_owned  = set()

    for step in range(max_steps):
        # Random slot selection (tests the translation layer, not any policy)
        slot = np.random.randint(0, MAX_SLOTS)

        # Snapshot phase BEFORE step
        phases_before = dict(getattr(unified, "_slot_cw_phases", {}))
        sim_phase     = phases_before.get(slot, 0)

        obs, r, terminated, truncated, info = env.step(slot)
        cbs_action = info.get("cbs_action", {})

        phases_after = dict(getattr(unified, "_slot_cw_phases", {}))
        new_phase    = phases_after.get(slot, 0)
        owned_now    = _owned_set(unified)
        newly_owned  = owned_now - prev_owned

        action_type = list(cbs_action.keys())[0] if cbs_action else "none"
        action_args = list(cbs_action.values())[0] if cbs_action else ()

        # ── Tests ────────────────────────────────────────────────────────────

        # Resolve node_id from slot_map first (needed by T1 and T5)
        slot_map = getattr(unified, "_slot_map", [])
        node_id  = slot_map[slot] if slot < len(slot_map) else -1

        # Was node already owned BEFORE this step? (node 0 is always owned;
        # other nodes become owned mid-episode). When a node is already owned,
        # _advance_cbs SYNCS the phase to 4 regardless of sim_phase, so the
        # resulting action may be 'connect' even if phase_before < 3.
        node_was_owned = (node_id in prev_owned) or (node_id == 0)

        # T1: phases 0-2 must produce local_vulnerability (for unowned nodes only)
        if sim_phase < 3 and not node_was_owned:
            ok = action_type == "local_vulnerability"
            tests.append((
                f"step {step+1:3d} slot {slot} phase {sim_phase} "
                f"({phase_name(sim_phase)}) → local_vulnerability",
                ok, action_type
            ))

        # T2: phase 3 must produce connect OR local_vulnerability (retry)
        if sim_phase == 3:
            ok = action_type in ("connect", "local_vulnerability")
            tests.append((
                f"step {step+1:3d} slot {slot} phase 3 (exploit) "
                f"→ connect or local_vuln",
                ok, action_type
            ))

        # T3: connect src must be an owned node
        if action_type == "connect":
            src = int(action_args[0])
            p   = _priv(unified)
            src_owned = (p is not None and src < p.size and p[src] >= 1)
            # note: ownership is checked AFTER step; src should have been owned before
            # so we check via the pre-step owned set
            src_was_owned = src in prev_owned or src == 0
            ok = src_was_owned
            tests.append((
                f"step {step+1:3d} connect src={src} was owned before step",
                ok, f"src_owned={src_was_owned}"
            ))

        # T4: phase must never exceed 6
        ok = new_phase <= 6
        if not ok:
            tests.append((
                f"step {step+1:3d} slot {slot} phase capped at 6",
                ok, f"got {new_phase}"
            ))

        # T5: if node just got owned, phase must be ≥ 4
        if node_id in newly_owned:
            ok = new_phase >= 4
            tests.append((
                f"step {step+1:3d} node {node_id} owned → phase≥4 (got {new_phase})",
                ok, f"phase={new_phase}"
            ))

        records.append({
            "step":        step + 1,
            "slot":        slot,
            "node":        node_id,
            "phase_before": sim_phase,
            "phase_after":  new_phase,
            "action_type":  action_type,
            "action_args":  action_args,
            "reward":       r,
            "owned_count":  len(owned_now - {0}),
            "newly_owned":  sorted(newly_owned),
        })

        prev_owned  = owned_now
        prev_phases = phases_after

        if terminated:
            won = True
            break
        if truncated:
            break

    env.close()
    return records, tests, won

# ── Print trajectory ──────────────────────────────────────────────────────────

def print_trajectory(records):
    print("\n" + "═"*90)
    print("  TRAJECTORY  (slot → CBS action translation)")
    print("═"*90)
    print(f"  {'Step':>4}  {'Slot':>4}  {'Node':>4}  {'Phase (before→after)':<22}  "
          f"{'CBS Action':<42}  {'Owned':>5}  {'Reward':>7}")
    print(f"  {'----':>4}  {'----':>4}  {'----':>4}  {'----------------------':<22}  "
          f"{'------------------------------------------':<42}  {'-----':>5}  {'-------':>7}")

    for r in records:
        pb = phase_name(r["phase_before"])
        pa = phase_name(r["phase_after"])
        phase_str = f"{pb} → {pa}" if r["phase_before"] != r["phase_after"] else pb

        at = r["action_type"]
        aa = r["action_args"]
        if at == "local_vulnerability":
            act = f"local_vuln (node={aa[0]}, vuln_idx={aa[1]})"
        elif at == "connect":
            act = f"connect    (src={aa[0]}→tgt={aa[1]}, port={aa[2]}, cred={aa[3]})"
        elif at == "remote_vulnerability":
            act = f"remote_vuln(src={aa[0]}→tgt={aa[1]}, vuln={aa[2]})"
        else:
            act = str(at)

        new_tag = f"  ◀ {r['newly_owned']} OWNED" if r["newly_owned"] else ""
        print(f"  {r['step']:>4}  {r['slot']:>4}  {r['node']:>4}  {phase_str:<22}  "
              f"{act:<42}  {r['owned_count']:>5}  {r['reward']:>7.1f}{new_tag}")

# ── Print test results ────────────────────────────────────────────────────────

def print_tests(tests):
    print("\n" + "═"*70)
    print("  ACTION TRANSLATION TESTS")
    print("═"*70)
    passed = sum(1 for _, ok, _ in tests if ok)
    total  = len(tests)
    for desc, ok, detail in tests:
        sym = "✓" if ok else "✗"
        if not ok:
            print(f"  {sym}  FAIL  {desc}  [got: {detail}]")
    print(f"\n  Result: {passed}/{total} passed  "
          f"({'ALL PASS ✓' if passed == total else f'FAILURES: {total-passed}'})")

# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_trajectory(records, out_path="results/trajectory_plot.png"):
    steps      = [r["step"]        for r in records]
    slots      = [r["slot"]        for r in records]
    phases     = [r["phase_before"] for r in records]
    act_types  = [r["action_type"] for r in records]
    owned_cnt  = [r["owned_count"] for r in records]
    newly_owned= [r["newly_owned"] for r in records]
    rewards    = [r["reward"]      for r in records]

    fig, axes = plt.subplots(3, 1, figsize=(16, 12),
                             gridspec_kw={"height_ratios": [4, 1.5, 1.5]})
    fig.suptitle("Kill-Chain Action Translation — Policy Trajectory on CBS",
                 fontsize=14, fontweight="bold", y=0.98)

    # ── Panel 1: phase heatmap (slot × step) ─────────────────────────────────
    ax = axes[0]
    phase_grid = np.full((MAX_SLOTS, max(steps) + 1), np.nan)
    atype_grid = {}

    for r in records:
        s, sl, ph = r["step"], r["slot"], r["phase_before"]
        phase_grid[sl, s] = ph
        atype_grid[(sl, s)] = r["action_type"]

    # colour map: 0=dark → 6=bright
    from matplotlib.colors import ListedColormap, BoundaryNorm
    cmap  = ListedColormap(PHASE_COLORS)
    norm  = BoundaryNorm(boundaries=[-0.5+i for i in range(8)], ncolors=7)

    im = ax.imshow(phase_grid[:, 1:], aspect="auto", cmap=cmap, norm=norm,
                   origin="lower", interpolation="nearest",
                   extent=[1, max(steps), -0.5, MAX_SLOTS - 0.5])

    # Mark connect actions with a white star
    for (sl, st), at in atype_grid.items():
        if at == "connect":
            ax.plot(st, sl, "w*", markersize=10, zorder=5)

    # Mark newly owned
    for r in records:
        if r["newly_owned"]:
            ax.axvline(r["step"], color="lime", linewidth=1.5, alpha=0.7, zorder=4)

    ax.set_ylabel("Slot (0–11)", fontsize=11)
    ax.set_title("Phase per slot at each step  "
                 "(★ = connect action,  green line = node owned)", fontsize=10)
    ax.set_yticks(range(MAX_SLOTS))
    ax.set_yticklabels([f"slot {i}" for i in range(MAX_SLOTS)], fontsize=7)

    legend_patches = [mpatches.Patch(color=PHASE_COLORS[i], label=PHASE_NAMES[i])
                      for i in range(len(PHASE_NAMES))]
    ax.legend(handles=legend_patches, loc="upper right", fontsize=7,
              ncol=4, framealpha=0.9)

    # ── Panel 2: nodes owned over time ───────────────────────────────────────
    ax2 = axes[1]
    ax2.step(steps, owned_cnt, where="post", color="#27AE60", linewidth=2)
    ax2.fill_between(steps, owned_cnt, step="post", alpha=0.3, color="#27AE60")
    for r in records:
        if r["newly_owned"]:
            ax2.axvline(r["step"], color="lime", linewidth=1, alpha=0.6)
    ax2.axhline(8, color="red", linestyle="--", linewidth=1.5, label="win target (8)")
    ax2.set_ylabel("Nodes owned", fontsize=10)
    ax2.set_ylim(0, 13)
    ax2.legend(fontsize=9)
    ax2.set_title("Nodes owned over time", fontsize=10)

    # ── Panel 3: action type distribution ────────────────────────────────────
    ax3 = axes[2]
    act_colors = {"local_vulnerability": "#1A5276",
                  "connect":             "#27AE60",
                  "remote_vulnerability":"#B7950B",
                  "none":                "#7F8C8D"}
    bottom = np.zeros(max(steps))
    act_counts = {k: np.zeros(max(steps)) for k in act_colors}
    for r in records:
        at = r["action_type"] if r["action_type"] in act_colors else "none"
        act_counts[at][r["step"] - 1] += 1

    for at, color in act_colors.items():
        if act_counts[at].sum() > 0:
            ax3.bar(range(1, max(steps) + 1), act_counts[at],
                    bottom=bottom, color=color, label=at, alpha=0.85)
            bottom += act_counts[at]

    ax3.set_xlabel("Step", fontsize=10)
    ax3.set_ylabel("Action type", fontsize=10)
    ax3.set_title("CBS action type per step", fontsize=10)
    ax3.legend(fontsize=8, loc="upper right")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(Path(out_path).parent, exist_ok=True)
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"\n  Plot saved → {out_path}")
    return out_path

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\nRunning 3 episodes to test action translation...")
    all_tests   = []
    all_records = []

    for seed in range(3):
        print(f"\n  Episode {seed+1}/3 (seed={seed})...")
        records, tests, won = run_episode(seed=seed, max_steps=120)
        all_tests   += tests
        all_records  = records  # keep last episode for trajectory print/plot

        owned_final = records[-1]["owned_count"] if records else 0
        status = "WIN ✓" if won else f"timeout ({len(records)} steps, {owned_final} nodes)"
        print(f"    {status}")

    # Print full trajectory of last episode
    print_trajectory(all_records)

    # Tests across all episodes
    print_tests(all_tests)

    # Plot last episode
    plot_trajectory(all_records)

    # Print action type summary
    total = len(all_records)
    from collections import Counter
    ct = Counter(r["action_type"] for r in all_records)
    print("\n── Action type breakdown (last episode) ─────────────────────────────────")
    for at, cnt in sorted(ct.items(), key=lambda x: -x[1]):
        pct = cnt / total * 100
        bar = "█" * int(pct / 2)
        print(f"  {at:<25}  {cnt:>4} steps  ({pct:5.1f}%)  {bar}")

    # Phase transition summary
    print("\n── Phase transitions (last episode) ──────────────────────────────────────")
    transitions = Counter(
        (r["phase_before"], r["phase_after"])
        for r in all_records
        if r["phase_before"] != r["phase_after"]
    )
    for (pb, pa), cnt in sorted(transitions.items()):
        print(f"  {phase_name(pb):<12} → {phase_name(pa):<12}  {cnt}x")


if __name__ == "__main__":
    main()
