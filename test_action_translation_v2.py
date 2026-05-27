"""
Test and visualise the v2 kill-chain action translation layer.

V2 resolves CBS actions from REAL CBS state (no phase counter):
  Case 1: target node owned        → local_vulnerability(node_id, ...)
  Case 2: creds in CBS cache       → connect(src → node_id, port, cred)
  Case 3: no creds, not owned      → local_vulnerability(frontier, ...)

Checks:
  T1: target owned      → local_vulnerability where src == node_id
  T2: creds in cache    → connect to that node
  T3: connect src must be owned before step
  T4: no creds + not owned → local_vulnerability (probe)
  T5: newly owned nodes had creds in cache before step (connect fired)

Output:
  - Pass/fail results printed to terminal
  - Trajectory table (slot, CBS state, action)
  - results/trajectory_plot_v2.png

Usage:
  PYTHONPATH=$PWD:$PWD/cyberwheel:$PWD/CyberBattleSim python3.10 test_action_translation_v2.py
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
from adapters.unified_env_v2 import UnifiedSecEnvV2
from adapters.unified_env import MAX_SLOTS
from config.env_builders import make_cbs_env

# ── State label helpers ───────────────────────────────────────────────────────
STATE_NAMES  = ["probe", "has_creds", "owned"]
STATE_COLORS = ["#2C3E50", "#B7950B", "#27AE60"]

def state_name(s):
    return STATE_NAMES[s] if s < len(STATE_NAMES) else f"state{s}"

def state_color(s):
    return STATE_COLORS[s] if s < len(STATE_COLORS) else "#7F8C8D"

# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_unified(env):
    e = env
    while e is not None:
        if isinstance(e, UnifiedSecEnvV2):
            return e
        e = getattr(e, "env", None)
    return None

def _priv(unified):
    raw = getattr(unified, "_raw_obs", None)
    if not isinstance(raw, dict): return None
    p = raw.get("nodes_privilegelevel")
    return np.asarray(p, dtype=np.int32) if p is not None else None

def _cred_nodes(unified):
    """Return set of node IDs with credentials in the CBS cache."""
    e = unified
    if e is None: return set()
    return e._get_cred_nodes()

def _owned_set(unified):
    p = _priv(unified)
    if p is None: return set()
    return {i for i in range(p.size) if p[i] >= 1}

def _node_state(node_id, owned, creds):
    """0=probe, 1=has_creds, 2=owned"""
    if node_id in owned: return 2
    if node_id in creds: return 1
    return 0

# ── Test runner ───────────────────────────────────────────────────────────────

def run_episode(seed=0, max_steps=150):
    env     = TimeLimit(UnifiedSecEnvV2("cbs", cbs_factory=make_cbs_env),
                        max_episode_steps=max_steps)
    unified = _get_unified(env)

    obs, _ = env.reset(seed=seed)

    records = []
    tests   = []
    won     = False

    prev_owned = set()
    prev_creds = set()

    for step in range(max_steps):
        slot = np.random.randint(0, MAX_SLOTS)

        # Snapshot state BEFORE step
        owned_before = _owned_set(unified)
        creds_before = _cred_nodes(unified)

        slot_map = getattr(unified, "_slot_map", [])
        node_id  = slot_map[slot] if slot < len(slot_map) else -1
        node_state_before = _node_state(node_id, owned_before, creds_before)

        obs, r, terminated, truncated, info = env.step(slot)

        owned_after  = _owned_set(unified)
        creds_after  = _cred_nodes(unified)
        newly_owned  = owned_after - owned_before

        cbs_action  = info.get("cbs_action", {})
        action_type = list(cbs_action.keys())[0]  if cbs_action else "none"
        action_args = list(cbs_action.values())[0] if cbs_action else ()

        # ── Tests ────────────────────────────────────────────────────────────

        # T1: target was owned → action must be local_vulnerability on that node
        if node_id in owned_before:
            ok = (action_type == "local_vulnerability")
            tests.append((
                f"step {step+1:3d} slot {slot} node {node_id} owned → local_vuln",
                ok, action_type
            ))

        # T2: creds exist for target, target not yet owned → connect
        elif node_id in creds_before:
            ok = (action_type == "connect")
            tests.append((
                f"step {step+1:3d} slot {slot} node {node_id} has creds → connect",
                ok, action_type
            ))

        # T4: no creds, not owned → local_vulnerability (probe frontier)
        else:
            ok = (action_type == "local_vulnerability")
            tests.append((
                f"step {step+1:3d} slot {slot} node {node_id} no creds → local_vuln",
                ok, action_type
            ))

        # T3: connect src must have been owned before step
        if action_type == "connect":
            src = int(action_args[0])
            src_was_owned = (src in owned_before) or (src == 0)
            ok = src_was_owned
            tests.append((
                f"step {step+1:3d} connect src={src} was owned before step",
                ok, f"src_owned={src_was_owned}"
            ))

        # T5: newly owned node must have had creds in cache before connect fired
        for nid in newly_owned:
            ok = (nid in creds_before) or (action_type == "connect" and int(action_args[1]) == nid)
            tests.append((
                f"step {step+1:3d} node {nid} owned → creds existed before",
                ok, f"had_creds={nid in creds_before}"
            ))

        records.append({
            "step":         step + 1,
            "slot":         slot,
            "node":         node_id,
            "node_state":   node_state_before,   # 0=probe,1=creds,2=owned
            "action_type":  action_type,
            "action_args":  action_args,
            "reward":       r,
            "owned_count":  len(owned_after - {0}),
            "newly_owned":  sorted(newly_owned),
            "cred_count":   len(creds_after),
        })

        prev_owned = owned_after
        prev_creds = creds_after

        if terminated:
            won = True
            break
        if truncated:
            break

    env.close()
    return records, tests, won

# ── Print trajectory ──────────────────────────────────────────────────────────

def print_trajectory(records):
    print("\n" + "═"*100)
    print("  TRAJECTORY  (v2 — real CBS state action translation)")
    print("═"*100)
    print(f"  {'Step':>4}  {'Slot':>4}  {'Node':>4}  {'State':<10}  "
          f"{'CBS Action':<45}  {'Owned':>5}  {'Creds':>5}  {'Reward':>7}")
    print(f"  {'----':>4}  {'----':>4}  {'----':>4}  {'----------':<10}  "
          f"{'---------------------------------------------':<45}  {'-----':>5}  {'-----':>5}  {'-------':>7}")

    for r in records:
        state_str = state_name(r["node_state"])

        at = r["action_type"]
        aa = r["action_args"]
        if at == "local_vulnerability":
            act = f"local_vuln  (node={aa[0]}, vuln_idx={aa[1]})"
        elif at == "connect":
            act = f"connect     (src={aa[0]}→tgt={aa[1]}, port={aa[2]}, cred={aa[3]})"
        elif at == "remote_vulnerability":
            act = f"remote_vuln (src={aa[0]}→tgt={aa[1]}, vuln={aa[2]})"
        else:
            act = str(at)

        new_tag = f"  ◀ {r['newly_owned']} OWNED" if r["newly_owned"] else ""
        print(f"  {r['step']:>4}  {r['slot']:>4}  {r['node']:>4}  {state_str:<10}  "
              f"{act:<45}  {r['owned_count']:>5}  {r['cred_count']:>5}  "
              f"{r['reward']:>7.1f}{new_tag}")

# ── Print test results ────────────────────────────────────────────────────────

def print_tests(tests):
    print("\n" + "═"*70)
    print("  V2 ACTION TRANSLATION TESTS")
    print("═"*70)
    passed = sum(1 for _, ok, _ in tests if ok)
    total  = len(tests)
    for desc, ok, detail in tests:
        if not ok:
            print(f"  ✗  FAIL  {desc}  [got: {detail}]")
    print(f"\n  Result: {passed}/{total} passed  "
          f"({'ALL PASS ✓' if passed == total else f'FAILURES: {total-passed}'})")

# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_trajectory(records, out_path="results/trajectory_plot_v2.png"):
    steps      = [r["step"]       for r in records]
    slots      = [r["slot"]       for r in records]
    states     = [r["node_state"] for r in records]
    act_types  = [r["action_type"] for r in records]
    owned_cnt  = [r["owned_count"] for r in records]
    cred_cnt   = [r["cred_count"]  for r in records]
    rewards    = [r["reward"]      for r in records]

    fig, axes = plt.subplots(3, 1, figsize=(16, 12),
                             gridspec_kw={"height_ratios": [4, 1.5, 1.5]})
    fig.suptitle("V2 Action Translation — Real CBS State Policy Trajectory",
                 fontsize=14, fontweight="bold", y=0.98)

    # ── Panel 1: node state heatmap (slot × step) ─────────────────────────────
    ax = axes[0]
    state_grid = np.full((MAX_SLOTS, max(steps) + 1), np.nan)
    atype_grid = {}

    for r in records:
        s, sl, st = r["step"], r["slot"], r["node_state"]
        state_grid[sl, s] = st
        atype_grid[(sl, s)] = r["action_type"]

    from matplotlib.colors import ListedColormap, BoundaryNorm
    cmap = ListedColormap(STATE_COLORS)
    norm = BoundaryNorm(boundaries=[-0.5, 0.5, 1.5, 2.5], ncolors=3)

    im = ax.imshow(state_grid[:, 1:], aspect="auto", cmap=cmap, norm=norm,
                   origin="lower", interpolation="nearest",
                   extent=[1, max(steps), -0.5, MAX_SLOTS - 0.5])

    # Mark connect actions with white star
    for (sl, st), at in atype_grid.items():
        if at == "connect":
            ax.plot(st, sl, "w*", markersize=10, zorder=5)

    # Green line at ownership events
    for r in records:
        if r["newly_owned"]:
            ax.axvline(r["step"], color="lime", linewidth=1.5, alpha=0.7, zorder=4)

    ax.set_ylabel("Slot (0–11)", fontsize=11)
    ax.set_title("Node state per slot at each step  "
                 "(★ = connect,  green = ownership event)", fontsize=10)
    ax.set_yticks(range(MAX_SLOTS))
    ax.set_yticklabels([f"slot {i}" for i in range(MAX_SLOTS)], fontsize=7)

    legend_patches = [mpatches.Patch(color=STATE_COLORS[i], label=STATE_NAMES[i])
                      for i in range(len(STATE_NAMES))]
    ax.legend(handles=legend_patches, loc="upper right", fontsize=9,
              ncol=3, framealpha=0.9)

    # ── Panel 2: nodes owned + creds discovered over time ─────────────────────
    ax2 = axes[1]
    ax2.step(steps, owned_cnt, where="post", color="#27AE60", linewidth=2, label="nodes owned")
    ax2.step(steps, cred_cnt,  where="post", color="#B7950B", linewidth=1.5,
             linestyle="--", label="creds in cache")
    ax2.fill_between(steps, owned_cnt, step="post", alpha=0.3, color="#27AE60")
    for r in records:
        if r["newly_owned"]:
            ax2.axvline(r["step"], color="lime", linewidth=1, alpha=0.6)
    ax2.axhline(8, color="red", linestyle="--", linewidth=1.5, label="win target (8)")
    ax2.set_ylabel("Count", fontsize=10)
    ax2.set_ylim(0, 13)
    ax2.legend(fontsize=9)
    ax2.set_title("Nodes owned & credentials discovered over time", fontsize=10)

    # ── Panel 3: action type distribution ─────────────────────────────────────
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
    print("\nRunning 3 episodes to test V2 action translation...")
    all_tests   = []
    all_records = []

    for seed in range(3):
        print(f"\n  Episode {seed+1}/3 (seed={seed})...")
        records, tests, won = run_episode(seed=seed, max_steps=120)
        all_tests   += tests
        all_records  = records

        owned_final = records[-1]["owned_count"] if records else 0
        status = "WIN ✓" if won else f"timeout ({len(records)} steps, {owned_final} nodes)"
        print(f"    {status}")

    print_trajectory(all_records)
    print_tests(all_tests)
    plot_trajectory(all_records)

    # Action type summary
    total = len(all_records)
    from collections import Counter
    ct = Counter(r["action_type"] for r in all_records)
    print("\n── Action type breakdown (last episode) ─────────────────────────────────")
    for at, cnt in sorted(ct.items(), key=lambda x: -x[1]):
        pct = cnt / total * 100
        bar = "█" * int(pct / 2)
        print(f"  {at:<25}  {cnt:>4} steps  ({pct:5.1f}%)  {bar}")

    # State transition summary
    print("\n── V2 state transitions (last episode) ──────────────────────────────────")
    transitions = Counter(
        (r["node_state"], all_records[i+1]["node_state"] if i+1 < len(all_records) else r["node_state"])
        for i, r in enumerate(all_records)
        if i+1 < len(all_records) and all_records[i+1]["slot"] == r["slot"]
           and all_records[i+1]["node_state"] != r["node_state"]
    )
    for (sb, sa), cnt in sorted(transitions.items()):
        print(f"  {state_name(sb):<12} → {state_name(sa):<12}  {cnt}x")

    # CBS state → action mapping verification
    print("\n── CBS state → action mapping (v2 logic) ────────────────────────────────")
    state_action = Counter((r["node_state"], r["action_type"]) for r in all_records)
    for (st, at), cnt in sorted(state_action.items(), key=lambda x: -x[1]):
        pct = cnt / total * 100
        expected = {0: "local_vulnerability", 1: "connect", 2: "local_vulnerability"}
        mark = "✓" if at == expected.get(st) else "✗"
        print(f"  {mark}  state={state_name(st):<10}  action={at:<25}  {cnt:>4}x  ({pct:5.1f}%)")


if __name__ == "__main__":
    main()
