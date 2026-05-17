#!/usr/bin/env python3
"""
Plot DAPN transfer evaluation results from a JSON file produced by
eval/eval_cw_dapn_on_cbs.py --save-json.

Generates a 2×2 figure:
  [A] Mean return ± std          [B] Kill-chain stage distribution
  [C] Return distribution        [D] Stage-4 (impact) reach rate

Usage:
    python3 plot_results.py --results results/eval_results.json
    python3 plot_results.py --results results/eval_results.json --out figures/eval.png
"""

import argparse
import json
import os
import sys
import numpy as np

# ── Plotting imports ──────────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")          # no display needed; works on headless servers
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec
except ImportError:
    print("matplotlib not found. Install with:  pip install matplotlib")
    sys.exit(1)


# ── Style ─────────────────────────────────────────────────────────────────────
CONDITION_COLORS = {
    "random":       "#9e9e9e",   # grey
    "cw_raw→cbs":   "#42a5f5",   # blue
    "cw_dapn→cbs":  "#ef5350",   # red
}
CONDITION_LABELS = {
    "random":       "Random",
    "cw_raw→cbs":   "CW→CBS\n(no DAPN)",
    "cw_dapn→cbs":  "CW→CBS\n(DAPN)",
}
STAGE_COLORS = ["#e0e0e0", "#fff176", "#ffb74d", "#ef9a9a", "#e53935"]
STAGE_NAMES  = ["Stage 0\nnothing", "Stage 1\nrecon",
                "Stage 2\nfoothold", "Stage 3\nlateral", "Stage 4\nimpact"]
FONT_TITLE   = {"fontsize": 11, "fontweight": "bold"}
FONT_LABEL   = {"fontsize": 9}


# ── Panel helpers ─────────────────────────────────────────────────────────────

def _color(label):
    for key, col in CONDITION_COLORS.items():
        if key in label:
            return col
    return "#90a4ae"


def _display_label(label):
    return CONDITION_LABELS.get(label, label)


def panel_mean_return(ax, conditions):
    """Bar chart: mean return ± std per condition."""
    labels  = [_display_label(c["label"]) for c in conditions]
    means   = [c["mean_return"] for c in conditions]
    stds    = [float(np.std(c["returns"])) for c in conditions]
    colors  = [_color(c["label"]) for c in conditions]
    x = np.arange(len(labels))

    bars = ax.bar(x, means, yerr=stds, capsize=6, color=colors,
                  edgecolor="white", linewidth=0.8, error_kw={"elinewidth": 1.5})

    # Annotate bar tops
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + std + abs(max(means, default=1)) * 0.02,
                f"{mean:.1f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, **FONT_LABEL)
    ax.set_ylabel("Episode Return", **FONT_LABEL)
    ax.set_title("A — Mean Return ± Std", **FONT_TITLE)
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def panel_stage_distribution(ax, conditions, n_episodes):
    """Stacked bar chart: episode count at each final stage per condition."""
    labels = [_display_label(c["label"]) for c in conditions]
    x = np.arange(len(labels))
    width = 0.55

    bottoms = np.zeros(len(conditions))
    for stage in range(5):
        counts = np.array([c["stage_dist"].get(stage, 0) for c in conditions], dtype=float)
        bars = ax.bar(x, counts, width, bottom=bottoms,
                      color=STAGE_COLORS[stage], edgecolor="white", linewidth=0.5,
                      label=STAGE_NAMES[stage])
        # Label non-zero segments
        for i, (cnt, bot) in enumerate(zip(counts, bottoms)):
            if cnt > 0:
                ax.text(x[i], bot + cnt / 2, str(int(cnt)),
                        ha="center", va="center", fontsize=7.5, fontweight="bold",
                        color="black" if stage < 3 else "white")
        bottoms += counts

    ax.set_xticks(x)
    ax.set_xticklabels(labels, **FONT_LABEL)
    ax.set_ylabel("Episodes", **FONT_LABEL)
    ax.set_yticks(range(0, n_episodes + 1, max(1, n_episodes // 5)))
    ax.set_title("B — Final Kill-Chain Stage Distribution", **FONT_TITLE)
    ax.legend(loc="upper right", fontsize=7, framealpha=0.7,
              ncol=2, handlelength=1.0, handletextpad=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def panel_return_distribution(ax, conditions):
    """Box plot: per-episode return distribution per condition."""
    data    = [c["returns"] for c in conditions]
    labels  = [_display_label(c["label"]) for c in conditions]
    colors  = [_color(c["label"]) for c in conditions]

    bp = ax.boxplot(data, patch_artist=True, notch=False,
                    medianprops={"color": "black", "linewidth": 1.5},
                    whiskerprops={"linewidth": 1.0},
                    capprops={"linewidth": 1.0},
                    flierprops={"marker": "o", "markersize": 3,
                                "markerfacecolor": "grey", "alpha": 0.6})

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    ax.set_xticklabels(labels, **FONT_LABEL)
    ax.set_ylabel("Episode Return", **FONT_LABEL)
    ax.set_title("C — Return Distribution", **FONT_TITLE)
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def panel_impact_rate(ax, conditions, n_episodes):
    """Bar chart: % of episodes that reached stage 4 (impact)."""
    labels = [_display_label(c["label"]) for c in conditions]
    rates  = [100.0 * c["stage_dist"].get(4, 0) / max(n_episodes, 1) for c in conditions]
    colors = [_color(c["label"]) for c in conditions]
    x = np.arange(len(labels))

    bars = ax.bar(x, rates, color=colors, edgecolor="white", linewidth=0.8)

    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.0,
                f"{rate:.0f}%", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, **FONT_LABEL)
    ax.set_ylabel("Episodes reaching Impact (%)", **FONT_LABEL)
    ax.set_ylim(0, 110)
    ax.set_title("D — Stage-4 (Impact) Reach Rate", **FONT_TITLE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ── Main ──────────────────────────────────────────────────────────────────────

def plot(results_path: str, out_path: str):
    with open(results_path) as f:
        data = json.load(f)

    conditions  = data["conditions"]
    n_episodes  = data.get("n_episodes", max(len(c["returns"]) for c in conditions))

    # JSON serialises dict keys as strings; normalise stage_dist keys back to int
    for c in conditions:
        c["stage_dist"] = {int(k): v for k, v in c["stage_dist"].items()}

    if not conditions:
        print("No conditions found in results file.")
        sys.exit(1)

    # ── Layout ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(13, 9))
    fig.patch.set_facecolor("#fafafa")
    gs = GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32,
                  left=0.08, right=0.97, top=0.90, bottom=0.08)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    for ax in (ax_a, ax_b, ax_c, ax_d):
        ax.set_facecolor("#f5f5f5")

    panel_mean_return(ax_a, conditions)
    panel_stage_distribution(ax_b, conditions, n_episodes)
    panel_return_distribution(ax_c, conditions)
    panel_impact_rate(ax_d, conditions, n_episodes)

    # ── Legend strip for conditions ───────────────────────────────────────────
    handles = [
        mpatches.Patch(color=col, label=_display_label(key).replace("\n", " "))
        for key, col in CONDITION_COLORS.items()
        if any(c["label"] == key for c in conditions)
    ]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles),
               framealpha=0.8, fontsize=9,
               bbox_to_anchor=(0.5, 0.97))

    # ── Title ─────────────────────────────────────────────────────────────────
    fig.suptitle(
        f"CW → CBS Zero-Shot Transfer  |  {n_episodes} episodes per condition",
        fontsize=13, fontweight="bold", y=1.01
    )

    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved to {out_path}")

    # Also try to show interactively if a display is available
    try:
        matplotlib.use("TkAgg")
        plt.show()
    except Exception:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True,
                        help="Path to JSON file from eval_cw_dapn_on_cbs.py --save-json")
    parser.add_argument("--out", default="figures/eval_transfer.png",
                        help="Output image path (PNG/PDF/SVG)")
    args = parser.parse_args()

    if not os.path.exists(args.results):
        print(f"Results file not found: {args.results}")
        sys.exit(1)

    plot(args.results, args.out)
