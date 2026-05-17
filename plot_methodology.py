#!/usr/bin/env python3
"""
Generate a methodology flowchart for the DAPN transfer learning pipeline.

Usage:
    python3 plot_methodology.py
    python3 plot_methodology.py --out figures/methodology.png
"""

import argparse, os, sys

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    import matplotlib.patheffects as pe
except ImportError:
    print("matplotlib not found.  pip install matplotlib"); sys.exit(1)

# ── Palette ───────────────────────────────────────────────────────────────────
CW     = "#1565C0"   ;  CW_BG     = "#E3F2FD"
CBS    = "#2E7D32"   ;  CBS_BG    = "#E8F5E9"
DAPN   = "#6A1B9A"   ;  DAPN_BG   = "#F3E5F5"
TRAIN  = "#BF360C"   ;  TRAIN_BG  = "#FBE9E7"
EVAL   = "#880E4F"   ;  EVAL_BG   = "#FCE4EC"
OUT    = "#37474F"   ;  OUT_BG    = "#ECEFF1"
PH     = "#263238"   ;  PH_BG     = "#CFD8DC"


def rbox(ax, cx, cy, w, h, title, sub=None,
         fc="white", ec="black", lw=1.6, r=0.012,
         tsize=9, ssize=7.5, tbold=True):
    """Rounded box.  cx/cy = centre coords in axes (0-1 space)."""
    rx, ry = cx - w/2, cy - h/2
    patch = FancyBboxPatch((rx, ry), w, h,
                           boxstyle=f"round,pad=0,rounding_size={r}",
                           fc=fc, ec=ec, lw=lw, zorder=3,
                           transform=ax.transAxes, clip_on=False)
    ax.add_patch(patch)
    tw = "bold" if tbold else "normal"
    yoff = h * 0.14 if sub else 0
    ax.text(cx, cy + yoff, title,
            ha="center", va="center", fontsize=tsize,
            fontweight=tw, color=ec, zorder=4, transform=ax.transAxes)
    if sub:
        ax.text(cx, cy - h * 0.22, sub,
                ha="center", va="center", fontsize=ssize,
                color=ec, alpha=0.85, zorder=4, transform=ax.transAxes,
                style="italic")


def arr(ax, x0, y0, x1, y1, color, lw=1.5, rad=0.0):
    """Arrow in axes-fraction coords."""
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=dict(
                    arrowstyle="-|>,head_length=0.018,head_width=0.010",
                    color=color, lw=lw, shrinkA=4, shrinkB=4,
                    connectionstyle=f"arc3,rad={rad}"),
                zorder=5)


def ph_banner(ax, y, label, color):
    """Phase header banner spanning full width."""
    ax.text(0.015, y, label,
            ha="left", va="center", fontsize=9, fontweight="bold",
            color="white", zorder=6, transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.32", fc=color, ec=color, lw=0))


def hline(ax, y):
    ax.axhline(y, color="#90A4AE", lw=0.8, ls="--",
               xmin=0.0, xmax=1.0, zorder=2)


# ── Main ──────────────────────────────────────────────────────────────────────

def build(out_path):
    fig = plt.figure(figsize=(14, 20))
    ax  = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    fig.patch.set_facecolor("#F8F9FA")
    ax.set_facecolor("#F8F9FA")

    # ── Title ─────────────────────────────────────────────────────────────────
    ax.text(0.5, 0.977, "DAPN Zero-Shot Transfer — Full Methodology",
            ha="center", va="center", fontsize=15, fontweight="bold",
            color="#1A237E", transform=ax.transAxes)
    ax.text(0.5, 0.962, "CyberWheel  (source domain)   →   CyberBattleSim  (target domain)",
            ha="center", va="center", fontsize=10.5, color="#455A64",
            transform=ax.transAxes)

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 1  Data Collection
    # ══════════════════════════════════════════════════════════════════════════
    hline(ax, 0.946); ph_banner(ax, 0.935, "  PHASE 1 — Data Collection", CW)

    # Envs
    rbox(ax, 0.26, 0.905, 0.36, 0.042,
         "CyberWheel Environment",
         "15-host network  |  rl_red_agent.yaml  |  action space: 7 unified actions",
         fc=CW_BG, ec=CW)
    rbox(ax, 0.74, 0.905, 0.36, 0.042,
         "CyberBattleSim Environment",
         "CyberBattleChain-v0  |  size=6  |  goal: own ≥50 % nodes",
         fc=CBS_BG, ec=CBS)

    # Directed heuristic
    rbox(ax, 0.26, 0.848, 0.34, 0.038,
         "Directed Heuristic Policy  (CW)",
         "ping_sweep → port_scan → service_disc → move → escalate → impact",
         fc=CW_BG, ec=CW, tsize=8.5, ssize=7)
    rbox(ax, 0.74, 0.848, 0.34, 0.038,
         "Directed Heuristic Policy  (CBS)",
         "local_vuln → connect → remote_vuln  (stage-aware priority)",
         fc=CBS_BG, ec=CBS, tsize=8.5, ssize=7)

    arr(ax, 0.26, 0.883, 0.26, 0.868, CW)
    arr(ax, 0.74, 0.883, 0.74, 0.868, CBS)

    # Raw obs
    rbox(ax, 0.26, 0.793, 0.34, 0.038,
         'Raw CW Observations  (2 000)',
         '{"blue": …, "red": obs_vec}  —  kill-chain stages 0 – 4',
         fc=CW_BG, ec=CW, tsize=8.5, ssize=7)
    rbox(ax, 0.74, 0.793, 0.34, 0.038,
         'Raw CBS Observations  (2 000)',
         'dict(discovered, owned, creds, priv_level)  —  stages 0 – 4',
         fc=CBS_BG, ec=CBS, tsize=8.5, ssize=7)

    arr(ax, 0.26, 0.829, 0.26, 0.813, CW)
    arr(ax, 0.74, 0.829, 0.74, 0.813, CBS)

    # Stage dist notes
    rbox(ax, 0.26, 0.745, 0.33, 0.034,
         "Stage distribution",
         "s0: 2%  s1: 22%  s2: 6%  s3: 2%  s4: 68%",
         fc=CW_BG, ec=CW, tsize=8, ssize=7, tbold=False)
    rbox(ax, 0.74, 0.745, 0.33, 0.034,
         "Stage distribution",
         "s0: <1%  s1: 22%  s2: 31%  s3: 46%  s4: <1%",
         fc=CBS_BG, ec=CBS, tsize=8, ssize=7, tbold=False)

    arr(ax, 0.26, 0.774, 0.26, 0.762, CW)
    arr(ax, 0.74, 0.774, 0.74, 0.762, CBS)

    # NPZ file
    rbox(ax, 0.50, 0.703, 0.26, 0.038,
         "data/obs.npz",
         "source_obs (CW) + target_obs (CBS)",
         fc=DAPN_BG, ec=DAPN, tbold=True)

    arr(ax, 0.26, 0.728, 0.39, 0.703, CW,  lw=1.4)
    arr(ax, 0.74, 0.728, 0.61, 0.703, CBS, lw=1.4)

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 2  DAPN Encoder Training
    # ══════════════════════════════════════════════════════════════════════════
    hline(ax, 0.682); ph_banner(ax, 0.671, "  PHASE 2 — DAPN Encoder Training", DAPN)

    # Preprocessors
    rbox(ax, 0.26, 0.641, 0.35, 0.038,
         "UnifiedFullObsPreprocessor  (CW)",
         "raw obs_vec  →  z-score normalised  →  512-D",
         fc=CW_BG, ec=CW, tsize=8.5, ssize=7)
    rbox(ax, 0.74, 0.641, 0.35, 0.038,
         "UnifiedFullObsPreprocessor  (CBS)",
         "raw obs dict  →  z-score normalised  →  512-D",
         fc=CBS_BG, ec=CBS, tsize=8.5, ssize=7)

    arr(ax, 0.39, 0.684, 0.28, 0.660, CW,  lw=1.3)
    arr(ax, 0.61, 0.684, 0.72, 0.660, CBS, lw=1.3)

    # Pairing
    rbox(ax, 0.50, 0.591, 0.40, 0.036,
         "Stage-Based Semantic Pairing",
         "match CW ↔ CBS samples by kill-chain stage  (0 – 4)",
         fc=DAPN_BG, ec=DAPN)

    arr(ax, 0.26, 0.622, 0.38, 0.591, CW,  lw=1.3)
    arr(ax, 0.74, 0.622, 0.62, 0.591, CBS, lw=1.3)

    # DANN block
    rbox(ax, 0.50, 0.525, 0.46, 0.076,
         "Domain-Adversarial Neural Network  (DANN)",
         fc=DAPN_BG, ec=DAPN, tsize=10, tbold=True)
    # inner detail rows
    for dy, txt, clr in [
        ( 0.010, "Shared Encoder  512D → 256D  (3-layer MLP)",   DAPN),
        (-0.012, "Stage Classifier head  (cross-entropy loss, primary objective)", DAPN),
        (-0.034, "Domain Discriminator head  (adversarial, Gradient Reversal Layer)", DAPN),
    ]:
        ax.text(0.50, 0.525 + dy, txt,
                ha="center", va="center", fontsize=7.5, color=clr,
                alpha=0.9, zorder=4, transform=ax.transAxes)

    arr(ax, 0.50, 0.573, 0.50, 0.565, DAPN)

    # Frozen encoder output
    rbox(ax, 0.50, 0.455, 0.36, 0.038,
         "Frozen DAPN Encoder   (512D → 256D)",
         "dapn_encoder.pt  — domain-invariant representation",
         fc=DAPN_BG, ec=DAPN, tbold=True)

    arr(ax, 0.50, 0.487, 0.50, 0.474, DAPN)

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 3  Policy Training
    # ══════════════════════════════════════════════════════════════════════════
    hline(ax, 0.436); ph_banner(ax, 0.425, "  PHASE 3 — Policy Training  (CyberWheel only)", TRAIN)

    # Two branches
    rbox(ax, 0.27, 0.390, 0.38, 0.040,
         "Condition 2 — Raw 8D  (no domain adaptation)",
         "CW env → ObservationTranslator → 8-D unified obs  |  no encoder",
         fc=TRAIN_BG, ec=TRAIN, tsize=8.5, ssize=7)
    rbox(ax, 0.73, 0.390, 0.38, 0.040,
         "Condition 3 — DAPN 256D  (with domain adaptation)",
         "CW env → Preprocessor → 512D → frozen DAPN enc → 256D",
         fc=TRAIN_BG, ec=TRAIN, tsize=8.5, ssize=7)

    # Arrow from encoder → cond 3
    arr(ax, 0.50, 0.436, 0.73, 0.410, DAPN, lw=1.2, rad=-0.15)
    # Arrow cond 2 from top (no encoder path)
    ax.annotate("", xy=(0.27, 0.410), xytext=(0.27, 0.436),
                xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>,head_length=0.018,head_width=0.010",
                                color=TRAIN, lw=1.4, shrinkA=4, shrinkB=4), zorder=5)

    rbox(ax, 0.27, 0.338, 0.34, 0.038,
         "PPO  (MultiInputPolicy)",
         'obs: {"obs": 8D, "mask": 7D}  |  200 000 steps  |  seed 42',
         fc=TRAIN_BG, ec=TRAIN, tsize=8.5, ssize=7)
    rbox(ax, 0.73, 0.338, 0.34, 0.038,
         "PPO  (MultiInputPolicy)",
         'obs: {"obs": 256D, "mask": 7D}  |  200 000 steps  |  seed 42',
         fc=TRAIN_BG, ec=TRAIN, tsize=8.5, ssize=7)

    arr(ax, 0.27, 0.370, 0.27, 0.357, TRAIN)
    arr(ax, 0.73, 0.370, 0.73, 0.357, TRAIN)

    rbox(ax, 0.27, 0.292, 0.28, 0.034,
         "cw_raw_policy.zip",
         "frozen  |  8-D obs interface",
         fc=TRAIN_BG, ec=TRAIN, tbold=True, tsize=9, ssize=7.5)
    rbox(ax, 0.73, 0.292, 0.28, 0.034,
         "cw_dapn_policy.zip",
         "frozen  |  256-D obs interface",
         fc=TRAIN_BG, ec=TRAIN, tbold=True, tsize=9, ssize=7.5)

    arr(ax, 0.27, 0.319, 0.27, 0.309, TRAIN)
    arr(ax, 0.73, 0.319, 0.73, 0.309, TRAIN)

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 4  Zero-Shot Evaluation
    # ══════════════════════════════════════════════════════════════════════════
    hline(ax, 0.272); ph_banner(ax, 0.261, "  PHASE 4 — Zero-Shot Evaluation  (CyberBattleSim)", EVAL)

    # Three condition boxes
    rbox(ax, 0.15, 0.224, 0.22, 0.042,
         "Condition 1",
         "Random baseline\nCBS action_space.sample()",
         fc=EVAL_BG, ec=EVAL, tsize=9, ssize=7.5)
    rbox(ax, 0.50, 0.224, 0.24, 0.042,
         "Condition 2",
         "8-D raw transfer\nObsTranslator only,  no encoder",
         fc=EVAL_BG, ec=EVAL, tsize=9, ssize=7.5)
    rbox(ax, 0.85, 0.224, 0.24, 0.042,
         "Condition 3",
         "DAPN 256-D transfer\nfrozen enc  +  cw_dapn_policy",
         fc=EVAL_BG, ec=EVAL, tsize=9, ssize=7.5)

    # Policy arrows → conditions
    arr(ax, 0.27, 0.275, 0.50, 0.245, EVAL, lw=1.2)
    arr(ax, 0.73, 0.275, 0.85, 0.245, EVAL, lw=1.2)
    # Encoder dashed arrow → cond 3
    ax.annotate("", xy=(0.85, 0.245), xytext=(0.68, 0.455),
                xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>,head_length=0.015,head_width=0.008",
                                color=DAPN, lw=1.1, ls="dashed",
                                connectionstyle="arc3,rad=-0.28",
                                shrinkA=4, shrinkB=4), zorder=4)
    # Random baseline arrow
    ax.annotate("", xy=(0.15, 0.245), xytext=(0.15, 0.261),
                xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>,head_length=0.018,head_width=0.010",
                                color=EVAL, lw=1.4, shrinkA=4, shrinkB=4), zorder=5)

    # CBS eval env
    rbox(ax, 0.50, 0.172, 0.58, 0.038,
         "CyberBattleChain-v0   via   UnifiedSecEnv / DAPNEnvWrapper",
         "20 episodes × 500 steps  |  terminated=True ⟹ stage 4  (attacker wins)",
         fc=CBS_BG, ec=CBS, tsize=8.5, ssize=7)

    arr(ax, 0.15, 0.203, 0.23, 0.172, EVAL, lw=1.2)
    arr(ax, 0.50, 0.203, 0.50, 0.191, EVAL, lw=1.2)
    arr(ax, 0.85, 0.203, 0.77, 0.172, EVAL, lw=1.2)

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 5  Results
    # ══════════════════════════════════════════════════════════════════════════
    hline(ax, 0.152); ph_banner(ax, 0.141, "  PHASE 5 — Results & Visualisation", OUT)

    rbox(ax, 0.50, 0.111, 0.30, 0.036,
         "results/eval_results.json",
         "returns, stage_dist, mean_steps  per condition",
         fc=OUT_BG, ec=OUT, tbold=True)

    arr(ax, 0.50, 0.153, 0.50, 0.129, OUT)

    panels = [
        (0.13, "A\nMean Return\n± Std"),
        (0.38, "B\nKill-Chain\nStage Dist"),
        (0.62, "C\nReturn\nDistribution"),
        (0.87, "D\nStage-4\nImpact Rate"),
    ]
    for px, label in panels:
        rbox(ax, px, 0.053, 0.20, 0.052, label,
             fc=OUT_BG, ec=OUT, tsize=8, tbold=False, r=0.010)
        arr(ax, 0.50, 0.093, px, 0.079, OUT, lw=1.1)

    # ── Legend strip ──────────────────────────────────────────────────────────
    legend = [
        (CW,    CW_BG,    "CyberWheel (source)"),
        (CBS,   CBS_BG,   "CyberBattleSim (target)"),
        (DAPN,  DAPN_BG,  "Shared / DAPN encoder"),
        (TRAIN, TRAIN_BG, "Policy training  (CW)"),
        (EVAL,  EVAL_BG,  "Zero-shot evaluation  (CBS)"),
        (OUT,   OUT_BG,   "Results & outputs"),
    ]
    lx0, ly = 0.015, 0.020
    for i, (ec, fc, lbl) in enumerate(legend):
        xi = lx0 + i * 0.163
        sq = FancyBboxPatch((xi, ly - 0.009), 0.011, 0.018,
                            boxstyle="round,pad=0,rounding_size=0.004",
                            fc=fc, ec=ec, lw=1.3, zorder=3,
                            transform=ax.transAxes, clip_on=False)
        ax.add_patch(sq)
        ax.text(xi + 0.014, ly, lbl, va="center",
                fontsize=7.5, color=ec, fontweight="bold",
                transform=ax.transAxes)

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"Saved → {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures/methodology.png")
    build(ap.parse_args().out)
