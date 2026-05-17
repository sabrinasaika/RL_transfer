"""
Publication-quality architecture diagram for DAPN transfer learning.
Saved to figures/architecture.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import os

# ── Canvas ─────────────────────────────────────────────────────────────────────
W, H = 20, 11
fig, ax = plt.subplots(figsize=(W, H))
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")
fig.patch.set_facecolor("white")

# ── Colour palette ──────────────────────────────────────────────────────────────
C = dict(
    cw      = "#D6EAF8",
    cbs     = "#D5F5E3",
    prep    = "#EAF2FF",
    enc     = "#E8DAEF",
    grl     = "#FADBD8",
    disc    = "#FDEBD0",
    stage   = "#FEF9E7",
    policy  = "#F2F3F4",
    frozen  = "#D5D8DC",
    white   = "#FFFFFF",
    edge    = "#2C3E50",
    blue    = "#2471A3",
    green   = "#1E8449",
    red     = "#C0392B",
    purple  = "#7D3C98",
    orange  = "#CA6F1E",
)

# ── Helpers ─────────────────────────────────────────────────────────────────────

def rbox(cx, cy, w, h, line1, line2=None, fc="white", ec=None,
         fs=8.5, bold=False, dashed=False, lw=1.4):
    ec = ec or C["edge"]
    patch = FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle="round,pad=0.1",
        facecolor=fc, edgecolor=ec,
        linewidth=lw, linestyle="--" if dashed else "-",
        zorder=2,
    )
    ax.add_patch(patch)
    fw = "bold" if bold else "normal"
    if line2:
        ax.text(cx, cy + h*0.17, line1, ha="center", va="center",
                fontsize=fs, fontweight=fw, color=C["edge"], zorder=3)
        ax.text(cx, cy - h*0.22, line2, ha="center", va="center",
                fontsize=fs - 1.5, color="#555", style="italic", zorder=3)
    else:
        ax.text(cx, cy, line1, ha="center", va="center",
                fontsize=fs, fontweight=fw, color=C["edge"], zorder=3)


def arr(x1, y1, x2, y2, label=None, color=None, lw=1.5, rad=0.0, ls="-"):
    color = color or C["edge"]
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="-|>", color=color, lw=lw,
            linestyle=ls,
            connectionstyle=f"arc3,rad={rad}",
            mutation_scale=11,
        ),
        zorder=4,
    )
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx, my + (0.13 if rad == 0 else 0),
                label, ha="center", va="bottom",
                fontsize=7, color=color, fontweight="bold", zorder=5)


def phase_box(x, y, w, h, title, color):
    """Dashed section rectangle with title inside at top."""
    bg = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                         facecolor=color, edgecolor=color,
                         linewidth=2, linestyle="--", alpha=0.07, zorder=0)
    ax.add_patch(bg)
    border = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                             facecolor="none", edgecolor=color,
                             linewidth=1.8, linestyle="--", zorder=1)
    ax.add_patch(border)
    # Title banner inside top of box
    banner = FancyBboxPatch((x + 0.15, y + h - 0.55), w - 0.3, 0.45,
                             boxstyle="round,pad=0.05",
                             facecolor=color, edgecolor=color,
                             alpha=0.35, linewidth=0, zorder=1)
    ax.add_patch(banner)
    ax.text(x + w/2, y + h - 0.32, title, ha="center", va="center",
            fontsize=9.5, fontweight="bold", color=color, zorder=5)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN TITLE
# ══════════════════════════════════════════════════════════════════════════════
ax.text(W/2, 10.65,
        "Domain-Adversarial Policy Network (DAPN) — System Architecture",
        ha="center", va="center", fontsize=13, fontweight="bold", color=C["edge"])

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — DANN Encoder Training   y: 5.5 → 10.3
# ══════════════════════════════════════════════════════════════════════════════
phase_box(0.3, 5.5, 19.4, 4.6,
          "Phase 1 — Domain-Adversarial Encoder Training (DANN)", C["blue"])

# Domain side-labels
ax.text(1.15, 9.1,  "Source\n(CW)",  ha="center", fontsize=8,
        color=C["blue"],  fontweight="bold")
ax.text(1.15, 7.05, "Target\n(CBS)", ha="center", fontsize=8,
        color=C["green"], fontweight="bold")

# ── Observations ────────────────────────────────────────────────────────────
rbox(2.85, 9.1,  2.3, 0.65, "CyberWheel Obs",
     line2='dict  {"red": vec,  "blue": vec}', fc=C["cw"], fs=8)
rbox(2.85, 7.05, 2.3, 0.65, "CyberBattleSim Obs",
     line2="dict  {nodes, creds, priv_level …}", fc=C["cbs"], fs=8)

# ── Preprocessor ─────────────────────────────────────────────────────────────
rbox(6.0, 8.1, 2.2, 0.8, "Unified Obs\nPreprocessor",
     line2="both domains  →  512-D", fc=C["prep"], bold=True, fs=9)
arr(4.0, 9.1, 4.9, 8.45)     # CW  → prep
arr(4.0, 7.05, 4.9, 7.75)    # CBS → prep

# ── Shared Encoder ───────────────────────────────────────────────────────────
rbox(9.1, 8.1, 2.3, 0.9, "Shared Encoder",
     line2="MLP   512-D  →  256-D",
     fc=C["enc"], bold=True, lw=2.0, fs=9.5)
arr(7.1, 8.1, 7.95, 8.1, label="512-D")

# ── Stage Classifier (top branch) ───────────────────────────────────────────
rbox(13.0, 9.35, 2.5, 0.7, "Stage Classifier",
     line2="256-D  →  5 classes  (cross-entropy)", fc=C["stage"], fs=8)
rbox(16.9, 9.35, 2.0, 0.6, "Stage\nAlignment Loss",
     fc=C["white"], ec=C["purple"], fs=7.5)

arr(10.25, 8.45, 11.7, 9.1,  color=C["purple"], rad=-0.2, label="256-D")
arr(14.25, 9.35, 15.9, 9.35, color=C["purple"])
ax.text(13.0, 9.95,
        "Same kill-chain stage  →  nearby embedding (semantic alignment)",
        ha="center", fontsize=7, color=C["purple"], style="italic")

# ── GRL + Discriminator (bottom branch) ──────────────────────────────────────
rbox(12.5, 7.0, 2.0, 0.65, "GRL",
     line2="Gradient Reversal Layer", fc=C["grl"], bold=True, fs=8.5)
rbox(15.5, 7.0, 2.2, 0.7, "Domain\nDiscriminator",
     line2="CyberWheel vs CBS", fc=C["disc"], fs=8)
rbox(18.5, 7.0, 1.8, 0.6, "Domain\nAdv. Loss",
     fc=C["white"], ec=C["red"], fs=7.5)

arr(10.25, 7.75, 11.5, 7.3,  color=C["red"], rad=0.2, label="256-D")
arr(13.5,  7.0,  14.4, 7.0,  color=C["red"])
arr(16.6,  7.0,  17.6, 7.0,  color=C["red"])
ax.text(15.3, 6.35,
        "Encoder cannot distinguish source from target  (domain confusion)",
        ha="center", fontsize=7, color=C["red"], style="italic")

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — RL Policy Training + Zero-shot Transfer   y: 0.3 → 5.1
# ══════════════════════════════════════════════════════════════════════════════
phase_box(0.3, 0.3, 19.4, 4.9,
          "Phase 2 — RL Policy Training (source) & Zero-shot Transfer (target)",
          C["green"])

# ── Training row: CW  (y ≈ 3.8) ──────────────────────────────────────────────
ax.text(1.15, 3.8, "Train\n(CW)", ha="center", fontsize=8,
        color=C["blue"], fontweight="bold")

rbox(2.9,  3.8, 2.0, 0.7, "CyberWheel Env",
     line2="episode reward signal", fc=C["cw"], fs=8)
rbox(5.7,  3.8, 2.0, 0.7, "Unified Obs\nPreprocessor",
     line2="→ 512-D", fc=C["prep"], fs=8)
rbox(8.6,  3.8, 2.2, 0.8, "Frozen Encoder",
     line2="512-D  →  256-D",
     fc=C["frozen"], dashed=True, lw=1.8, bold=True, fs=8.5)
rbox(11.6, 3.8, 2.1, 0.8, "PPO Policy",
     line2="trains on 256-D obs",
     fc=C["policy"], bold=True, lw=2.0, fs=8.5)

ax.text(8.6, 4.42, "[frozen weights]", ha="center",
        fontsize=7, color="#7F8C8D", style="italic")

arr(3.9, 3.8, 4.7, 3.8, label="raw obs")
arr(6.7, 3.8, 7.5, 3.8, label="512-D")
arr(9.7, 3.8, 10.5, 3.8, label="256-D")

# Action path
arr(12.65, 3.8, 13.5, 3.8)
rbox(14.4,  3.8, 1.7, 0.6,  "Unified Action\n(0 – 6)", fc=C["white"], fs=7.5)
arr(15.25, 3.8, 16.0, 3.8)
rbox(17.1,  3.8, 1.9, 0.6, "ActionTranslator\n→ CW action", fc=C["cw"], fs=7.5)

# Reward back-arrow (curved, red)
arr(18.05, 3.5, 12.1, 3.38, label="reward",
    color=C["red"], rad=0.3, lw=1.3)

# ── Zero-shot row: CBS  (y ≈ 1.65) ───────────────────────────────────────────
ax.text(1.15, 1.65, "Zero-shot\nTransfer\n(CBS)", ha="center", fontsize=8,
        color=C["green"], fontweight="bold")

rbox(2.9,  1.65, 2.0, 0.7, "CyberBattleSim Env",
     line2="unseen during training",
     fc=C["cbs"], ec=C["green"], lw=2.0, fs=8)
rbox(5.7,  1.65, 2.0, 0.7, "Unified Obs\nPreprocessor",
     line2="→ 512-D", fc=C["prep"], fs=8)
rbox(8.6,  1.65, 2.2, 0.8, "Frozen Encoder",
     line2="512-D  →  256-D",
     fc=C["frozen"], dashed=True, lw=1.8, bold=True, fs=8.5)
rbox(11.6, 1.65, 2.1, 0.8, "PPO Policy",
     line2="same weights · no fine-tuning",
     fc=C["policy"], bold=True, lw=2.0, ec=C["green"], fs=8.5)

ax.text(8.6,  2.27, "[frozen weights]", ha="center",
        fontsize=7, color="#7F8C8D", style="italic")
ax.text(11.6, 0.9,
        "★  Zero-shot evaluation — no retraining on CBS",
        ha="center", fontsize=8.5, color=C["green"], fontweight="bold")

arr(3.9,  1.65, 4.7,  1.65, label="raw obs", color=C["green"])
arr(6.7,  1.65, 7.5,  1.65, label="512-D",   color=C["green"])
arr(9.7,  1.65, 10.5, 1.65, label="256-D",   color=C["green"])
arr(12.65, 1.65, 13.5, 1.65, color=C["green"])
rbox(14.4,  1.65, 1.7, 0.6, "Unified Action\n(0 – 6)", fc=C["white"], fs=7.5)
arr(15.25, 1.65, 16.0, 1.65, color=C["green"])
rbox(17.1,  1.65, 1.9, 0.6, "ActionTranslator\n→ CBS action", fc=C["cbs"], fs=7.5)

# ── Vertical connector: copy encoder weights Phase1 → Phase2 ─────────────────
arr(8.6, 5.62, 8.6, 4.72,
    label="copy weights", color=C["purple"], lw=2.0, ls="--")

# ── Legend ─────────────────────────────────────────────────────────────────────
handles = [
    mpatches.Patch(fc=C["cw"],     ec=C["edge"], label="CyberWheel (source)"),
    mpatches.Patch(fc=C["cbs"],    ec=C["edge"], label="CyberBattleSim (target)"),
    mpatches.Patch(fc=C["prep"],   ec=C["edge"], label="Obs Preprocessor"),
    mpatches.Patch(fc=C["enc"],    ec=C["edge"], label="Shared Encoder (trainable)"),
    mpatches.Patch(fc=C["frozen"], ec=C["edge"], label="Frozen Encoder"),
    mpatches.Patch(fc=C["grl"],    ec=C["edge"], label="Gradient Reversal Layer"),
    mpatches.Patch(fc=C["stage"],  ec=C["edge"], label="Stage Classifier"),
    mpatches.Patch(fc=C["policy"], ec=C["edge"], label="PPO Policy"),
]
ax.legend(handles=handles, loc="lower center", ncol=8,
          bbox_to_anchor=(0.5, -0.03), fontsize=8,
          frameon=True, edgecolor="#cccccc", fancybox=True)

os.makedirs("figures", exist_ok=True)
plt.tight_layout(rect=[0, 0.04, 1, 0.99])
plt.savefig("figures/architecture.png", dpi=180,
            bbox_inches="tight", facecolor="white")
print("Saved → figures/architecture.png")
