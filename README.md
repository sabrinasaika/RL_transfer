# RL Transfer Learning: CyberWheel → CyberBattleSim

Transfer a reinforcement learning attacker policy trained in **CyberWheel (CW)**
to **CyberBattleSim (CBS)** using a **Domain-Adaptive Policy Network (DAPN)**
encoder and a shared **kill-chain intent interface**.

---

## Overview

Training directly on CyberBattleSim is slow and reward-sparse.
CyberWheel is faster and richer but uses a completely different observation space.
This project bridges the two using:

1. **Kill-chain intent interface** — both environments share the same abstract
   action space (`Discrete(MAX_SLOTS + 1)`).  The agent picks *which host/node
   to advance* along the kill-chain; the wrapper resolves that into the concrete
   environment action.

2. **Simulated CW phases inside CBS** — CBS nodes lack CW's 7-phase recon
   progression (ping_sweep → port_scan → service_disc → exploit → owned →
   escalated → impacted).  The wrapper simulates those phases per-slot so the
   policy sees the same phase sequence it was trained on.

3. **Phase-aware DAPN encoder** — an adversarial domain-adaptation encoder
   (DANN + stage prediction + pair alignment) maps CBS observations into the
   same 512-D space the CW-trained policy expects.  Phase values (0.0–1.0)
   are injected into `nodes_privilegelevel` before encoding so intermediate
   recon progress is visible to the encoder.

---

## CBS Chain Structure

`CBS_SIZE=12` creates **14 nodes**:

```
node 0 (entry, pre-owned)  →  nodes 1–12 (chain)  →  node 13 (flag)
                                 ↑
                         12 intermediate nodes
                         (alternating Linux/Windows)
```

- **Win condition** (`--win-nodes 8`): own ≥ 8 chain nodes (= 9 total including node 0)
- **MAX_SLOTS = 12**: policy can target nodes 0–11 via slots 0–11; slot 12 = no-op

---

## Prerequisites

- Python 3.10
- Poetry ≥ 1.5 (for CyberWheel)
- Graphviz

---

## Setup

```bash
cd ~/Downloads/rl-transfer-sec-clean

# Create and activate virtual environment
python3.10 -m venv .venv
source .venv/bin/activate

# Install CyberBattleSim
pip install -e CyberBattleSim
pip install gymnasium==0.29.1 stable-baselines3==2.3.2 numpy==1.26.4
pip install tqdm pydantic jsonpickle python-dotenv networkx pyyaml
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu

# Install CyberWheel
cd cyberwheel && poetry install && cd ..
```

Set PYTHONPATH for all commands below:

```bash
export PYTHONPATH=$PWD:$PWD/cyberwheel:$PWD/CyberBattleSim
```

---

## Training Pipeline

### Step 1 — Train CW kill-chain policy

Trains a PPO policy on CyberWheel using the kill-chain intent interface
(12 slots, 200 000 timesteps).

```bash
python3.10 train_kc_policy.py \
    --no-encoder \
    --timesteps 200000 \
    --out artifacts/policies/cw_kc_raw_policy.zip
```

Best checkpoint saved to:
`artifacts/policies/best_kc_raw_12slot/best_kc_raw/best_model.zip`

### Step 2 — Train phase-aware DAPN encoder

Collects paired CW/CBS observations (with simulated phases injected) and
trains the DAPN encoder using adversarial domain adaptation.

```bash
python3.10 train_dapn_encoder.py \
    --num-samples 2000 \
    --epochs 50 \
    --save-encoder artifacts/transfer_models/dapn_encoder_phase_aware.pt
```

### Step 3 — Evaluate transfer (4 conditions)

```bash
python3.10 eval_kc_transfer.py \
    --raw-policy artifacts/policies/best_kc_raw_12slot/best_kc_raw/best_model.zip \
    --encoder    artifacts/transfer_models/dapn_encoder_phase_aware.pt \
    --cbs-size   12 \
    --win-nodes  8 \
    --episodes   20 \
    --max-steps  200 \
    --out        results/kc_eval_4cond_20ep.json
```

Runs in approximately **15 minutes**.

### Step 4 (optional) — Fine-tune on CBS

Few-shot fine-tuning of the CW policy on CBS via DAPN translation.

```bash
python3.10 finetune_kc_cbs.py \
    --raw-policy artifacts/policies/best_kc_raw_12slot/best_kc_raw/best_model.zip \
    --encoder    artifacts/transfer_models/dapn_encoder_phase_aware.pt \
    --timesteps  10000
```

---

## Evaluation Results

Four conditions evaluated on CBS chain-12, win = 8 nodes owned, 20 episodes:

| Condition | Win Rate | Nodes Owned | Steps to 1st | Mean Return |
|-----------|----------|-------------|--------------|-------------|
| True Random | 0% | 2.60 | 21.9 | 336 |
| KC-Random | 100% | 8.00 | 13.0 | 6,022 |
| No DAPN | 0% | 0.00 | — | 19 |
| **DAPN** | **100%** | **8.00** | **13.1** | **6,024** |

**Conditions explained:**

| Condition | Slot selection | Action translation |
|-----------|---------------|-------------------|
| True Random | — | None — raw CBS action mask (genuine lower bound) |
| KC-Random | Uniform random | Kill-chain translation (`_advance_cbs`) |
| No DAPN | CW-trained policy | None — wrong obs space, always fails |
| DAPN | CW-trained policy | Phase-aware DAPN encoder |

**Key findings:**
- **True Random → 0%**: sampling randomly from raw CBS actions (80B+ combos) is
  essentially impossible — most actions are invalid or traps
- **No DAPN → 0%**: the CW policy cannot operate on raw CBS observations
- **KC-Random = DAPN (100%)** with 500 steps: the kill-chain translation layer
  alone is sufficient; the learned policy's advantage appears under tight step budgets
- **Under tight budget (60 steps)**: DAPN owns 3.50 nodes vs KC-Random 3.30,
  showing the policy does add value when efficiency matters

---

## Step-by-Step Policy Trace

Watch the policy operate episode-by-episode:

```bash
# DAPN policy trace (default)
python3.10 trace_policy.py --episodes 3 --max-steps 150 --win-nodes 8

# True random baseline
python3.10 trace_policy.py --random --episodes 3 --max-steps 150 --win-nodes 8

# Raw CW policy (no DAPN — should fail)
python3.10 trace_policy.py --no-dapn --episodes 3 --max-steps 150 --win-nodes 8
```

Output columns:

| Column | Meaning |
|--------|---------|
| Step | Step number in the episode |
| Slot | Host slot the policy chose (0–11) |
| Phase | Simulated CW kill-chain phase for that slot |
| CBS Action | Actual action sent to CBS (`local_vuln`, `connect`, `remote_vuln`) |
| Owned | Total chain nodes owned so far |
| Reward | Reward earned this step |

---

## Kill-Chain Phase Mapping

Each slot tracks its own simulated CW phase:

| Phase | Name | CBS Action |
|-------|------|-----------|
| 0 | ping_sweep | `local_vulnerability(prev_owned, vuln_idx=0)` |
| 1 | port_scan | `local_vulnerability(prev_owned, vuln_idx=1)` |
| 2 | service_disc | `local_vulnerability(prev_owned, vuln_idx=2)` |
| 3 | exploit | `connect(src → target, port, cred)` if credential found |
| 4+ | owned/pivot | `local_vulnerability(node_id)` or `connect` to next node |

Phases 0–2 cycle through `vuln_idx` to find the credential-stealing vulnerability.
Phase 3 uses the credential to own the target node via `connect`.

---

## Environment Variables

| Variable | Default | Meaning |
|----------|---------|---------|
| `CBS_SIZE` | 6 | Chain length (use 12 for 14-node chain) |
| `CBS_WIN_NODES` | — | Absolute node count to win (e.g. 8) |
| `CBS_GOAL_OWN_PCT` | 0.5 | Percentage threshold (used when CBS_WIN_NODES not set) |

---

## Key Files

```
adapters/
  unified_env.py              # Kill-chain intent wrapper (CW + CBS)
  kc_dapn_translate_wrapper.py # DAPN obs-translation for CBS eval
  kc_dapn_wrapper.py          # DAPN obs-compression for CW training
  kill_chain.py               # Stage extraction helpers

config/
  env_builders.py             # make_cw_env() and make_cbs_env()

train_kc_policy.py            # Step 1: train CW kill-chain policy
train_dapn_encoder.py         # Step 2: train DAPN encoder
eval_kc_transfer.py           # Step 3: 4-condition transfer evaluation
finetune_kc_cbs.py            # Step 4: few-shot CBS fine-tuning
trace_policy.py               # Step-by-step policy visualisation
test_action_translation.py    # Verify kill-chain action translation (259 tests)

artifacts/
  policies/best_kc_raw_12slot/best_kc_raw/best_model.zip  # CW policy
  transfer_models/dapn_encoder_phase_aware.pt              # DAPN encoder

results/
  kc_eval.json                # 4-condition eval results
  trajectory_plot.png         # Phase heatmap + ownership timeline
```
