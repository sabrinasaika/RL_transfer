# Data Collection Summary: Deterministic Lockstep

## What This Is

**Script:** `collect_deterministic_lockstep.py`

**Purpose:** Collect **paired** (source, target) data for transfer learning: same trajectory in both domains so you can train an encoder to map source (Cyberwheel) observations to target (CyberBattleSim) observations. No encoder and no policy during collection—only raw observations and a fixed action rule.

**Main idea:** Both environments are stepped in **lockstep** with the **same action** each step and the **same seed** per episode, so (CW state, CBS state, action) triples are aligned and reproducible.

---

## How It Works

### 1. Setup

- **Source domain:** Cyberwheel (CW) — `UnifiedSecEnv("cw", ...)`  
- **Target domain:** CyberBattleSim (CBS) — `UnifiedSecEnv("cbs", ...)`  
- **CBS env:** Uses `CyberBattleCW10-v0` (10-host topology aligned with CW).  
- **Determinism:** `DETERMINISTIC_BACKEND_ACTION=1` so when a unified action is translated to CW/CBS native actions, the first valid option is chosen (no random tie-breaking).

### 2. Per-Episode Loop

- **Seed:** Episode `i` uses `seed + i` (e.g. seed=42 → episodes 42, 43, 44, …).  
- **Reset:** `cw_env.reset(seed=ep_seed)` and `cbs_env.reset(seed=ep_seed)` so both start from a deterministic initial state.

### 3. Per-Step Loop (Lockstep)

Each step:

1. **Read raw observations** (before applying the action):
   - `raw_cw` = `cw_env._last_raw_obs` (e.g. 701-dim array or empty array).
   - `raw_cbs` = `cbs_env._last_raw_obs` (dict: `discovered_node_count`, `credential_cache_length`, `discovered_nodes_properties`, `action_mask`, etc.).

2. **Choose one action (deterministic, no policy):**  
   `action = step_index % 7` → round-robin over unified actions 0–6:  
   `noop`, `ping_sweep`, `port_scan`, `discovery`, `lateral_move`, `privilege_escalation`, `impact`.

3. **Store one paired row:**
   - `source_obs.append(raw_cw)`
   - `target_obs.append(raw_cbs)`
   - `source_actions.append(action)` and `target_actions.append(action)` (same action for both).
   - Backend actions: `source_backend_actions`, `target_backend_actions` (what each env actually executed).

4. **Step both envs with the same action:**  
   `cw_env.step(action)`, `cbs_env.step(action)`.

5. **Stop** when either env is done or `max_steps_per_episode` is reached; then start the next episode until `num_samples` steps are collected.

### 4. Post-Processing

- **Validation split:** First `val_fraction` (default 20%) of **target** samples are put in `val_obs`; the rest stay in `target_obs` (and corresponding `source_obs`, actions, and backend actions are split the same way).
- **Shuffle:** Train (source, target, actions, backend) are shuffled with the same index so pairs stay aligned.
- **Save:** All lists/arrays are written to a single `.npz` with `allow_pickle=True`.

---

## What Gets Saved (.npz)

| Key | Type | Description |
|-----|------|-------------|
| `source_obs` | list of arrays | Raw CW observation per step (e.g. shape `(701,)` or empty). |
| `target_obs` | list of dicts | Raw CBS observation per step (dict with `discovered_node_count`, `credential_cache_length`, `discovered_nodes_properties`, `action_mask`, etc.). |
| `val_obs` | list of dicts | Validation subset of `target_obs` (same structure). |
| `source_actions` | int array | Unified action index 0–6 per step (CW side). |
| `target_actions` | int array | Same as `source_actions` (lockstep). |
| `source_backend_actions` | list | CW’s actual executed action (e.g. dict) per step. |
| `target_backend_actions` | list | CBS’s actual executed action (e.g. `{"connect": (src, dst, port, cred)}`) per step. |

After the script runs you see:

- `Collected N paired samples.`
- `Validation: K; train pairs: M` (if `val_fraction` > 0).
- `Saved to <path>` (if `--save` was given).

---

## Verification (Determinism)

**Command:**

```bash
python collect_deterministic_lockstep.py --verify
```

**What it does:** Runs the same lockstep collection **twice** with the same seed (and same max steps). For each step it compares:

- Unified action (must match).
- Checksum of CW observation (must match).
- Checksum of CBS observation (must match).

**Output:**

- A table: step index, action R1/R2, CW checksum R1/R2, CBS checksum R1/R2, and `OK` or `MISMATCH`.
- Final line: `Same state & action across runs: True` or `False`.

**Interpretation:**

- **All OK + True:** Collection is deterministic; re-running with the same seed reproduces the same (state, action) sequence.
- **Any MISMATCH or False:** Something non-deterministic is present (env, backend action, or RNG).

Optional: `--verify --print` prints a short state/action summary per step (action name, CW shape/min/max, CBS scalars) for inspection.

---

## Result Analysis

### 1. First-Observation Check

On the first step of the first episode the script prints:

```text
[CBS first obs] discovered_node_count=X credential_cache_length=Y
```

- **Expected (CyberBattleCW10-v0):** `X ≥ 1`, `Y ≥ 1` (seeded reset gives at least one discovered node and one credential). If both are 0, a warning suggests checking `adapters/cbs_topologies.py` seeded reset.

### 2. CBS Observation Structure

- **Scalars:** `discovered_node_count`, `credential_cache_length`, `newly_discovered_nodes_count`, `probe_result`, `escalation`, etc. These are the main “counts” and outcomes; watch these for non-zero, sensible values.
- **Arrays:** Fixed-size, zero-padded (e.g. `discovered_nodes_properties` shape `(max_nodes, 3)`). Only the first `discovered_node_count` rows are meaningful; the rest are padding. `action_mask` is mostly 0 (invalid); 1 = valid. So “lots of zeros” in arrays is normal.

### 3. Loaded Data Sanity Checks

After loading the `.npz`:

```python
import numpy as np
d = np.load("artifacts/training_data/lockstep.npz", allow_pickle=True)
# Check counts
assert len(d["source_obs"]) == len(d["target_obs"]) == len(d["source_actions"])
# Check alignment: same action on both sides
assert np.array_equal(d["source_actions"], d["target_actions"])
# Check first CBS obs has non-zero scalars (if using CyberBattleCW10-v0)
o0 = d["target_obs"][0]
print(o0["discovered_node_count"], o0["credential_cache_length"])
```

### 4. Downstream Use

- **Encoder training:** Use `source_obs` and `target_obs` (and optionally `val_obs`) to train a mapping (e.g. DAPN-style encoder) from CW observations to CBS-comparable representation.
- **Backend actions:** `source_backend_actions` and `target_backend_actions` record exactly what each env executed, useful for debugging or action-conditioned models.

---

## Quick Reference: Commands

| Goal | Command |
|------|---------|
| Verify determinism | `python collect_deterministic_lockstep.py --verify` |
| Verify + print state/action | `python collect_deterministic_lockstep.py --verify --print` |
| Collect and save | `python collect_deterministic_lockstep.py --num-samples 500 --save artifacts/training_data/lockstep.npz` |
| Custom seed / steps | `python collect_deterministic_lockstep.py --num-samples 1000 --save out.npz --seed 42 --max-steps 200 --val-fraction 0.2` |

---

## Summary Table

| Aspect | Detail |
|--------|--------|
| **Mode** | Deterministic lockstep (same seed per episode, same action per step for both envs). |
| **Action** | Round-robin `step % 7` over unified actions 0–6; no policy. |
| **State** | Full raw obs: CW = array, CBS = dict (all keys preserved). |
| **Action saved** | Unified index (0–6) + full backend action per env. |
| **Reproducibility** | Check with `--verify`; same seed ⇒ same (state, action) sequence. |
| **Output** | `.npz`: `source_obs`, `target_obs`, `val_obs`, `source_actions`, `target_actions`, `source_backend_actions`, `target_backend_actions`. |
