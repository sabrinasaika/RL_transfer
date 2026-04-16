# Disk Space Guide

## Current Status

**Disk Usage**: 91% used, **328GB free** (out of 3.6TB total)

This should be **enough** for collecting the full dataset (50 episodes each).

---

## Where Space Is Being Used

### Current Disk Usage Breakdown

| Location | Size | Can Delete? | Notes |
|----------|------|-------------|-------|
| `.venv` | 7.1GB | ❌ **NO** | Python virtual environment (required) |
| `artifacts/` | 473MB | ⚠️ **Selective** | See breakdown below |
| `cyberwheel/` | 128MB | ⚠️ **Selective** | Source code + some data |
| `CyberBattleSim/` | 4.4MB | ❌ **NO** | Source code (required) |
| `DAPN-master/` | 3.9MB | ❌ **NO** | Source code (required) |

### Artifacts Breakdown

| Directory | Size | Can Delete? |
|-----------|------|-------------|
| `artifacts/wandb/` | 245MB | ✅ **YES** (old training logs) |
| `artifacts/transfer_models/` | 215MB | ⚠️ **Selective** (keep latest) |
| `artifacts/training_data/` | 5.6MB | ✅ **YES** (old/partial files) |
| `artifacts/policies/` | 5.7MB | ❌ **NO** (needed for collection) |
| `artifacts/plots/` | 2.3MB | ✅ **YES** (old plots) |
| `artifacts/logs/` | 364KB | ✅ **YES** (old logs) |

---

## How Much Space Do You Need?

### For Full Collection (50 episodes each):

**Estimated file size**: ~10-20GB (with compression)
- Raw CBS observations are large (dictionaries with arrays)
- Compressed format helps but still substantial
- **You have 328GB free** - **plenty of space!**

### For Smaller Collection (30 episodes each):

**Estimated file size**: ~5-10GB
- Still enough for training
- More manageable

---

## How to Free Up Space

### Option 1: Clean Up Old Training Data (Safe)

```bash
# Remove old/partial collection files
rm -f artifacts/training_data/episodic_obs*.npz
rm -f artifacts/training_data/*_old.npz
rm -f artifacts/training_data/*_partial.npz

# Check what you're deleting first
ls -lh artifacts/training_data/
```

### Option 2: Clean Up Old WandB Logs (Safe)

```bash
# WandB logs can be large and are usually not needed after training
# Keep only recent runs if needed
du -sh artifacts/wandb/*
ls -lht artifacts/wandb/ | head -10  # See newest files

# Delete old WandB runs (be careful - check first!)
# rm -rf artifacts/wandb/run-YYYYMMDD_*  # Delete specific old runs
```

### Option 3: Clean Up Old Transfer Models (Selective)

```bash
# Keep only the latest/best models
ls -lht artifacts/transfer_models/*.pt

# Delete old iterations (keep final models)
rm -f artifacts/transfer_models/*_iter_*.pt  # Old checkpoints
# Keep: dapn_encoder_episodic.pt, dapn_encoder.pt (final models)
```

### Option 4: Clean Up Old Logs and Plots (Safe)

```bash
# Old logs and plots
rm -f artifacts/*.log
rm -f artifacts/plots/*.png
rm -f artifacts/logs/*
```

### Option 5: Clean Up Cyberwheel Artifacts (If Not Needed)

```bash
# Check what's in cyberwheel
du -sh cyberwheel/* | sort -h

# Old training logs, checkpoints, etc.
# Be careful - only delete if you're sure you don't need them
```

---

## Quick Cleanup Script

Here's a safe cleanup script:

```bash
#!/bin/bash
# Safe cleanup - removes old/unnecessary files

echo "Cleaning up old files..."

# Remove old/partial training data
echo "Removing old training data files..."
rm -f artifacts/training_data/episodic_obs*.npz
rm -f artifacts/training_data/*_old.npz
rm -f artifacts/training_data/*_partial.npz

# Remove old model checkpoints (keep final models)
echo "Removing old model checkpoints..."
find artifacts/transfer_models -name "*_iter_*.pt" -type f -delete

# Remove old logs
echo "Removing old logs..."
rm -f artifacts/*.log
rm -f artifacts/logs/*.log

# Show space freed
echo ""
echo "✅ Cleanup complete!"
df -h . | tail -1
```

---

## What NOT to Delete

❌ **DO NOT DELETE:**
- `.venv/` - Python virtual environment (required)
- `artifacts/policies/*.zip` - Trained policies (needed for collection)
- `artifacts/transfer_models/dapn_encoder*.pt` - Latest encoder models
- `CyberBattleSim/` - Source code
- `DAPN-master/` - Source code
- `cyberwheel/cyberwheel/` - Source code

---

## Space Requirements Summary

### Current Situation:
- **Free space**: 328GB
- **Needed for full collection**: ~10-20GB
- **Status**: ✅ **You have enough space!**

### If You Need More Space:
1. Delete old training data files (safest)
2. Delete old WandB logs (if not needed)
3. Delete old model checkpoints (keep final models)
4. Delete old logs and plots

---

## Recommended Action

**You have 328GB free - this is enough for the full collection!**

Just run:
```bash
python collect_episodic_data_with_policies.py \
    --cw-policy artifacts/policies/cw_ppo_dapn.zip \
    --cbs-policy artifacts/policies/cbs_ppo_final.zip \
    --cw-episodes 50 \
    --cbs-episodes 50 \
    --max-steps 200 \
    --output artifacts/training_data/episodic_obs_policy.npz
```

If you want to free up more space first (optional):
```bash
# Quick cleanup
rm -f artifacts/training_data/episodic_obs*.npz
rm -f artifacts/transfer_models/*_iter_*.pt
rm -f artifacts/*.log
```
