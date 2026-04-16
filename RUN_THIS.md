# How to Run the Code - Simple Guide

## Step 1: Open Terminal and Navigate to Project

```bash
cd /home/ssaika/rl-transfer-sec-clean
```

## Step 2: Test That Everything Works

```bash
python test_scenarios.py
```

**Expected output:** Both scenarios should show "✓ PASSED"

---

## Step 3: Choose What to Run

### Option A: Train on Cyberwheel Scenario

```bash
# Set the environment
export CW_ENV_YAML=credential_preference_scenario.yaml

# Run training
python train/train_cw_ppo_minimal.py
```

**This will:**
- Train a PPO agent on the credential preference scenario
- Save the model to `artifacts/policies/cw_ppo_minimal.zip`
- Take several minutes (50,000 training steps)

---

### Option B: Train on CyberBattleSim Scenario

```bash
# Set the environment variables
export CBS_ENV=CyberBattleFlat-v0
export CBS_FLAT_NODES=20
export CBS_CRED_REUSE_PROB=0.6
export CBS_EXPLOIT_PROB=0.3

# Run training
python train/train_cbs_ppo_minimal.py
```

**This will:**
- Train a PPO agent on the flat network scenario
- Save the model to `artifacts/policies/cbs_ppo_minimal.zip`
- Take several minutes (50,000 training steps)

---

### Option C: Train DAPN Encoder (for Transfer Learning)

```bash
python train_dapn_encoder.py --num-samples 1000 --epochs 50
```

**This will:**
- Collect observations from both domains
- Train the DAPN encoder
- Save to `artifacts/transfer_models/dapn_encoder.pt`
- Take several minutes

---

## Complete Example: Run Everything

Copy and paste this entire block:

```bash
# Navigate to project
cd /home/ssaika/rl-transfer-sec-clean

# Test first
python test_scenarios.py

# Train on Cyberwheel
export CW_ENV_YAML=credential_preference_scenario.yaml
python train/train_cw_ppo_minimal.py

# Train on CyberBattleSim (in a new terminal or after first completes)
export CBS_ENV=CyberBattleFlat-v0
export CBS_FLAT_NODES=20
export CBS_CRED_REUSE_PROB=0.6
export CBS_EXPLOIT_PROB=0.3
python train/train_cbs_ppo_minimal.py
```

---

## If You Get Errors

### Error: "No module named 'cyberwheel'"

Run this first:
```bash
export PYTHONPATH=/home/ssaika/rl-transfer-sec-clean/cyberwheel:$PYTHONPATH
```

### Error: "KeyError" or "FileNotFoundError"

Run the diagnostic:
```bash
python diagnose_cyberwheel.py
```

### Still Having Issues?

Run the fix script:
```bash
./fix_and_test.sh
```

---

## Quick Reference Card

```bash
# TEST
python test_scenarios.py

# CYBERWHEEL
export CW_ENV_YAML=credential_preference_scenario.yaml
python train/train_cw_ppo_minimal.py

# CYBERBATTLESIM
export CBS_ENV=CyberBattleFlat-v0 && python train/train_cbs_ppo_minimal.py

# DAPN ENCODER
python train_dapn_encoder.py --num-samples 1000 --epochs 50
```

---

## What Each Command Does

| Command | What It Does | Output |
|---------|-------------|--------|
| `python test_scenarios.py` | Tests both scenarios work | Shows PASSED/FAILED |
| `python train/train_cw_ppo_minimal.py` | Trains agent on Cyberwheel | Saves model to `artifacts/policies/` |
| `python train/train_cbs_ppo_minimal.py` | Trains agent on CyberBattleSim | Saves model to `artifacts/policies/` |
| `python train_dapn_encoder.py ...` | Trains DAPN encoder | Saves to `artifacts/transfer_models/` |

---

## Need Help?

1. **Check if files exist:**
   ```bash
   ls cyberwheel/cyberwheel/data/configs/environment/credential_preference_scenario.yaml
   ```

2. **Run diagnostic:**
   ```bash
   python diagnose_cyberwheel.py
   ```

3. **Check Python path:**
   ```bash
   python -c "import sys; print('\n'.join(sys.path))"
   ```

