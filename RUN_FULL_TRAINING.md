# How to Run Full Training on Both Scenarios

## Quick Start - Run Everything

### Option 1: Run Both Scenarios (One After Another)

```bash
cd /home/ssaika/rl-transfer-sec-clean

# Set Python path (IMPORTANT!)
export PYTHONPATH=/home/ssaika/rl-transfer-sec-clean/cyberwheel:$PYTHONPATH

# Train on Cyberwheel scenario
export CW_ENV_YAML=credential_preference_scenario.yaml
python train/train_cw_ppo_minimal.py

# Train on CyberBattleSim scenario (after first completes)
export CBS_ENV=CyberBattleFlat-v0
export CBS_FLAT_NODES=20
export CBS_CRED_REUSE_PROB=0.6
export CBS_EXPLOIT_PROB=0.3
python train/train_cbs_ppo_minimal.py
```

### Option 2: Run in Background (Parallel)

```bash
cd /home/ssaika/rl-transfer-sec-clean
export PYTHONPATH=/home/ssaika/rl-transfer-sec-clean/cyberwheel:$PYTHONPATH

# Start Cyberwheel training in background
export CW_ENV_YAML=credential_preference_scenario.yaml
nohup python train/train_cw_ppo_minimal.py > cw_training.log 2>&1 &

# Start CyberBattleSim training in background
export CBS_ENV=CyberBattleFlat-v0
export CBS_FLAT_NODES=20
export CBS_CRED_REUSE_PROB=0.6
export CBS_EXPLOIT_PROB=0.3
nohup python train/train_cbs_ppo_minimal.py > cbs_training.log 2>&1 &

# Check progress
tail -f cw_training.log
tail -f cbs_training.log
```

---

## Detailed Instructions

### Step 1: Train on Cyberwheel Scenario

**What it does:**
- Trains a PPO agent on the credential preference scenario
- 100 hosts network (70 user, 20 app servers, 10 infrastructure)
- Rewards: +2 for authenticated access, -1 for failed exploit
- Default: 50,000 training steps

**Command:**
```bash
cd /home/ssaika/rl-transfer-sec-clean
export PYTHONPATH=/home/ssaika/rl-transfer-sec-clean/cyberwheel:$PYTHONPATH
export CW_ENV_YAML=credential_preference_scenario.yaml
python train/train_cw_ppo_minimal.py
```

**Custom training steps:**
```bash
export CW_TRAIN_STEPS=100000  # Train for 100k steps instead of 50k
python train/train_cw_ppo_minimal.py
```

**Output:**
- Model saved to: `artifacts/policies/cw_ppo_minimal.zip`
- Training progress shown in terminal

---

### Step 2: Train on CyberBattleSim Scenario

**What it does:**
- Trains a PPO agent on the flat network scenario
- Tests if agent prioritizes credential actions before exploit
- Credential reuse success: 60%, Exploit success: 30%
- Default: 50,000 training steps

**Command:**
```bash
cd /home/ssaika/rl-transfer-sec-clean
export CBS_ENV=CyberBattleFlat-v0
export CBS_FLAT_NODES=20
export CBS_CRED_REUSE_PROB=0.6
export CBS_EXPLOIT_PROB=0.3
python train/train_cbs_ppo_minimal.py
```

**Custom parameters:**
```bash
export CBS_FLAT_NODES=30              # More nodes
export CBS_CRED_REUSE_PROB=0.7        # Higher credential success
export CBS_EXPLOIT_PROB=0.2           # Lower exploit success
python train/train_cbs_ppo_minimal.py
```

**Output:**
- Model saved to: `artifacts/policies/cbs_ppo_minimal.zip`
- Training progress shown in terminal

---

## Complete Example Session

Copy and paste this entire block:

```bash
# Navigate to project
cd /home/ssaika/rl-transfer-sec-clean

# Set Python path (CRITICAL!)
export PYTHONPATH=/home/ssaika/rl-transfer-sec-clean/cyberwheel:$PYTHONPATH

# ============================================================
# Train on Cyberwheel Scenario
# ============================================================
echo "Starting Cyberwheel training..."
export CW_ENV_YAML=credential_preference_scenario.yaml
python train/train_cw_ppo_minimal.py

# ============================================================
# Train on CyberBattleSim Scenario
# ============================================================
echo "Starting CyberBattleSim training..."
export CBS_ENV=CyberBattleFlat-v0
export CBS_FLAT_NODES=20
export CBS_CRED_REUSE_PROB=0.6
export CBS_EXPLOIT_PROB=0.3
python train/train_cbs_ppo_minimal.py

echo "All training complete!"
```

---

## What to Expect

### During Training:
- Progress bars showing training steps
- Episode rewards and statistics
- Model checkpoints saved periodically

### After Training:
- **Cyberwheel model:** `artifacts/policies/cw_ppo_minimal.zip`
- **CyberBattleSim model:** `artifacts/policies/cbs_ppo_minimal.zip`

### Training Time:
- Each scenario: ~10-30 minutes (depending on hardware)
- 50,000 steps is the default
- You can reduce steps for faster testing: `export CW_TRAIN_STEPS=5000`

---

## Quick Test (Shorter Training)

If you want to test quickly with fewer steps:

```bash
cd /home/ssaika/rl-transfer-sec-clean
export PYTHONPATH=/home/ssaika/rl-transfer-sec-clean/cyberwheel:$PYTHONPATH

# Quick Cyberwheel test (5k steps, ~2-3 minutes)
export CW_ENV_YAML=credential_preference_scenario.yaml
export CW_TRAIN_STEPS=5000
python train/train_cw_ppo_minimal.py

# Quick CyberBattleSim test (5k steps, ~2-3 minutes)
export CBS_ENV=CyberBattleFlat-v0
python train/train_cbs_ppo_minimal.py
```

---

## Check Training Results

After training completes, check the saved models:

```bash
ls -lh artifacts/policies/
```

You should see:
- `cw_ppo_minimal.zip` (Cyberwheel model)
- `cbs_ppo_minimal.zip` (CyberBattleSim model)

---

## Next Steps After Training

1. **Evaluate the trained models**
2. **Compare agent behavior** (credential preference vs exploit preference)
3. **Use DAPN for transfer learning** between scenarios
4. **Analyze reward signals and learning curves`

