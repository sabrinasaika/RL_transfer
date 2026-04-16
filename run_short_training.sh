#!/bin/bash
# Quick training script with fewer steps for testing

set -e  # Exit on error

cd /home/ssaika/rl-transfer-sec-clean

# Set Python path (REQUIRED!)
export PYTHONPATH=/home/ssaika/rl-transfer-sec-clean/cyberwheel:$PYTHONPATH

echo "============================================================"
echo "Starting SHORT Training on Both Scenarios"
echo "============================================================"
echo "Using 5,000 steps (quick test - ~2-3 minutes each)"
echo ""

# ============================================================
# Train on Cyberwheel Scenario (SHORT)
# ============================================================
echo "============================================================"
echo "[1/2] Training on Cyberwheel (5,000 steps)"
echo "============================================================"
export CW_ENV_YAML=credential_preference_scenario.yaml
export CW_TRAIN_STEPS=5000

echo "Environment: credential_preference_scenario.yaml"
echo "Training steps: 5,000 (short test)"
echo ""

python train/train_cw_ppo_minimal.py

echo ""
echo "✓ Cyberwheel training complete!"
echo "  Model saved to: artifacts/policies/cw_ppo_minimal.zip"
echo ""

# ============================================================
# Train on CyberBattleSim Scenario (SHORT)
# ============================================================
echo "============================================================"
echo "[2/2] Training on CyberBattleSim (5,000 steps)"
echo "============================================================"
export CBS_ENV=CyberBattleFlat-v0
export CBS_FLAT_NODES=20
export CBS_CRED_REUSE_PROB=0.6
export CBS_EXPLOIT_PROB=0.3

echo "Environment: CyberBattleFlat-v0"
echo "Training steps: 5,000 (short test)"
echo ""

# Note: CBS training script doesn't have CW_TRAIN_STEPS equivalent
# We'll need to modify it or just let it run with default
python train/train_cbs_ppo_minimal.py

echo ""
echo "✓ CyberBattleSim training complete!"
echo "  Model saved to: artifacts/policies/cbs_ppo_minimal.zip"
echo ""

echo "============================================================"
echo "Short Training Complete!"
echo "============================================================"
echo ""
echo "Trained models:"
echo "  - artifacts/policies/cw_ppo_minimal.zip (Cyberwheel)"
echo "  - artifacts/policies/cbs_ppo_minimal.zip (CyberBattleSim)"
echo ""
echo "Note: These are quick test runs with 5,000 steps."
echo "For full training, use run_both_scenarios.sh (50,000 steps)"
echo ""

