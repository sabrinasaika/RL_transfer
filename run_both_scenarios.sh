#!/bin/bash
# Complete script to run full training on both scenarios

set -e  # Exit on error

cd /home/ssaika/rl-transfer-sec-clean

# Set Python path (REQUIRED!)
export PYTHONPATH=/home/ssaika/rl-transfer-sec-clean/cyberwheel:$PYTHONPATH

echo "============================================================"
echo "Starting Full Training on Both Scenarios"
echo "============================================================"
echo ""

# ============================================================
# Train on Cyberwheel Scenario
# ============================================================
echo "============================================================"
echo "[1/2] Training on Cyberwheel Credential Preference Scenario"
echo "============================================================"
export CW_ENV_YAML=credential_preference_scenario.yaml

echo "Environment: credential_preference_scenario.yaml"
echo "Training steps: ${CW_TRAIN_STEPS:-50000} (default)"
echo "This will take approximately 10-30 minutes..."
echo ""

python train/train_cw_ppo_minimal.py

echo ""
echo "✓ Cyberwheel training complete!"
echo "  Model saved to: artifacts/policies/cw_ppo_minimal.zip"
echo ""

# ============================================================
# Train on CyberBattleSim Scenario  
# ============================================================
echo "============================================================"
echo "[2/2] Training on CyberBattleSim Flat Network Scenario"
echo "============================================================"
export CBS_ENV=CyberBattleFlat-v0
export CBS_FLAT_NODES=20
export CBS_CRED_REUSE_PROB=0.6
export CBS_EXPLOIT_PROB=0.3

echo "Environment: CyberBattleFlat-v0"
echo "Nodes: 20"
echo "Credential reuse probability: 0.6"
echo "Exploit probability: 0.3"
echo "Training steps: 50000 (default)"
echo "This will take approximately 10-30 minutes..."
echo ""

python train/train_cbs_ppo_minimal.py

echo ""
echo "✓ CyberBattleSim training complete!"
echo "  Model saved to: artifacts/policies/cbs_ppo_minimal.zip"
echo ""

echo "============================================================"
echo "All Training Complete!"
echo "============================================================"
echo ""
echo "Trained models:"
echo "  - artifacts/policies/cw_ppo_minimal.zip (Cyberwheel)"
echo "  - artifacts/policies/cbs_ppo_minimal.zip (CyberBattleSim)"
echo ""

