#!/bin/bash
# VERY QUICK training - just 2,000 steps each (~1-2 minutes)

set -e

cd /home/ssaika/rl-transfer-sec-clean

# Set Python path
export PYTHONPATH=/home/ssaika/rl-transfer-sec-clean/cyberwheel:$PYTHONPATH

echo "============================================================"
echo "QUICK TEST - 2,000 steps each (~1-2 minutes)"
echo "============================================================"
echo ""

# Cyberwheel - Quick
echo "[1/2] Cyberwheel (2,000 steps)..."
export CW_ENV_YAML=credential_preference_scenario.yaml
export CW_TRAIN_STEPS=2000
python train/train_cw_ppo_minimal.py

echo ""
echo "✓ Cyberwheel done!"
echo ""

# CyberBattleSim - Quick
echo "[2/2] CyberBattleSim (2,000 steps)..."
export CBS_ENV=CyberBattleFlat-v0
export CBS_FLAT_NODES=20
export CBS_CRED_REUSE_PROB=0.6
export CBS_EXPLOIT_PROB=0.3
export CBS_TRAIN_STEPS=2000
python train/train_cbs_ppo_short.py

echo ""
echo "============================================================"
echo "✓ Quick test complete!"
echo "============================================================"

