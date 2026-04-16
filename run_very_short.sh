#!/bin/bash
# VERY SHORT training - 500 steps each (~30 seconds)

cd /home/ssaika/rl-transfer-sec-clean
export PYTHONPATH=/home/ssaika/rl-transfer-sec-clean/cyberwheel:$PYTHONPATH

echo "============================================================"
echo "VERY SHORT TRAINING: 500 steps each (~30 seconds)"
echo "============================================================"
echo ""

# Cyberwheel
echo "[1/2] Cyberwheel (500 steps)..."
export CW_ENV_YAML=credential_preference_scenario.yaml
python train/train_cw_ppo_very_short.py

echo ""
echo "✓ Cyberwheel done!"
echo ""

# CyberBattleSim
echo "[2/2] CyberBattleSim (500 steps)..."
export CBS_ENV=CyberBattleFlat-v0
export CBS_FLAT_NODES=20
export CBS_CRED_REUSE_PROB=0.6
export CBS_EXPLOIT_PROB=0.3
python train/train_cbs_ppo_very_short.py

echo ""
echo "============================================================"
echo "✓ Very short training complete!"
echo "============================================================"

