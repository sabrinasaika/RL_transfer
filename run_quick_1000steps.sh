#!/bin/bash
# Quick training: 1000 steps, 5 episodes

set -e

cd /home/ssaika/rl-transfer-sec-clean

# Set Python path
export PYTHONPATH=/home/ssaika/rl-transfer-sec-clean/cyberwheel:$PYTHONPATH

echo "============================================================"
echo "QUICK TRAINING: 1000 steps, ~5 episodes"
echo "============================================================"
echo ""

# Cyberwheel - 1000 steps
echo "[1/2] Training Cyberwheel (1000 steps)..."
export CW_ENV_YAML=credential_preference_scenario.yaml
export CW_TRAIN_STEPS=1000
export CW_NUM_EPISODES=5
python train/train_cw_ppo_quick.py

echo ""
echo "✓ Cyberwheel done!"
echo ""

# CyberBattleSim - 1000 steps
echo "[2/2] Training CyberBattleSim (1000 steps)..."
export CBS_ENV=CyberBattleFlat-v0
export CBS_FLAT_NODES=20
export CBS_CRED_REUSE_PROB=0.6
export CBS_EXPLOIT_PROB=0.3
export CBS_TRAIN_STEPS=1000
export CBS_NUM_EPISODES=5
python train/train_cbs_ppo_quick.py

echo ""
echo "============================================================"
echo "✓ Quick training complete!"
echo "============================================================"
echo ""
echo "Models saved:"
echo "  - artifacts/policies/cw_ppo_quick.zip"
echo "  - artifacts/policies/cbs_ppo_quick.zip"
echo ""

