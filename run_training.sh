#!/bin/bash
# Script to run full training on both scenarios

set -e  # Exit on error

cd /home/ssaika/rl-transfer-sec-clean

# Set Python paths
export PYTHONPATH=/home/ssaika/rl-transfer-sec-clean:/home/ssaika/rl-transfer-sec-clean/cyberwheel:$PYTHONPATH

echo "============================================================"
echo "Starting Full Training"
echo "============================================================"
echo ""

# Option 1: Train on Cyberwheel
if [ "$1" == "cyberwheel" ] || [ "$1" == "cw" ] || [ -z "$1" ]; then
    echo "============================================================"
    echo "Training on Cyberwheel Credential Preference Scenario"
    echo "============================================================"
    export CW_ENV_YAML=credential_preference_scenario.yaml
    
    # Allow custom training steps
    TRAIN_STEPS=${CW_TRAIN_STEPS:-50000}
    export CW_TRAIN_STEPS=$TRAIN_STEPS
    
    echo "Environment: credential_preference_scenario.yaml"
    echo "Training steps: $TRAIN_STEPS"
    echo ""
    
    python train/train_cw_ppo_minimal.py
    
    echo ""
    echo "✓ Cyberwheel training complete!"
    echo "  Model saved to: artifacts/policies/cw_ppo_minimal.zip"
    echo ""
fi

# Option 2: Train on CyberBattleSim
if [ "$1" == "cyberbattlesim" ] || [ "$1" == "cbs" ] || [ -z "$1" ]; then
    echo "============================================================"
    echo "Training on CyberBattleSim Flat Network Scenario"
    echo "============================================================"
    export CBS_ENV=CyberBattleFlat-v0
    export CBS_FLAT_NODES=20
    export CBS_CRED_REUSE_PROB=0.6
    export CBS_EXPLOIT_PROB=0.3
    
    echo "Environment: CyberBattleFlat-v0"
    echo "Nodes: 20"
    echo "Credential reuse prob: 0.6"
    echo "Exploit prob: 0.3"
    echo ""
    
    python train/train_cbs_ppo_minimal.py
    
    echo ""
    echo "✓ CyberBattleSim training complete!"
    echo "  Model saved to: artifacts/policies/cbs_ppo_minimal.zip"
    echo ""
fi

echo "============================================================"
echo "Training Complete!"
echo "============================================================"

