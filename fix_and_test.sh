#!/bin/bash
# Fix script to ensure Cyberwheel works correctly

cd /home/ssaika/rl-transfer-sec-clean

# Set Python paths
export PYTHONPATH=/home/ssaika/rl-transfer-sec-clean:/home/ssaika/rl-transfer-sec-clean/cyberwheel:$PYTHONPATH

# Set environment variables
export CW_ENV_YAML=credential_preference_scenario.yaml
export CBS_ENV=CyberBattleFlat-v0
export CBS_FLAT_NODES=20
export CBS_CRED_REUSE_PROB=0.6
export CBS_EXPLOIT_PROB=0.3

echo "============================================================"
echo "Running diagnostic..."
echo "============================================================"
python diagnose_cyberwheel.py

echo ""
echo "============================================================"
echo "Running test scenarios..."
echo "============================================================"
python test_scenarios.py

