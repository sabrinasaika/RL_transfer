#!/bin/bash
# Fix script for Cyberwheel issues

echo "============================================================"
echo "Fixing Cyberwheel Environment"
echo "============================================================"

cd /home/ssaika/rl-transfer-sec-clean

# Set Python paths
export PYTHONPATH=/home/ssaika/rl-transfer-sec-clean:/home/ssaika/rl-transfer-sec-clean/cyberwheel:$PYTHONPATH

# Set environment variable
export CW_ENV_YAML=credential_preference_scenario.yaml

echo "✓ Set PYTHONPATH"
echo "✓ Set CW_ENV_YAML=$CW_ENV_YAML"
echo ""

echo "Running test..."
python test_cyberwheel_only.py

echo ""
echo "============================================================"
echo "If it still fails, check the error message above"
echo "============================================================"

