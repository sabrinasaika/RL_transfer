#!/bin/bash
# Complete fix and test script for Cyberwheel

set -e  # Exit on error

cd /home/ssaika/rl-transfer-sec-clean

echo "============================================================"
echo "Setting up environment..."
echo "============================================================"

# Set Python paths - CRITICAL for Cyberwheel to work
export PYTHONPATH=/home/ssaika/rl-transfer-sec-clean:/home/ssaika/rl-transfer-sec-clean/cyberwheel:$PYTHONPATH

# Set environment variable
export CW_ENV_YAML=credential_preference_scenario.yaml

echo "✓ PYTHONPATH set"
echo "✓ CW_ENV_YAML=$CW_ENV_YAML"
echo ""

echo "============================================================"
echo "Running Cyberwheel test..."
echo "============================================================"

python test_cyberwheel_only.py

echo ""
echo "============================================================"
echo "Done!"
echo "============================================================"

