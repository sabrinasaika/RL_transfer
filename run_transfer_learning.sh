set -e

cd /home/ssaika/rl-transfer-sec-clean

# Set Python path
export PYTHONPATH=/home/ssaika/rl-transfer-sec-clean/cyberwheel:$PYTHONPATH



# Step 1: Train DAPN Encoder (if not exists)
DAPN_ENCODER="artifacts/transfer_models/dapn_encoder.pt"
if [ ! -f "$DAPN_ENCODER" ]; then
    echo "[Step 1/3] Training DAPN encoder..."
    echo "This adapts observations between Cyberwheel and CyberBattleSim"
    python train_dapn_encoder.py --num-samples 1000 --epochs 50
    echo "✓ DAPN encoder trained"
else
    echo "[Step 1/3] DAPN encoder already exists: $DAPN_ENCODER"
    echo "✓ Skipping DAPN training"
fi
echo ""

# Step 2: Train on Source Domain (Cyberwheel - Scenario 1) WITH DAPN
echo "[Step 2/3] Training on SOURCE domain (Cyberwheel - Scenario 1) WITH DAPN..."
export CW_ENV_YAML=credential_preference_scenario.yaml
export CW_TRAIN_STEPS=1000
export DAPN_ENCODER_PATH="$DAPN_ENCODER"
python train/train_cw_ppo_with_dapn.py
echo "✓ Training complete on Scenario 1 (with DAPN)"
echo ""

# Step 3: Test on Target Domain (CyberBattleSim - Scenario 2)
echo "[Step 3/3] Testing on TARGET domain (CyberBattleSim - Scenario 2)..."
echo "Using DAPN to adapt the model..."
python evaluate_transfer.py


