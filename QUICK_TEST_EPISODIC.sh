#!/bin/bash
# Quick test script for episodic training with reduced samples

echo "Quick episodic training test (reduced samples for speed)"
echo "========================================================="

# Step 1: Collect observations (reduced samples)
python train_dapn_encoder.py \
    --num-samples 200 \
    --save-data artifacts/quick_test_obs.npz \
    --episodic

# Step 2: Train with episodic structure (fewer iterations)
python train_dapn_encoder.py \
    --load-data artifacts/quick_test_obs.npz \
    --episodic \
    --iterations 50 \
    --n-sc 7 \
    --n-dc 5 \
    --k 5 \
    --query 10 \
    --save-encoder artifacts/transfer_models/dapn_encoder_episodic_quick.pt

echo "Done! Check artifacts/transfer_models/dapn_encoder_episodic_quick.pt"
