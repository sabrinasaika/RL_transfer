#!/bin/bash
# Quick comparison script with fast training

cd /home/ssaika/rl-transfer-sec-clean

echo "Running quick comparison (200 steps, fast training)..."
echo ""

python compare_transfer_approaches.py \
    --cw_checkpoint /home/ssaika/rl-transfer-sec-clean/cyberwheel/cyberwheel/data/models/CWRun_CW10_long/red_124416.pt \
    --encoder_path artifacts/transfer_models/cw_red_agent_encoder.pt \
    --episodes 5 \
    --max_steps 200 \
    --train_policy \
    --train_timesteps 500

echo ""
echo "Done! Check results above."

