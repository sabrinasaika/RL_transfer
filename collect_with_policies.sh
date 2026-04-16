#!/bin/bash
# Quick script to collect data with trained policies

# Default policies (adjust paths if needed)
CW_POLICY=${1:-"artifacts/policies/cw_ppo_dapn.zip"}
CBS_POLICY=${2:-"artifacts/policies/cbs_ppo_final.zip"}
OUTPUT_FILE=${3:-"artifacts/training_data/episodic_obs_policy.npz"}

echo "Collecting data with trained policies..."
echo "  Cyberwheel policy: ${CW_POLICY}"
echo "  CBS policy: ${CBS_POLICY}"
echo "  Output: ${OUTPUT_FILE}"
echo ""

python collect_episodic_data_with_policies.py \
    --cw-policy ${CW_POLICY} \
    --cbs-policy ${CBS_POLICY} \
    --cw-episodes 50 \
    --cbs-episodes 50 \
    --max-steps 200 \
    --output ${OUTPUT_FILE}

echo ""
echo "✅ Done! To train with this data:"
echo "   python train_dapn_encoder_episodic.py --load-data ${OUTPUT_FILE} --n-sc 20"
