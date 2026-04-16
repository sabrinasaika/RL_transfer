#!/bin/bash
# Collect a smaller dataset to avoid disk space issues

echo "Collecting smaller dataset (30 episodes each) to save space..."
echo "This should give ~5000-6000 samples per domain (enough for training)"
echo ""

python collect_episodic_data_with_policies.py \
    --cw-policy artifacts/policies/cw_ppo_dapn.zip \
    --cbs-policy artifacts/policies/cbs_ppo_final.zip \
    --cw-episodes 30 \
    --cbs-episodes 30 \
    --max-steps 150 \
    --output artifacts/training_data/episodic_obs_policy_small.npz

echo ""
echo "✅ Done! File should be much smaller (~5-10GB instead of 300GB+)"
