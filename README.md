# RL Transfer Learning: Cyberwheel → CyberBattleSim

**CyberBattleSim** using DAPN for domain adaptation.

## Prerequisites

- **Python 3.10**
- **Poetry** (>= 1.5)
- **Graphviz**

## Setup Project

```bash
# Navigate to project directory
cd /home/ssaika/rl-transfer-sec-clean

# Create and activate virtual environment
python3.10 -m venv .venv
source .venv/bin/activate

# Install CyberBattleSim
pip install -e CyberBattleSim
pip install gymnasium==0.29.1 stable-baselines3==2.3.2 numpy==1.26.4 pandas==2.2.2
pip install tqdm pydantic jsonpickle python-dotenv networkx pyyaml
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu

# Set up Cyberwheel
cd cyberwheel
poetry install
cd ..

# Set environment variables
export PYTHONPATH=/home/ssaika/rl-transfer-sec-clean/cyberwheel:$PYTHONPATH
export CW_ENV_YAML=credential_preference_scenario.yaml
```

## Run the Project

```bash
# Activate virtual environment
source .venv/bin/activate

# Set environment variables
export PYTHONPATH=/home/ssaika/rl-transfer-sec-clean/cyberwheel:$PYTHONPATH
export CW_ENV_YAML=credential_preference_scenario.yaml

# Run the complete pipeline
bash run_transfer_learning.sh
```

This will:

1. Train DAPN encoder (if needed)
2. Train PPO agent on Cyberwheel
3. Evaluate transfer to CyberBattleSim

## Output Files

After running:

- `artifacts/transfer_models/dapn_encoder.pt` - Trained encoder
- `artifacts/policies/cw_ppo_dapn.zip` - Trained policy

## View Encoder Training Results

To see encoder training progress and results, run the training manually:

```bash
# Activate virtual environment
source .venv/bin/activate

# Set environment variables
export PYTHONPATH=/home/ssaika/rl-transfer-sec-clean/cyberwheel:$PYTHONPATH
export CW_ENV_YAML=credential_preference_scenario.yaml

# Run encoder training with custom parameters
python train_dapn_encoder.py --num-samples 1000 --epochs 50
```

## Encoder full training (policy-based collection + episodic encoder)

Train CW and CBS policies first, then collect data with those policies and train the episodic DAPN encoder:

```bash
source .venv/bin/activate
export PYTHONPATH=/home/ssaika/rl-transfer-sec-clean/cyberwheel:$PYTHONPATH
export CW_ENV_YAML=credential_preference_scenario.yaml

# 1. Train Cyberwheel policy
python train/train_cw_ppo_very_short.py

# 2. Train CyberBattleSim policy
python train/train_cbs_ppo_very_short.py

# 3. Collect data with both policies and train the encoder
python train_dapn_encoder_episodic.py \
  --num-samples 2000 \
  --cw-policy artifacts/policies/cw_ppo_very_short.zip \
  --cbs-policy artifacts/policies/cbs_ppo_very_short.zip \
  --max-steps 200 \
  --save-data artifacts/training_data/obs_policy.npz \
  --label-mode situation_action
```

Encoder is saved to `artifacts/transfer_models/dapn_encoder_episodic.pt`. Optional: add `--deterministic-policy` for greedy actions during collection.
