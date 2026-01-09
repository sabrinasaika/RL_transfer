RL Transfer Learning: Cyberwheel → CyberBattleSim

**CyberBattleSim** using DAPN for domain adaptation.
## Prerequisites

- **Python 3.10**
- **Poetry** (>= 1.5)
- **Graphviz**

Setup Project
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

**What you'll see:**
1. **Data Collection**: Progress bars showing observation collection from each domain
2. **Training Setup**: Summary of samples per domain and training method
3. **Training Progress**: Every 10 epochs, you'll see:
   - `Epoch X/50: Loss=0.XXXX, Adv=0.XXXX`
   - `Loss`: Encoder alignment loss (lower is better)
   - `Adv`: Adversarial discriminator loss
4. **Final Message**: `Saved trained encoder to artifacts/transfer_models/dapn_encoder.pt`

**Training Parameters:**
- `--num-samples`: Number of observations per domain (default: 1000)
- `--epochs`: Training epochs (default: 50)
- `--batch-size`: Batch size (default: 64)
- `--lr`: Learning rate (default: 0.001)
- `--save-encoder`: Path to save encoder (default: `artifacts/transfer_models/dapn_encoder.pt`)

**Example output:**
```
Collecting observations from Source domain (Cyberwheel)...
100%|████████| 1000/1000 [00:30<00:00, 33.2it/s]

Domain Alignment Training Setup:
  Source domain (Cyberwheel): 1000 samples
  Target domain (Normal CyberBattleSim): 1000 samples
  Validation domain: 500 samples
  Method: Adversarial Domain Adaptation (DANN)

Training for 50 epochs...
Epoch 10/50: Loss=0.8234, Adv=0.4567
Epoch 20/50: Loss=0.7123, Adv=0.3890
Epoch 30/50: Loss=0.6543, Adv=0.3456
Epoch 40/50: Loss=0.6123, Adv=0.3123
Epoch 50/50: Loss=0.5890, Adv=0.2987
Saved trained encoder to artifacts/transfer_models/dapn_encoder.pt
Training complete!
```
