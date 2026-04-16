# How to Run the Transfer Learning Experiment

This guide explains how to run the cybersecurity transfer learning experiment that trains an RL agent on Cyberwheel and transfers it to CyberBattleSim using DAPN.

## Quick Start (If Already Set Up)

If dependencies are already installed:

```bash
# Navigate to project directory
cd /home/ssaika/rl-transfer-sec-clean

# Set environment variables
export PYTHONPATH=/home/ssaika/rl-transfer-sec-clean/cyberwheel:$PYTHONPATH
export CW_ENV_YAML=credential_preference_scenario.yaml

# Run the complete pipeline
bash run_transfer_learning.sh
```

This will:
1. Train DAPN encoder (if not already trained)
2. Train PPO agent on Cyberwheel (1000 steps - quick test)
3. Evaluate transfer to CyberBattleSim

---

## Full Setup (First Time)

### 1. Prerequisites

- **Python 3.10+** (3.11.7 is currently installed)
- **Poetry** (>= 1.5) - for Cyberwheel dependencies
- **Graphviz** - for network visualization

### 2. Install Dependencies

```bash
# Navigate to project directory
cd /home/ssaika/rl-transfer-sec-clean

# Create and activate virtual environment (if not already created)
python3 -m venv .venv
source .venv/bin/activate

# Install CyberBattleSim
pip install -e CyberBattleSim

# Install core dependencies
pip install gymnasium==0.29.1 stable-baselines3==2.3.2 numpy==1.26.4 pandas==2.2.2
pip install tqdm pydantic jsonpickle python-dotenv networkx pyyaml

# Install PyTorch (CPU version)
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu

# Set up Cyberwheel
cd cyberwheel
poetry install
cd ..
```

### 3. Set Environment Variables

```bash
export PYTHONPATH=/home/ssaika/rl-transfer-sec-clean/cyberwheel:$PYTHONPATH
export CW_ENV_YAML=credential_preference_scenario.yaml
```

---

## Running Options

### Option 1: Quick Test (Default - 1000 steps)

**Fast but limited learning** - Good for testing the pipeline:

```bash
cd /home/ssaika/rl-transfer-sec-clean
export PYTHONPATH=/home/ssaika/rl-transfer-sec-clean/cyberwheel:$PYTHONPATH
export CW_ENV_YAML=credential_preference_scenario.yaml
bash run_transfer_learning.sh
```

**Time**: ~2-5 minutes  
**Result**: Agent learns basic behavior but may not be optimal

---

### Option 2: Proper Training (50,000 steps)

**Recommended for actual experiments** - Agent learns meaningful policy:

```bash
cd /home/ssaika/rl-transfer-sec-clean
export PYTHONPATH=/home/ssaika/rl-transfer-sec-clean/cyberwheel:$PYTHONPATH
export CW_ENV_YAML=credential_preference_scenario.yaml
export CW_TRAIN_STEPS=50000  # Increase from default 1000
bash run_transfer_learning.sh
```

**Time**: ~30-60 minutes  
**Result**: Agent learns proper action sequences

---

### Option 3: Step-by-Step Manual Execution

Run each step individually for more control:

#### Step 1: Train DAPN Encoder
```bash
export PYTHONPATH=/home/ssaika/rl-transfer-sec-clean/cyberwheel:$PYTHONPATH
python train_dapn_encoder.py --num-samples 1000 --epochs 50
```

#### Step 2: Train Agent on Cyberwheel
```bash
export PYTHONPATH=/home/ssaika/rl-transfer-sec-clean/cyberwheel:$PYTHONPATH
export CW_ENV_YAML=credential_preference_scenario.yaml
export CW_TRAIN_STEPS=50000
export DAPN_ENCODER_PATH=artifacts/transfer_models/dapn_encoder.pt
python train/train_cw_ppo_with_dapn.py
```

#### Step 3: Evaluate Transfer
```bash
export PYTHONPATH=/home/ssaika/rl-transfer-sec-clean/cyberwheel:$PYTHONPATH
python evaluate_transfer.py
```

---

### Option 4: Train Both Scenarios Separately

Train and evaluate each scenario independently (without transfer):

```bash
cd /home/ssaika/rl-transfer-sec-clean
export PYTHONPATH=/home/ssaika/rl-transfer-sec-clean/cyberwheel:$PYTHONPATH
bash run_both_scenarios.sh
```

This trains:
- Cyberwheel scenario → `artifacts/policies/cw_ppo_minimal.zip`
- CyberBattleSim scenario → `artifacts/policies/cbs_ppo_minimal.zip`

Then evaluate:
```bash
python evaluate_scenarios.py
```

---

## Customizing Training

### Change Training Steps

```bash
export CW_TRAIN_STEPS=100000  # 100k steps for thorough training
bash run_transfer_learning.sh
```

### Change DAPN Encoder Parameters

Edit `run_transfer_learning.sh` or run manually:
```bash
python train_dapn_encoder.py \
    --num-samples 5000 \      # More samples
    --epochs 100 \             # More epochs
    --batch-size 128 \         # Larger batch
    --feature-size 256         # Feature dimension
```

### Use Different Scenario

```bash
export CW_ENV_YAML=your_scenario.yaml
bash run_transfer_learning.sh
```

---

## Output Files

After running, you'll find:

- **DAPN Encoder**: `artifacts/transfer_models/dapn_encoder.pt`
- **Trained Policy**: `artifacts/policies/cw_ppo_dapn.zip`
- **Training Logs**: Check console output for metrics

---

## Troubleshooting

### Import Errors

```bash
# Make sure PYTHONPATH is set
export PYTHONPATH=/home/ssaika/rl-transfer-sec-clean/cyberwheel:$PYTHONPATH

# Verify Cyberwheel is installed
python -c "import cyberwheel; print('OK')"
```

### Missing Dependencies

```bash
# Reinstall CyberBattleSim
pip install -e CyberBattleSim

# Reinstall Cyberwheel
cd cyberwheel && poetry install && cd ..
```

### Constant Rewards Issue

If you see constant rewards (22.00), see `EXPLANATION_CONSTANT_REWARD.md`:
- Increase training steps: `export CW_TRAIN_STEPS=50000`
- The evaluation now uses stochastic actions by default

### Out of Memory

Reduce batch size or training steps:
```bash
export CW_TRAIN_STEPS=10000  # Smaller training
```

---

## Expected Results

### Quick Test (1000 steps)
- Training time: ~2-5 minutes
- Episode rewards: Variable but low (129-157)
- Agent behavior: Basic, may repeat same actions

### Proper Training (50,000+ steps)
- Training time: ~30-60 minutes
- Episode rewards: Higher and more consistent
- Agent behavior: Learns proper action sequences (discovery → lateral move → privilege escalation → impact)

---

## Next Steps

1. **Monitor Training**: Watch console output for episode rewards increasing
2. **Evaluate Results**: Check `evaluate_transfer.py` output
3. **Compare Scenarios**: Use `evaluate_scenarios.py` to compare both scenarios
4. **Diagnose Issues**: Use `diagnose_constant_reward.py` if rewards are constant

---

## Quick Reference

```bash
# Full pipeline (quick test)
export PYTHONPATH=/home/ssaika/rl-transfer-sec-clean/cyberwheel:$PYTHONPATH
export CW_ENV_YAML=credential_preference_scenario.yaml
bash run_transfer_learning.sh

# Full pipeline (proper training)
export PYTHONPATH=/home/ssaika/rl-transfer-sec-clean/cyberwheel:$PYTHONPATH
export CW_ENV_YAML=credential_preference_scenario.yaml
export CW_TRAIN_STEPS=50000
bash run_transfer_learning.sh

# Just evaluate existing model
export PYTHONPATH=/home/ssaika/rl-transfer-sec-clean/cyberwheel:$PYTHONPATH
python evaluate_transfer.py
```
