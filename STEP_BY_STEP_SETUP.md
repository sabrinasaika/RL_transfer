# Step-by-Step Guide to Run the Full Project

This guide will walk you through setting up and running the complete transfer learning pipeline: **Train on Cyberwheel (Scenario 1) → Test on CyberBattleSim (Scenario 2)**.

---

## Prerequisites

Before starting, ensure you have:
- **Python 3.10** installed and available on PATH
- **Poetry** (version >= 1.5) installed for Cyberwheel dependency management
- **Graphviz** installed (for Cyberwheel visualization)
- **Git** (if cloning the repository)

---

## Step 1: Navigate to Project Directory

```bash
cd /home/ssaika/rl-transfer-sec-clean
```

---

## Step 2: Set Up Python Virtual Environment

Create and activate a virtual environment for the main project dependencies:

```bash
# Create virtual environment
python3.10 -m venv .venv

# Activate virtual environment
source .venv/bin/activate
```

**Note**: On Windows, use `.venv\Scripts\activate` instead.

---

## Step 3: Install CyberBattleSim

Install CyberBattleSim and its dependencies:

```bash
# Install CyberBattleSim in editable mode
pip install -e CyberBattleSim

# Install core dependencies
pip install gymnasium==0.29.1 stable-baselines3==2.3.2 numpy==1.26.4 pandas==2.2.2

# Install PyTorch (CPU version - adjust if you have GPU)
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu
```

**Note**: If you have a CUDA-enabled GPU, install the appropriate PyTorch version from [pytorch.org](https://pytorch.org/).

---

## Step 4: Set Up Cyberwheel with Poetry

Cyberwheel uses Poetry for dependency management:

```bash
# Navigate to cyberwheel directory
cd cyberwheel

# Install dependencies using Poetry
poetry install

# Verify installation
poetry run python3 -m cyberwheel -h
```

**Note**: If you don't have Poetry installed, install it from [python-poetry.org](https://python-poetry.org/docs/#installation).

---

## Step 5: Install Additional Dependencies

Return to the project root and install any additional dependencies needed for the transfer learning pipeline:

```bash
# Return to project root
cd /home/ssaika/rl-transfer-sec-clean

# Ensure virtual environment is activated
source .venv/bin/activate

# Install additional dependencies (if not already installed)
pip install tqdm  # Progress bars
```

---

## Step 6: Set Environment Variables

Set the required environment variables:

```bash
# Set Python path to include cyberwheel
export PYTHONPATH=/home/ssaika/rl-transfer-sec-clean/cyberwheel:$PYTHONPATH

# Set Cyberwheel environment configuration
export CW_ENV_YAML=credential_preference_scenario.yaml

# Optional: Set training parameters
export CW_TRAIN_STEPS=1000  # Number of training steps (default: 1000)

# Optional: Set CyberBattleSim environment parameters
export CBS_ENV=CyberBattleFlat-v0
export CBS_FLAT_NODES=20
export CBS_CRED_REUSE_PROB=0.6
export CBS_EXPLOIT_PROB=0.3
```

**Note**: You can add these to your `~/.bashrc` or `~/.zshrc` to make them persistent, or create a setup script.

---

## Step 7: Create Required Directories

Ensure the output directories exist:

```bash
# Create artifacts directories
mkdir -p artifacts/transfer_models
mkdir -p artifacts/policies
```

---

## Step 8: Run the Complete Pipeline

You have two options:

### Option A: Run the Automated Script (Recommended)

The easiest way is to use the provided automation script:

```bash
# Make sure you're in the project root
cd /home/ssaika/rl-transfer-sec-clean

# Ensure virtual environment is activated
source .venv/bin/activate

# Ensure PYTHONPATH is set
export PYTHONPATH=/home/ssaika/rl-transfer-sec-clean/cyberwheel:$PYTHONPATH

# Run the complete pipeline
bash run_transfer_learning.sh
```

This script will:
1. **Train DAPN encoder** (if it doesn't exist) - adapts observations between Cyberwheel and CyberBattleSim
2. **Train PPO agent on Cyberwheel** (Scenario 1) with DAPN wrapper
3. **Evaluate transfer** to CyberBattleSim (Scenario 2)

### Option B: Run Steps Manually

If you prefer to run each step individually:

#### Step 8a: Train DAPN Encoder

```bash
python train_dapn_encoder.py --num-samples 1000 --epochs 50
```

**Output**: `artifacts/transfer_models/dapn_encoder.pt`

#### Step 8b: Train PPO Agent on Cyberwheel

```bash
export CW_ENV_YAML=credential_preference_scenario.yaml
export CW_TRAIN_STEPS=1000
export DAPN_ENCODER_PATH=artifacts/transfer_models/dapn_encoder.pt

python train/train_cw_ppo_with_dapn.py
```

**Output**: `artifacts/policies/cw_ppo_dapn.zip`

#### Step 8c: Evaluate Transfer to CyberBattleSim

```bash
python evaluate_transfer.py
```

This will load the trained model and evaluate it on CyberBattleSim, reporting performance metrics.

---

## Step 9: Verify Results

After running the pipeline, check the output:

```bash
# Check that DAPN encoder was created
ls -lh artifacts/transfer_models/dapn_encoder.pt

# Check that trained policy was created
ls -lh artifacts/policies/cw_ppo_dapn.zip
```

The evaluation script will print performance metrics to the console.

---

## Troubleshooting

### Issue: Poetry not found
**Solution**: Install Poetry:
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

### Issue: Graphviz not found
**Solution**: Install Graphviz:
- **Ubuntu/Debian**: `sudo apt-get install graphviz`
- **macOS**: `brew install graphviz`
- **Windows**: Download from [graphviz.org](https://graphviz.org/download/)

### Issue: ModuleNotFoundError for cyberwheel
**Solution**: Ensure PYTHONPATH is set:
```bash
export PYTHONPATH=/home/ssaika/rl-transfer-sec-clean/cyberwheel:$PYTHONPATH
```

### Issue: CyberBattleSim environment not found
**Solution**: Ensure CyberBattleSim is installed in editable mode:
```bash
pip install -e CyberBattleSim
```

### Issue: CUDA/GPU errors
**Solution**: If you don't have a GPU, ensure you installed the CPU version of PyTorch. If you have a GPU, install the appropriate CUDA version of PyTorch.

---

## Expected Runtime

- **DAPN Encoder Training**: ~5-15 minutes (depending on hardware and number of samples)
- **PPO Training on Cyberwheel**: ~10-30 minutes (depending on CW_TRAIN_STEPS)
- **Transfer Evaluation**: ~1-5 minutes (depending on number of evaluation episodes)

**Total**: Approximately 20-50 minutes for a complete run.

---

## Customization

### Adjust Training Steps

To train for more steps (better performance, longer runtime):

```bash
export CW_TRAIN_STEPS=10000  # Instead of default 1000
```

### Adjust DAPN Encoder Training

Modify the encoder training parameters:

```bash
python train_dapn_encoder.py --num-samples 5000 --epochs 100
```

### Change CyberBattleSim Configuration

Modify environment variables:

```bash
export CBS_FLAT_NODES=50  # More nodes
export CBS_CRED_REUSE_PROB=0.8  # Higher credential reuse
export CBS_EXPLOIT_PROB=0.5  # Higher exploit probability
```

---

## File Structure After Running

```
artifacts/
├── transfer_models/
│   └── dapn_encoder.pt          # Trained DAPN encoder
└── policies/
    └── cw_ppo_dapn.zip          # Trained PPO policy
```

---

## Next Steps

- Review the evaluation results printed to the console
- Experiment with different hyperparameters
- Try different scenarios by modifying configuration files
- Visualize training progress (if TensorBoard/W&B logging is enabled)

---

## Summary

The complete workflow:
1. ✅ Set up Python 3.10 virtual environment
2. ✅ Install CyberBattleSim
3. ✅ Set up Cyberwheel with Poetry
4. ✅ Set environment variables
5. ✅ Run `run_transfer_learning.sh` or execute steps manually
6. ✅ Review results

**Main Entry Point**: `run_transfer_learning.sh`

For more details on individual components, see:
- `FILES_USED_IN_EXPERIMENT.md` - Overview of all files
- `HOW_TO_RUN_TRANSFER.md` - Alternative transfer learning approach
- `README.md` - General project information

