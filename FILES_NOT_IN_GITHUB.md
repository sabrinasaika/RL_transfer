# Files Used in Experiment But NOT Pushed to GitHub

This document lists files that are **required for the experiment** but are **excluded from Git** (via `.gitignore`).

---

## 🔴 Critical Files (Required to Run Experiment)

These files are **generated during the experiment** and must be created before running:

### 1. **DAPN Encoder Checkpoint**
- **File**: `artifacts/transfer_models/dapn_encoder.pt`
- **Excluded by**: `*.pt` in `.gitignore`
- **Size**: ~7-8 MB
- **Created by**: `train_dapn_encoder.py`
- **Used by**: 
  - `train/train_cw_ppo_with_dapn.py`
  - `evaluate_transfer.py`
- **What it contains**: Trained encoder weights for domain adaptation between Cyberwheel and CyberBattleSim

**To generate:**
```bash
python train_dapn_encoder.py --num-samples 1000 --epochs 50
```

---

### 2. **Trained PPO Policy**
- **File**: `artifacts/policies/cw_ppo_dapn.zip`
- **Excluded by**: Located in `artifacts/` (may be excluded if entire directory is ignored)
- **Size**: ~1-5 MB
- **Created by**: `train/train_cw_ppo_with_dapn.py`
- **Used by**: `evaluate_transfer.py`
- **What it contains**: Trained PPO agent weights trained on Cyberwheel with DAPN

**To generate:**
```bash
python train/train_cw_ppo_with_dapn.py
```

---

## 📊 Optional/Intermediate Files

These files may be created during development but are not strictly required:

### 3. **Training Data Files**
- **Files**: 
  - `artifacts/training_data/cbs_transitions.pkl`
  - `artifacts/training_data/cw_transitions.pkl`
  - `artifacts/training_data/*.npz` (saved observations)
- **Excluded by**: `data/` and `datasets/` in `.gitignore`
- **Created by**: `collect_more_training_data.py`, `train_dapn_encoder.py`
- **Purpose**: Pre-collected observations/transitions for training

---

### 4. **Other Encoder Variants**
- **Files**:
  - `artifacts/transfer_models/cbs_encoder.pt`
  - `artifacts/transfer_models/cw_red_agent_encoder.pt`
  - `artifacts/transfer_models/full_obs_encoder.pt`
  - `artifacts/transfer_models/demo_encoder.pt`
- **Excluded by**: `*.pt` in `.gitignore`
- **Purpose**: Alternative encoder implementations or experimental versions

---

### 5. **Other Trained Policies**
- **Files**:
  - `artifacts/policies/cw_ppo_minimal.zip`
  - `artifacts/policies/cbs_ppo_final.zip`
  - `artifacts/policies/cw_ppo_wandb/*.zip`
  - `artifacts/policies/full_obs_policy/`
- **Excluded by**: Located in `artifacts/`
- **Purpose**: Alternative training runs or experimental policies

---

### 6. **Training Logs and Metrics**
- **Directories**:
  - `wandb_runs/` - Weights & Biases logs
  - `runs/` - TensorBoard logs
  - `artifacts/wandb/` - W&B artifacts
  - `artifacts/policies/*/tensorboard/` - TensorBoard event files
- **Excluded by**: Explicitly in `.gitignore`
- **Purpose**: Training metrics, loss curves, evaluation results

---

### 7. **Checkpoints Directory**
- **Directory**: `checkpoints/`
- **Excluded by**: `checkpoints/` in `.gitignore`
- **Purpose**: Intermediate model checkpoints during training

---

### 8. **Models Directory**
- **Directory**: `models/`
- **Excluded by**: `models/` in `.gitignore`
- **Purpose**: General model storage (if used)

---

## 📝 Summary

### Files Required to Run Full Experiment:

| File | Status | How to Generate |
|------|--------|----------------|
| `artifacts/transfer_models/dapn_encoder.pt` | ❌ Not in Git | Run `train_dapn_encoder.py` |
| `artifacts/policies/cw_ppo_dapn.zip` | ❌ Not in Git | Run `train/train_cw_ppo_with_dapn.py` |

### Why These Files Are Excluded:

1. **Size**: Model files are large (MB to GB)
2. **Regeneratable**: Can be recreated by running training scripts
3. **Version Control**: Binary files don't diff well in Git
4. **Storage**: Would bloat repository size

---

## 🚀 For New Users

When cloning the repository, you need to:

1. **Generate the DAPN encoder** (if not provided separately):
   ```bash
   python train_dapn_encoder.py --num-samples 1000 --epochs 50
   ```

2. **Or use pre-trained encoder** (if available elsewhere):
   - Download from shared storage
   - Place in `artifacts/transfer_models/dapn_encoder.pt`

3. **Train the policy** (or use pre-trained):
   ```bash
   python train/train_cw_ppo_with_dapn.py
   ```

---

## 💡 Recommendations

### Option 1: Use Git LFS (Large File Storage)
If you want to track these files:
```bash
git lfs track "*.pt"
git lfs track "*.zip"
git add .gitattributes
```

### Option 2: External Storage
- Store models in cloud storage (S3, Google Drive, etc.)
- Provide download links in README
- Include checksums for verification

### Option 3: Documentation
- Document exact training commands to reproduce
- Include training hyperparameters
- Note expected file sizes

---

## 🔍 How to Check What's Excluded

```bash
# See what files are ignored
git status --ignored

# Check if a specific file would be ignored
git check-ignore -v artifacts/transfer_models/dapn_encoder.pt
```

---

## 📋 Complete List of Excluded Patterns

From `.gitignore`:
- `*.pt` - All PyTorch model files
- `*.pth` - PyTorch checkpoints
- `*.h5` - HDF5 files
- `*.onnx` - ONNX model files
- `checkpoints/` - Checkpoint directory
- `models/` - Models directory
- `data/` - Data directory
- `datasets/` - Datasets directory
- `artifacts/wandb/` - W&B logs
- `wandb_runs/` - W&B runs
- `runs/` - Training runs
- `outputs/` - Output directory

---

**Note**: The experiment is designed to be reproducible - all excluded files can be regenerated by running the training scripts with the same parameters.



