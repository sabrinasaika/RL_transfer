# Files Used in Transfer Learning Pipeline

## 📋 Overview

This document lists all files used in the transfer learning pipeline from Cyberwheel to CyberBattleSim.

---

## 🔧 Main Pipeline Scripts

### 1. **`collect_more_training_data.py`**
**Purpose**: Collects training data from both environments

**What it does**:
- Collects transitions (obs, action, next_obs, reward, done) from CBS
- Collects transitions from Cyberwheel (using trained agent or random actions)
- Saves data to pickle files

**Files it uses**:
- `adapters/unified_env.py` - UnifiedSecEnv wrapper
- `config/env_builders.py` - make_cbs_env, make_cw_env
- `eval/eval_cw_checkpoints_on_cbs.py` - infer_cyberwheel_config, RLPolicy

**Outputs**:
- `artifacts/training_data/cbs_transitions.pkl`
- `artifacts/training_data/cw_transitions.pkl`

---

### 2. **`train_full_observation_transfer.py`**
**Purpose**: Trains the encoder and dynamics model on collected data

**What it does**:
- Trains `CBSFullObservationEncoder` and `CWFullObservationEncoder`
- Trains `DynamicsModel` to predict next-state features and rewards
- Can load pre-collected data or collect on-the-fly

**Files it uses**:
- `adapters/full_observation_encoder.py` - CBSFullObservationEncoder, CWFullObservationEncoder
- `adapters/transfer_encoder.py` - DynamicsModel
- `adapters/unified_env.py` - UnifiedSecEnv
- `config/env_builders.py` - make_cbs_env, make_cw_env

**Outputs**:
- `artifacts/transfer_models/cw_red_agent_encoder.pt` (or custom path)

---

### 3. **`train_policy_full_obs.py`**
**Purpose**: Trains a new PPO policy on CBS using the trained encoder

**What it does**:
- Wraps CBS environment with `FullObsWrapper`
- Uses `FullObservationTranslator` to encode observations
- Trains PPO policy with Stable-Baselines3

**Files it uses**:
- `adapters/unified_env.py` - UnifiedSecEnv
- `adapters/full_obs_translator.py` - FullObservationTranslator
- `config/env_builders.py` - make_cbs_env
- Stable-Baselines3 (PPO, MultiInputPolicy)

**Inputs**:
- `artifacts/transfer_models/cw_red_agent_encoder.pt` (encoder)

**Outputs**:
- `artifacts/policies/full_obs_policy/best_model` (trained policy)

---

### 4. **`compare_transfer_approaches.py`**
**Purpose**: Compares two transfer learning approaches

**What it does**:
- **Approach 1**: Uses Cyberwheel's policy directly on CBS (with adapter head)
- **Approach 2**: Trains new policy on CBS using encoder

**Files it uses**:
- `adapters/unified_env.py` - UnifiedSecEnv
- `adapters/full_obs_translator.py` - FullObservationTranslator
- `adapters/observation_translator.py` - ObservationTranslator (fallback)
- `config/env_builders.py` - make_cbs_env
- `eval/eval_cw_checkpoints_on_cbs.py` - load_cyberwheel_policy, infer_cyberwheel_config
- Stable-Baselines3 (PPO, MultiInputPolicy)

**Inputs**:
- Cyberwheel checkpoint (e.g., `cyberwheel/cyberwheel/data/models/CWRun_CW10_long/red_124416.pt`)
- Encoder path (e.g., `artifacts/transfer_models/cw_red_agent_encoder.pt`)

---

### 5. **`eval_full_observation_transfer.py`**
**Purpose**: Evaluates a trained policy with the encoder

**Files it uses**:
- `adapters/unified_env.py` - UnifiedSecEnv
- `adapters/full_obs_translator.py` - FullObservationTranslator
- `config/env_builders.py` - make_cbs_env

---

## 🔌 Core Adapter Files

### **`adapters/unified_env.py`**
**Purpose**: Unified environment wrapper for both CBS and Cyberwheel

**Key classes**:
- `UnifiedSecEnv` - Main wrapper that unifies observation, action, and reward spaces

**Used by**: All main scripts

---

### **`adapters/full_observation_encoder.py`**
**Purpose**: Encoders for full observations from both environments

**Key classes**:
- `CBSFullObservationEncoder` - Encodes CBS dict observations → 64-dim features
- `CWFullObservationEncoder` - Encodes Cyberwheel 701-dim vector → 64-dim features
- `UnifiedFullObservationEncoder` - Wrapper for both encoders

**Used by**: 
- `train_full_observation_transfer.py` (training)
- `adapters/full_obs_translator.py` (inference)

---

### **`adapters/full_obs_translator.py`**
**Purpose**: Translates observations using the full observation encoder

**Key classes**:
- `FullObservationTranslator` - Integrates encoder with UnifiedSecEnv

**Used by**:
- `train_policy_full_obs.py`
- `compare_transfer_approaches.py`
- `eval_full_observation_transfer.py`

---

### **`adapters/transfer_encoder.py`**
**Purpose**: Dynamics model for predicting next-state features

**Key classes**:
- `DynamicsModel` - Predicts next features and rewards from current features + action
- `ObservationEncoder` - 8-dim encoder (legacy, not used in full pipeline)
- `L2Norm` - Normalization layer

**Used by**:
- `train_full_observation_transfer.py`

---

### **`adapters/observation_translator.py`**
**Purpose**: Legacy 8-dim observation translator (fallback)

**Used by**:
- `compare_transfer_approaches.py` (as fallback)

---

### **`adapters/action_translator.py`**
**Purpose**: Translates actions between CBS and Cyberwheel action spaces

**Used by**:
- `adapters/unified_env.py`

---

### **`adapters/reward_normalizer.py`**
**Purpose**: Normalizes rewards between environments

**Used by**:
- `adapters/unified_env.py`

---

## ⚙️ Configuration Files

### **`config/env_builders.py`**
**Purpose**: Factory functions to create CBS and Cyberwheel environments

**Key functions**:
- `make_cbs_env()` - Creates CyberBattleSim environment
- `make_cw_env()` - Creates Cyberwheel environment

**Used by**: All main scripts

---

## 🔍 Evaluation Files

### **`eval/eval_cw_checkpoints_on_cbs.py`**
**Purpose**: Utilities for loading and evaluating Cyberwheel checkpoints

**Key functions**:
- `load_cyberwheel_policy()` - Loads Cyberwheel RLPolicy from checkpoint
- `infer_cyberwheel_config()` - Infers action/observation space from checkpoint

**Used by**:
- `collect_more_training_data.py`
- `compare_transfer_approaches.py`

---

## 📁 Data Files (Artifacts)

### Training Data
- `artifacts/training_data/cbs_transitions.pkl` - CBS transitions
- `artifacts/training_data/cw_transitions.pkl` - Cyberwheel transitions

### Trained Models
- `artifacts/transfer_models/cw_red_agent_encoder.pt` - Trained encoder
- `artifacts/policies/full_obs_policy/best_model` - Trained PPO policy

### Cyberwheel Checkpoints
- `cyberwheel/cyberwheel/data/models/CWRun_CW10_long/red_124416.pt` - Cyberwheel trained policy

---

## 🔄 Pipeline Flow

```
1. collect_more_training_data.py
   ↓ (generates)
   artifacts/training_data/cw_transitions.pkl

2. train_full_observation_transfer.py
   ↓ (uses cw_transitions.pkl)
   ↓ (generates)
   artifacts/transfer_models/cw_red_agent_encoder.pt

3a. compare_transfer_approaches.py (Approach 1)
    ↓ (uses encoder + Cyberwheel checkpoint)
    Direct evaluation

3b. train_policy_full_obs.py (Approach 2)
    ↓ (uses encoder)
    ↓ (generates)
    artifacts/policies/full_obs_policy/best_model
    ↓
    compare_transfer_approaches.py (Approach 2)
    ↓ (uses trained policy)
    Evaluation
```

---

## 📦 External Dependencies

### Python Libraries
- `torch` - PyTorch for neural networks
- `stable_baselines3` - PPO policy training
- `gymnasium` - RL environment interface
- `numpy` - Numerical operations

### Project Dependencies
- `cyberwheel` - Cyberwheel environment (in `cyberwheel/` directory)
- `CyberBattleSim` - CBS environment (installed via pip or local)

---

## 🎯 Quick Reference

**To collect data**:
```bash
python collect_more_training_data.py --cw_episodes 20 --cw_checkpoint <path>
```

**To train encoder**:
```bash
python train_full_observation_transfer.py --load_cw_data <path> --output_path <path>
```

**To train policy**:
```bash
python train_policy_full_obs.py --encoder_path <path>
```

**To compare approaches**:
```bash
python compare_transfer_approaches.py --cw_checkpoint <path> --encoder_path <path>
```

