# Quick Start: How to Run DAPN

## Step 1: Basic Usage (No Training Required)

You can use DAPN immediately with random initialization:

```python
from adapters.unified_env import UnifiedSecEnv
from adapters.dapn_env_wrapper import DAPNEnvWrapper
from config.env_builders import make_cbs_env

# Create environment with DAPN
base_env = UnifiedSecEnv("cbs", cbs_factory=make_cbs_env)
dapn_env = DAPNEnvWrapper(base_env, use_dapn=True, feature_size=256)

# Use it normally
obs, info = dapn_env.reset()
action = dapn_env.action_space.sample()
obs, reward, done, truncated, info = dapn_env.step(action)
```

**Run this:**
```bash
python example_use_dapn.py
```

---

## Step 2: Train a DAPN Encoder (Recommended)

### Option A: Quick Training (100 samples, 10 epochs)
```bash
python train_dapn_encoder.py \
    --num-samples 100 \
    --epochs 10 \
    --batch-size 32 \
    --feature-size 256 \
    --save-encoder artifacts/transfer_models/dapn_encoder.pt
```

### Option B: Full Training (1000 samples, 50 epochs)
```bash
python train_dapn_encoder.py \
    --num-samples 1000 \
    --epochs 50 \
    --batch-size 64 \
    --feature-size 256 \
    --save-encoder artifacts/transfer_models/dapn_encoder.pt
```

### Option C: Save/Load Observations Separately
```bash
# Step 1: Collect observations
python train_dapn_encoder.py \
    --num-samples 1000 \
    --save-data artifacts/training_data/dapn_obs.npz

# Step 2: Train (can run multiple times with different settings)
python train_dapn_encoder.py \
    --load-data artifacts/training_data/dapn_obs.npz \
    --epochs 50 \
    --save-encoder artifacts/transfer_models/dapn_encoder.pt
```

---

## Step 3: Use Trained Encoder

### In Your Code:
```python
from adapters.dapn_env_wrapper import DAPNEnvWrapper
from adapters.unified_env import UnifiedSecEnv
from config.env_builders import make_cbs_env

base_env = UnifiedSecEnv("cbs", cbs_factory=make_cbs_env)
dapn_env = DAPNEnvWrapper(
    base_env,
    encoder_path="artifacts/transfer_models/dapn_encoder.pt",  # Use trained encoder
    feature_size=256,
    use_dapn=True
)
```

### In test_transfer_simple.py:
```python
test_transfer_learning_concept(
    use_dapn=True,
    dapn_encoder_path="artifacts/transfer_models/dapn_encoder.pt"
)
```

---

## Step 4: Train a Policy with DAPN

```python
from stable_baselines3 import PPO
from adapters.unified_env import UnifiedSecEnv
from adapters.dapn_env_wrapper import DAPNEnvWrapper
from config.env_builders import make_cbs_env

# Create environment with DAPN
base_env = UnifiedSecEnv("cbs", cbs_factory=make_cbs_env)
dapn_env = DAPNEnvWrapper(
    base_env,
    encoder_path="artifacts/transfer_models/dapn_encoder.pt",
    feature_size=256
)

# Train PPO
model = PPO("MultiInputPolicy", dapn_env, verbose=1)
model.learn(total_timesteps=100000)
model.save("artifacts/policies/ppo_dapn")
```

---

## Quick Test Commands

### Test 1: Basic DAPN (no training)
```bash
python test_transfer_simple_dapn.py
```

### Test 2: Examples
```bash
python example_use_dapn.py
```

### Test 3: With your existing test
```bash
# Edit test_transfer_simple.py to set use_dapn=True
python test_transfer_simple.py
```

---

## Common Issues & Solutions

### Issue: "Encoder not found"
**Solution:** Train an encoder first (Step 2) or use `encoder_path=None` for random initialization

### Issue: "Observation shape mismatch"
**Solution:** Make sure `feature_size` matches between training and usage (default: 256)

### Issue: "CUDA out of memory"
**Solution:** Use CPU: `device='cpu'` or reduce batch size: `--batch-size 32`

---

## Full Workflow Example

```bash
# 1. Train DAPN encoder
python train_dapn_encoder.py --num-samples 1000 --epochs 50

# 2. Test it works
python test_transfer_simple_dapn.py

# 3. Use in your code
# Edit your script to use DAPNEnvWrapper with the trained encoder path
```

