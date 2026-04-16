# How to Use DAPN Code - Complete Guide

## Quick Start

### Step 1: Train the DAPN Encoder

```bash
python train_dapn_encoder.py --num-samples 1000 --epochs 50
```

This will:
1. Collect observations from 3 domains:
   - Source: Cyberwheel
   - Target: Normal CyberBattleSim
   - Validation: CBS with Cyberwheel topology
2. Train the encoders with domain adaptation
3. Save to `artifacts/transfer_models/dapn_encoder.pt`

### Step 2: Use the Trained Encoder

```python
from adapters.dapn_env_wrapper import DAPNEnvWrapper
from adapters.unified_env import UnifiedSecEnv
from config.env_builders import make_cbs_env

# Create environment with DAPN
base_env = UnifiedSecEnv("cbs", cbs_factory=make_cbs_env)
dapn_env = DAPNEnvWrapper(
    base_env,
    encoder_path="artifacts/transfer_models/dapn_encoder.pt",
    feature_size=256
)

# Use it normally
obs, info = dapn_env.reset()
action = dapn_env.action_space.sample()
obs, reward, done, truncated, info = dapn_env.step(action)
```

---

## Detailed Usage

### Option 1: Training the Encoder

#### Basic Training
```bash
python train_dapn_encoder.py \
    --num-samples 1000 \
    --epochs 50 \
    --batch-size 64 \
    --feature-size 256 \
    --save-encoder artifacts/transfer_models/dapn_encoder.pt
```

#### Quick Test Training
```bash
python train_dapn_encoder.py \
    --num-samples 100 \
    --epochs 10 \
    --batch-size 32
```

#### Save/Load Observations Separately
```bash
# Step 1: Collect observations
python train_dapn_encoder.py \
    --num-samples 1000 \
    --save-data artifacts/training_data/dapn_observations.npz

# Step 2: Train (can run multiple times)
python train_dapn_encoder.py \
    --load-data artifacts/training_data/dapn_observations.npz \
    --epochs 50 \
    --save-encoder artifacts/transfer_models/dapn_encoder.pt
```

#### CBS-Only Training (if Cyberwheel unavailable)
```bash
python train_dapn_encoder.py \
    --num-samples 1000 \
    --epochs 50 \
    --cbs-only
```

---

### Option 2: Using the Encoder

#### Method 1: Environment Wrapper (Recommended)

```python
from adapters.unified_env import UnifiedSecEnv
from adapters.dapn_env_wrapper import DAPNEnvWrapper
from config.env_builders import make_cbs_env
from stable_baselines3 import PPO

# Create environment with DAPN
base_env = UnifiedSecEnv("cbs", cbs_factory=make_cbs_env)
dapn_env = DAPNEnvWrapper(
    base_env,
    encoder_path="artifacts/transfer_models/dapn_encoder.pt",  # Your trained encoder
    feature_size=256,
    use_dapn=True
)

# Train a policy
model = PPO("MultiInputPolicy", dapn_env, verbose=1)
model.learn(total_timesteps=100000)
model.save("artifacts/policies/ppo_with_dapn")
```

#### Method 2: Direct Translator

```python
from adapters.dapn_observation_encoder import DAPNObservationTranslator
import numpy as np

# Create translator
translator = DAPNObservationTranslator(
    use_dapn=True,
    encoder_path="artifacts/transfer_models/dapn_encoder.pt",
    feature_size=256
)

# Encode CBS observation
cbs_obs = {
    "discovered_node_count": 5,
    "nodes_privilegelevel": np.array([0, 1, 0, 1, 0]),
    "credential_cache_length": 2
}
encoded = translator.from_cbs(cbs_obs)  # Returns 256-dim array
print(f"Encoded shape: {encoded.shape}")  # (256,)

# Encode Cyberwheel observation
cw_obs = np.array([1, 1, 1, 1, 0, 0, 0] * 3 + [2])
encoded = translator.from_cw(cw_obs)  # Returns 256-dim array
print(f"Encoded shape: {encoded.shape}")  # (256,)
```

#### Method 3: Replace Observation Translator

```python
from adapters.unified_env import UnifiedSecEnv
from adapters.dapn_observation_encoder import DAPNObservationTranslator
from config.env_builders import make_cbs_env

# Create environment
env = UnifiedSecEnv("cbs", cbs_factory=make_cbs_env)

# Replace observation translator with DAPN
env.obs_t = DAPNObservationTranslator(
    use_dapn=True,
    encoder_path="artifacts/transfer_models/dapn_encoder.pt",
    feature_size=256
)

# Use environment
obs, info = env.reset()
# obs will now be 256-dim instead of 8-dim
```

---

### Option 3: Complete Training Pipeline

```python
#!/usr/bin/env python3
"""Complete example: Train encoder, then train policy"""

from stable_baselines3 import PPO
from adapters.unified_env import UnifiedSecEnv
from adapters.dapn_env_wrapper import DAPNEnvWrapper
from config.env_builders import make_cbs_env
import subprocess
import os

# Step 1: Train DAPN encoder
print("Step 1: Training DAPN encoder...")
subprocess.run([
    "python", "train_dapn_encoder.py",
    "--num-samples", "1000",
    "--epochs", "50",
    "--save-encoder", "artifacts/transfer_models/dapn_encoder.pt"
])

# Step 2: Train policy with DAPN
print("\nStep 2: Training policy with DAPN...")
base_env = UnifiedSecEnv("cbs", cbs_factory=make_cbs_env)
dapn_env = DAPNEnvWrapper(
    base_env,
    encoder_path="artifacts/transfer_models/dapn_encoder.pt",
    feature_size=256
)

model = PPO("MultiInputPolicy", dapn_env, verbose=1)
model.learn(total_timesteps=100000)
model.save("artifacts/policies/ppo_dapn")

print("\nDone! Trained policy saved to artifacts/policies/ppo_dapn")
```

---

## Command Line Options

### train_dapn_encoder.py

```bash
python train_dapn_encoder.py [OPTIONS]

Options:
  --num-samples N        Number of samples per domain (default: 1000)
  --epochs N            Training epochs (default: 50)
  --batch-size N        Batch size (default: 64)
  --feature-size N      Feature space size (default: 256)
  --lr FLOAT            Learning rate (default: 0.001)
  --save-encoder PATH   Path to save encoder (default: artifacts/transfer_models/dapn_encoder.pt)
  --save-data PATH      Path to save collected observations
  --load-data PATH      Path to load pre-collected observations
  --cbs-only            Train with CBS only (skip Cyberwheel)
```

---

## Examples

### Example 1: Quick Test

```bash
# Test if DAPN works
python run_dapn.py demo
```

### Example 2: Full Training

```bash
# Train encoder
python train_dapn_encoder.py --num-samples 1000 --epochs 50

# Test it
python test_transfer_simple_dapn.py
```

### Example 3: Use in Your Code

```python
# In your training script
from adapters.dapn_env_wrapper import DAPNEnvWrapper

env = DAPNEnvWrapper(
    your_base_env,
    encoder_path="artifacts/transfer_models/dapn_encoder.pt"
)
```

---

## Common Use Cases

### Use Case 1: Transfer Learning from Cyberwheel to CBS

```python
# 1. Train on Cyberwheel (source)
# 2. Use DAPN to adapt to CBS (target)
# 3. Policy trained on Cyberwheel can work on CBS

dapn_env = DAPNEnvWrapper(
    UnifiedSecEnv("cbs", cbs_factory=make_cbs_env),
    encoder_path="trained_dapn_encoder.pt"
)
```

### Use Case 2: Feature Extraction

```python
# Extract features from observations
translator = DAPNObservationTranslator(
    encoder_path="trained_dapn_encoder.pt"
)

features = translator.from_cbs(observation)  # 256-dim features
```

### Use Case 3: Domain Adaptation

```python
# Align observations from different domains
cw_features = translator.from_cw(cyberwheel_obs)  # 256-dim
cbs_features = translator.from_cbs(cbs_obs)        # 256-dim
# Both in same feature space!
```

---

## Troubleshooting

### Issue: "Encoder not found"
**Solution**: Train encoder first:
```bash
python train_dapn_encoder.py --num-samples 1000 --epochs 50
```

### Issue: "Cyberwheel import error"
**Solution**: Use `--cbs-only` flag:
```bash
python train_dapn_encoder.py --cbs-only --num-samples 1000
```

### Issue: "Observation shape mismatch"
**Solution**: Make sure `feature_size` matches:
```python
# When training
--feature-size 256

# When using
feature_size=256
```

### Issue: "CUDA out of memory"
**Solution**: Use CPU or reduce batch size:
```bash
python train_dapn_encoder.py --batch-size 32
```

---

## File Structure

```
artifacts/
  transfer_models/
    dapn_encoder.pt          # Trained encoder (created by training)
  training_data/
    dapn_observations.npz    # Collected observations (optional)
  policies/
    ppo_dapn.zip            # Policy trained with DAPN
```

---

## Summary

1. **Train**: `python train_dapn_encoder.py --num-samples 1000 --epochs 50`
2. **Use**: Wrap your environment with `DAPNEnvWrapper` and provide encoder path
3. **Train Policy**: Use the wrapped environment with your RL algorithm

That's it! The DAPN encoder will automatically convert 8-dim observations to 256-dim features for better domain adaptation.

