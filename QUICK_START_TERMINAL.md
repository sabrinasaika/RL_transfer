# Quick Start - Terminal Commands

## Prerequisites
Make sure you're in the project directory:
```bash
cd /home/ssaika/rl-transfer-sec-clean
```

## Step 1: Test Both Scenarios (Verify They Work)

```bash
python test_scenarios.py
```

Expected output: Both scenarios should show "✓ PASSED"

---

## Step 2: Run Cyberwheel Credential Preference Scenario

### Quick Training (50,000 steps)
```bash
export CW_ENV_YAML=credential_preference_scenario.yaml
python train/train_cw_ppo_minimal.py
```

### Custom Training Steps
```bash
export CW_ENV_YAML=credential_preference_scenario.yaml
export CW_TRAIN_STEPS=100000
python train/train_cw_ppo_minimal.py
```

### Check Results
The trained model will be saved to: `artifacts/policies/cw_ppo_minimal.zip`

---

## Step 3: Run CyberBattleSim Flat Network Scenario

### Quick Training (50,000 steps)
```bash
export CBS_ENV=CyberBattleFlat-v0
export CBS_FLAT_NODES=20
export CBS_CRED_REUSE_PROB=0.6
export CBS_EXPLOIT_PROB=0.3
python train/train_cbs_ppo_minimal.py
```

### Custom Parameters
```bash
export CBS_ENV=CyberBattleFlat-v0
export CBS_FLAT_NODES=30              # More nodes
export CBS_CRED_REUSE_PROB=0.7        # Higher credential success
export CBS_EXPLOIT_PROB=0.2           # Lower exploit success
python train/train_cbs_ppo_minimal.py
```

### Check Results
The trained model will be saved to: `artifacts/policies/cbs_ppo_minimal.zip`

---

## Step 4: Using DAPN for Transfer Learning

### 4a. Train DAPN Encoder (First Time Only)
```bash
python train_dapn_encoder.py --num-samples 1000 --epochs 50
```

This creates: `artifacts/transfer_models/dapn_encoder.pt`

### 4b. Train on Source Domain (Cyberwheel)
```bash
export CW_ENV_YAML=credential_preference_scenario.yaml
python train/train_cw_ppo_minimal.py
```

### 4c. Transfer to Target Domain (CyberBattleSim)
Create a transfer script or modify the training script to use DAPNEnvWrapper.

---

## Quick Test Commands

### Test Cyberwheel Only
```bash
export CW_ENV_YAML=credential_preference_scenario.yaml
python -c "
from adapters.unified_env import UnifiedSecEnv
from config.env_builders import make_cw_env
env = UnifiedSecEnv('cw', cw_factory=make_cw_env)
obs, info = env.reset()
print('✓ Cyberwheel environment works!')
print(f'Observation: {obs.shape}')
print(f'Action space: {env.action_space}')
"
```

### Test CyberBattleSim Only
```bash
export CBS_ENV=CyberBattleFlat-v0
python -c "
from adapters.unified_env import UnifiedSecEnv
from config.env_builders import make_cbs_env
env = UnifiedSecEnv('cbs', cbs_factory=make_cbs_env)
obs, info = env.reset()
print('✓ CyberBattleSim environment works!')
print(f'Observation: {type(obs)}')
print(f'Action space: {env.action_space}')
"
```

---

## Common Commands Reference

### Set Environment Variables (One Line)
```bash
# Cyberwheel
export CW_ENV_YAML=credential_preference_scenario.yaml

# CyberBattleSim
export CBS_ENV=CyberBattleFlat-v0 && export CBS_FLAT_NODES=20 && export CBS_CRED_REUSE_PROB=0.6 && export CBS_EXPLOIT_PROB=0.3
```

### Check Current Environment Variables
```bash
env | grep -E "CW_|CBS_"
```

### Clear Environment Variables
```bash
unset CW_ENV_YAML
unset CBS_ENV
unset CBS_FLAT_NODES
unset CBS_CRED_REUSE_PROB
unset CBS_EXPLOIT_PROB
```

---

## Troubleshooting

### If you get "No module named 'cyberwheel'"
```bash
export PYTHONPATH=/home/ssaika/rl-transfer-sec-clean/cyberwheel:$PYTHONPATH
```

### If you get "No module named 'CyberBattleSim'"
```bash
export PYTHONPATH=/home/ssaika/rl-transfer-sec-clean:$PYTHONPATH
```

### Check if files exist
```bash
# Check network config
ls -la cyberwheel/cyberwheel/data/configs/network/credential-preference-100host-network.yaml

# Check environment config
ls -la cyberwheel/cyberwheel/data/configs/environment/credential_preference_scenario.yaml

# Check red agent config
ls -la cyberwheel/cyberwheel/data/configs/red_agent/rl_red_agent_credential_preference.yaml
```

---

## Example: Complete Training Session

```bash
# 1. Test everything works
python test_scenarios.py

# 2. Train on Cyberwheel (in background)
export CW_ENV_YAML=credential_preference_scenario.yaml
nohup python train/train_cw_ppo_minimal.py > cw_training.log 2>&1 &

# 3. Train on CyberBattleSim (in background)
export CBS_ENV=CyberBattleFlat-v0
nohup python train/train_cbs_ppo_minimal.py > cbs_training.log 2>&1 &

# 4. Check training progress
tail -f cw_training.log
tail -f cbs_training.log
```

---

## Next Steps After Training

### Load and Test Trained Model
```python
from stable_baselines3 import PPO
from adapters.unified_env import UnifiedSecEnv
from config.env_builders import make_cw_env
import os

# Load model
model = PPO.load("artifacts/policies/cw_ppo_minimal")

# Create environment
os.environ["CW_ENV_YAML"] = "credential_preference_scenario.yaml"
env = UnifiedSecEnv("cw", cw_factory=make_cw_env)

# Test
obs, info = env.reset()
for _ in range(10):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    print(f"Reward: {reward}")
    if done or truncated:
        obs, info = env.reset()
```

