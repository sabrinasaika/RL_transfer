# Policy-Based Data Collection Guide

## Quick Start

### Option 1: Use the shell script (easiest)

```bash
./collect_with_policies.sh
```

This uses default policies:
- Cyberwheel: `artifacts/policies/cw_ppo_dapn.zip`
- CBS: `artifacts/policies/cbs_ppo_final.zip`

### Option 2: Use Python directly

```bash
python collect_episodic_data_with_policies.py \
    --cw-policy artifacts/policies/cw_ppo_dapn.zip \
    --cbs-policy artifacts/policies/cbs_ppo_final.zip \
    --cw-episodes 50 \
    --cbs-episodes 50 \
    --max-steps 200 \
    --output artifacts/training_data/episodic_obs_policy.npz
```

## Available Policies

Check what policies you have:

```bash
ls -lh artifacts/policies/*.zip
```

Common policies:
- `cw_ppo_dapn.zip` - Cyberwheel policy trained with DAPN
- `cbs_ppo_final.zip` - CBS policy
- `cw_ppo_minimal.zip` - Minimal Cyberwheel policy
- `cbs_ppo_minimal.zip` - Minimal CBS policy

## Why Use Policies Instead of Random?

### Random Collection Issues:
- ❌ Uneven state distribution (many early states, few late states)
- ❌ Poor clustering (only 5/20 classes usable)
- ❌ Inefficient (wastes samples on unreachable states)
- ❌ Not representative of real agent behavior

### Policy Collection Benefits:
- ✅ **Better state distribution**: Policies visit diverse but reachable states
- ✅ **More balanced classes**: Better clustering → more classes usable
- ✅ **Efficient**: Longer episodes, more progress
- ✅ **Realistic**: States that trained agents actually see
- ✅ **Transfer-relevant**: States that matter for policy transfer

## Features

### 1. Supports Multiple Policy Formats
- **Stable-Baselines3 PPO** (`.zip` files)
- **Cyberwheel native policies** (`.pt` files)

### 2. Epsilon-Greedy Exploration
- `--epsilon 0.0`: Pure policy (recommended)
- `--epsilon 0.2`: 80% policy, 20% random (balanced)
- `--epsilon 1.0`: Pure random (fallback)

### 3. Episode-Based Collection
- Collects full episodes (not one-per-reset)
- Faster and more efficient
- Captures state trajectories

## Examples

### Basic Collection
```bash
python collect_episodic_data_with_policies.py \
    --cw-policy artifacts/policies/cw_ppo_dapn.zip \
    --cbs-policy artifacts/policies/cbs_ppo_final.zip \
    --cw-episodes 50 \
    --cbs-episodes 50 \
    --output artifacts/training_data/episodic_obs_policy.npz
```

### With Epsilon-Greedy (30% random exploration)
```bash
python collect_episodic_data_with_policies.py \
    --cw-policy artifacts/policies/cw_ppo_dapn.zip \
    --cbs-policy artifacts/policies/cbs_ppo_final.zip \
    --epsilon 0.3 \
    --cw-episodes 50 \
    --cbs-episodes 50 \
    --output artifacts/training_data/episodic_obs_eps03.npz
```

### Cyberwheel Only (if no CBS policy)
```bash
python collect_episodic_data_with_policies.py \
    --cw-policy artifacts/policies/cw_ppo_dapn.zip \
    --cw-episodes 100 \
    --cbs-episodes 0 \
    --output artifacts/training_data/episodic_obs_cw_only.npz
```

### More Episodes (for more samples)
```bash
python collect_episodic_data_with_policies.py \
    --cw-policy artifacts/policies/cw_ppo_dapn.zip \
    --cbs-policy artifacts/policies/cbs_ppo_final.zip \
    --cw-episodes 100 \
    --cbs-episodes 100 \
    --max-steps 300 \
    --output artifacts/training_data/episodic_obs_large.npz
```

## Expected Results

With policy-based collection, you should see:
- ✅ **More balanced classes**: All 20 classes should have enough samples
- ✅ **Better clustering**: K-means works better with policy states
- ✅ **More samples**: Longer episodes = more samples per reset
- ✅ **No warnings**: "Using 5 classes" warning should disappear

## Training with Collected Data

After collection, train with:

```bash
python train_dapn_encoder_episodic.py \
    --load-data artifacts/training_data/episodic_obs_policy.npz \
    --n-sc 20 \
    --n-dc 5 \
    --k 5 \
    --query 15 \
    --iterations 500 \
    --test-interval 50 \
    --gpu
```

## Troubleshooting

### "Policy not found"
- Check policy path: `ls artifacts/policies/`
- Use correct file extension (`.zip` for PPO, `.pt` for Cyberwheel native)

### "Could not load PPO policy"
- Make sure Stable-Baselines3 is installed: `pip install stable-baselines3`
- Check if policy file is corrupted

### "Using random actions"
- Policy failed to load, falling back to random
- Check error messages above
- Verify policy file exists and is valid

### Still getting "only 5 classes" warning
- Collect more episodes: `--cw-episodes 100 --cbs-episodes 100`
- Use epsilon-greedy: `--epsilon 0.2` (adds exploration)
- Check class distribution with diagnostic script

## Comparison: Random vs Policy

| Aspect | Random | Policy |
|--------|--------|--------|
| State distribution | Uneven | Balanced |
| Classes usable | 5/20 | 20/20 |
| Samples per reset | 1 | 50-200 |
| Efficiency | Low | High |
| Realistic | No | Yes |
| Transfer-relevant | No | Yes |

## Next Steps

1. **Collect data** with policies (this script)
2. **Check distribution** with diagnostic script
3. **Train encoder** with collected data
4. **Monitor** - should see all 20 classes usable!

## Summary

**Policy-based collection is better because:**
- More balanced state distribution
- Better clustering results
- More efficient (longer episodes)
- More representative of real usage
- Better for domain adaptation

**Use this instead of random collection for production training!**
