# Cyberwheel Training Analysis & Recommendations

## Executive Summary

Training results show **poor learning performance** with negative returns and increasing losses. The agent is failing to learn effectively due to **suboptimal hyperparameters** and **sparse reward signals**.

## Key Findings

### 1. Performance Metrics

| Training Duration | Episodic Return Change | Policy Loss Change | Value Loss Change |
|------------------|----------------------|-------------------|------------------|
| 500 steps        | +12 (4.9% ↑)         | +1.78 (7% ↑)      | +150 (24% ↑)     |
| 1000 steps       | -40 (-16.2% ↓)        | +5.64 (23% ↑)     | +481 (76% ↑)     |
| 2000 steps       | -9 (-3.6% ↓)          | +1.05 (4% ↑)      | +93 (15% ↑)      |
| 5000 steps       | -3 (-1.2% ↓)          | +2.02 (8% ↑)      | +175 (28% ↑)     |

**All runs start at ~-247 return** (very negative baseline).

### 2. Root Causes

#### A. Reward Function Issues (`reward_decoy_hits`)
- **Sparse rewards**: Only +1.0 (or +10 in first quadrant) for successful actions
- **High penalties**: -1.0 for every failed action
- **Result**: Most actions fail, creating negative baseline (~-247 suggests ~247 failed actions per episode)
- **Impact**: Value function struggles to learn from sparse, mostly negative rewards

#### B. Hyperparameter Problems

| Parameter | Current Value | Issue | Recommended |
|-----------|--------------|-------|-------------|
| `num_envs` | 1 | Too small batch | 4-8 |
| `batch_size` | 100 | High variance | 400-800 |
| `update_epochs` | 1 | Poor sample efficiency | 4-8 |
| `ent_coef` | 0.0 | No exploration | 0.01-0.05 |
| `norm_adv` | False | Unstable gradients | True |
| `learning_rate` | 0.0001 | OK but needs larger batches | 0.00008-0.0001 |

### 3. Training Instability Indicators

- **Policy loss increasing**: Policy is getting worse over time
- **Value loss exploding**: Value function can't predict sparse rewards (76% increase in 1000 steps)
- **Negative returns throughout**: Agent fails most actions
- **No learning plateau**: Performance degrades with more training

## Recommendations

### Immediate Fixes (Implemented in `train_500_steps_improved.yaml`)

1. **Increase batch size**: `num_envs: 4` → batch size = 400
2. **Add exploration**: `ent_coef: 0.01`
3. **More update epochs**: `update_epochs: 4`
4. **Normalize advantages**: `norm_adv: true`
5. **Slightly reduce LR**: `learning_rate: 0.00008` for stability
6. **Minibatch training**: `num_minibatches: 4`

### Long-term Improvements

#### 1. Reward Shaping
- Consider adding intermediate rewards for progress (discovery, lateral movement)
- Reduce failure penalty or make it action-dependent
- Add reward normalization/standardization

#### 2. Architecture
- Increase neural network capacity if action space is large (1200 actions)
- Consider attention mechanisms for large action spaces

#### 3. Training Strategy
- Curriculum learning: Start with easier networks
- Pre-training: Initialize with demonstrations or pre-trained features
- Reward engineering: Make rewards denser and more informative

#### 4. Hyperparameter Tuning
- Systematic hyperparameter search (grid search or Bayesian optimization)
- Focus on: learning rate, entropy coefficient, batch size, update epochs

## Improved Configuration

See `train_500_steps_improved.yaml` for optimized hyperparameters:
- 4× larger batch size
- 4× more update epochs
- Added entropy bonus
- Normalized advantages
- Slightly reduced learning rate

## Next Steps

1. **Retrain with improved configs** for 500, 1000, 2000, 5000 steps
2. **Compare results** - check if losses stabilize and returns improve
3. **If still poor**: Consider reward function modification
4. **If improved**: Scale up to longer training runs

## Files Created

- `train_500_steps_improved.yaml` - Optimized configuration
- `plot_cyberwheel_training_comparison.py` - Visualization script
- Training logs in `cyberwheel/data/runs/CWRun_*`

