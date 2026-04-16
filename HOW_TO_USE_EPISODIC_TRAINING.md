# How to Use Episodic Training - Integration Guide

## Quick Integration

Episodic training is now **integrated into your existing `train_dapn_encoder.py`** script. Just add the `--episodic` flag!

## Usage

### Option 1: Use Existing Script with Episodic Flag (Recommended)

```bash
# Regular training (domain adversarial only)
python train_dapn_encoder.py --num-samples 1000 --epochs 50

# Episodic training (FSL + domain adversarial)
python train_dapn_encoder.py --num-samples 1000 --episodic --iterations 10000
```

### Option 2: Use Standalone Episodic Script

```bash
python train_dapn_encoder_episodic.py --num-samples 1000 --iterations 10000
```

## Command Line Options

### Regular Training (Default)
```bash
python train_dapn_encoder.py \
    --num-samples 1000 \
    --epochs 50 \
    --batch-size 64 \
    --lr 0.001 \
    --save-encoder artifacts/transfer_models/dapn_encoder.pt
```

### Episodic Training (Add `--episodic`)
```bash
python train_dapn_encoder.py \
    --num-samples 1000 \
    --episodic \
    --iterations 10000 \
    --n-sc 20 \
    --n-dc 5 \
    --k 5 \
    --query 15 \
    --lr 0.001 \
    --save-encoder artifacts/transfer_models/dapn_encoder_episodic.pt
```

## Parameters

### Common Parameters (Both Modes)
- `--num-samples`: Number of samples to collect per domain
- `--feature-size`: Feature space size (default: 256)
- `--lr`: Learning rate (default: 0.001)
- `--load-data`: Path to load pre-collected observations
- `--save-data`: Path to save collected observations
- `--save-encoder`: Path to save trained encoder

### Regular Training Only
- `--epochs`: Number of training epochs (default: 50)
- `--batch-size`: Batch size (default: 64)

### Episodic Training Only (requires `--episodic`)
- `--iterations`: Number of training iterations (default: 10000)
- `--n-sc`: Number of classes for source domain (Nsc, default: 20)
- `--n-dc`: Number of classes for target domain (Ndc, default: 5)
- `--k`: Number of shots per class (default: 5)
- `--query`: Number of query samples per class (default: 15)

## Examples

### Example 1: Quick Test
```bash
# Collect and train with episodic structure
python train_dapn_encoder.py \
    --num-samples 1000 \
    --episodic \
    --iterations 1000 \
    --save-encoder artifacts/transfer_models/test_episodic.pt
```

### Example 2: Full Training
```bash
# Step 1: Collect observations
python train_dapn_encoder.py \
    --num-samples 1000 \
    --save-data artifacts/observations.npz

# Step 2: Train with episodic structure
python train_dapn_encoder.py \
    --load-data artifacts/observations.npz \
    --episodic \
    --iterations 10000 \
    --n-sc 20 \
    --n-dc 5 \
    --k 5 \
    --query 15 \
    --save-encoder artifacts/transfer_models/dapn_encoder_episodic.pt
```

### Example 3: Compare Regular vs Episodic
```bash
# Regular training
python train_dapn_encoder.py \
    --load-data artifacts/observations.npz \
    --epochs 50 \
    --save-encoder artifacts/transfer_models/regular.pt

# Episodic training
python train_dapn_encoder.py \
    --load-data artifacts/observations.npz \
    --episodic \
    --iterations 10000 \
    --save-encoder artifacts/transfer_models/episodic.pt
```

## What Changes When Using `--episodic`?

| Aspect | Regular Training | Episodic Training |
|--------|------------------|-------------------|
| **Data sampling** | Random batches | Episodic (N-way K-shot) |
| **Loss function** | Domain adversarial only | FSL loss + Domain adversarial |
| **Class structure** | None | K-means clustering |
| **Support/Query** | None | Explicit split |
| **Training loop** | Epochs over dataset | Iterations with episodes |

## Integration Points

### In Your Code

The episodic training is integrated into `train_dapn_encoder.py`. When you use `--episodic`:

1. It imports `train_dapn_encoder_episodic` function
2. Uses episodic samplers instead of regular DataLoader
3. Computes prototypical network loss in addition to domain adversarial loss
4. Saves encoder the same way (compatible with existing code)

### Using the Trained Encoder

The trained encoder works exactly the same way regardless of training mode:

```python
from adapters.dapn_observation_encoder import DAPNObservationTranslator

# Works with both regular and episodic trained encoders
translator = DAPNObservationTranslator(
    encoder_path="artifacts/transfer_models/dapn_encoder_episodic.pt",
    use_dapn=True
)
```

## Files Created

1. **`adapters/episodic_training.py`**: Core episodic training utilities
   - `CategoriesSampler`: Episodic sampler
   - `cluster_observations()`: K-means clustering
   - `create_episodic_dataloaders()`: Creates samplers

2. **`train_dapn_encoder_episodic.py`**: Standalone episodic training script
   - Can be used independently
   - Or imported by `train_dapn_encoder.py` when `--episodic` is used

3. **`train_dapn_encoder.py`**: Updated with `--episodic` flag
   - Backward compatible (defaults to regular training)
   - Adds episodic option when flag is set

## Troubleshooting

**Issue**: "ModuleNotFoundError: No module named 'sklearn'"
- **Solution**: `pip install scikit-learn`

**Issue**: "Not enough samples for clustering"
- **Solution**: Increase `--num-samples` (need at least Nsc samples for Ds, Ndc for Dd)

**Issue**: "Cannot form episodes"
- **Solution**: Ensure you have enough samples:
  - Ds: Need at least `n_sc × (k + query)` = 20 × 20 = 400 samples
  - Dd: Need at least `n_dc × (k + query)` = 5 × 20 = 100 samples

## Summary

✅ **Episodic training is integrated!** Just add `--episodic` flag to your existing command.

```bash
# Before (regular)
python train_dapn_encoder.py --num-samples 1000 --epochs 50

# After (episodic)
python train_dapn_encoder.py --num-samples 1000 --episodic --iterations 10000
```

That's it! The rest of your code (using the encoder, training policies, etc.) remains unchanged.
