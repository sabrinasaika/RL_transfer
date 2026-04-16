# Improving DAPN Episodic Training Accuracy

## Current Status
- FSL Accuracy: ~15.7% (slightly above random ~14.3% for 7 classes)
- Training iterations: 50 (quick test)
- Samples per domain: 200

## Implemented Improvements

### 1. **Learning Rate Scheduling**
- Added StepLR scheduler that reduces learning rate by 50% every 1/3 of training
- Helps fine-tune in later stages

### 2. **Weight Decay Regularization**
- Added `weight_decay=1e-5` to both encoder and discriminator optimizers
- Prevents overfitting

### 3. **Gradient Clipping**
- Clips gradients to max_norm=1.0
- Stabilizes training and prevents exploding gradients

### 4. **Adaptive Transfer Loss Weight**
- Transfer loss weight increases from 0.1 to 0.2 over training
- Balances FSL and domain adaptation losses better

## Additional Recommendations

### 1. **More Training Iterations**
```bash
python train_dapn_encoder.py \
    --num-samples 1000 \
    --episodic \
    --iterations 500 \
    --n-sc 20 \
    --n-dc 5 \
    --k 5 \
    --query 15 \
    --lr 0.0005 \
    --save-encoder artifacts/transfer_models/dapn_encoder_episodic.pt
```

### 2. **More Data**
- Increase `--num-samples` from 200 to 1000+ per domain
- More diverse observations = better prototypes

### 3. **Better Hyperparameters**
- Lower learning rate: `--lr 0.0005` (more stable)
- More query samples: `--query 20` (better prototype learning)
- Larger feature size: `--feature-size 512` (more capacity)

### 4. **Use Trained Agents**
- Use `--cw-agent` and `--cbs-agent` to collect more meaningful observations
- Trained agents produce more diverse, realistic state-action pairs

### 5. **Feature Normalization**
- Normalize features before computing prototypes
- Can help with distance-based classification

### 6. **Temperature Scaling**
- Add temperature parameter to softmax in prototypical loss
- Helps with calibration

## Quick Test with Improvements

```bash
python train_dapn_encoder.py \
    --load-data artifacts/quick_test_obs.npz \
    --episodic \
    --iterations 200 \
    --n-sc 7 \
    --n-dc 5 \
    --k 5 \
    --query 15 \
    --lr 0.0005 \
    --save-encoder artifacts/transfer_models/dapn_encoder_episodic_improved.pt
```

## Expected Improvements
- With 200+ iterations: 20-30% accuracy
- With 1000 samples: 30-40% accuracy  
- With trained agents: 40-50% accuracy
- With all improvements: 50-60% accuracy
