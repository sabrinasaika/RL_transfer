# DAPN 3-Domain Setup Explanation

## The Issue You Identified

You're absolutely right! **DAPN requires 3 domains**, not 2. The original DAPN implementation uses:

1. **Source Domain** - Training data (e.g., mini-imagenet/train)
2. **Target Domain** - Adaptation target (e.g., mini-imagenet/val_new_domain)  
3. **Validation/Test Domain** - Evaluation (e.g., mini-imagenet/test_new_domain)

## How We're Handling It Now

I've updated the code to properly support 3 domains:

### Domain 1: Source (CBS - Training)
- Primary training domain
- Used for few-shot learning support set
- Standard CBS environment

### Domain 2: Target (Cyberwheel - Adaptation)
- Domain to adapt to
- Used for domain adversarial training
- Cyberwheel environment

### Domain 3: Validation (CBS Variant - Evaluation)
- Different CBS configuration (e.g., different network size)
- Used for validation/evaluation
- Tests generalization

## Implementation Details

### Collection Function
```python
collect_observations(
    num_samples=1000,
    use_3_domains=True  # Enable 3-domain setup
)
```

**What it does:**
1. Collects from Source (CBS standard)
2. Collects from Target (Cyberwheel)
3. Collects from Validation (CBS with different config, e.g., size=8 vs size=6)

### Training Function
```python
train_dapn_encoder(
    source_obs_list,    # Domain 1
    target_obs_list,    # Domain 2
    val_obs_list,        # Domain 3
    use_3_domains=True
)
```

**Training process:**
- Source + Target: Adversarial domain adaptation
- Validation: Used for evaluation during training
- All 3: Reconstruction loss

## Fallback Options

If 3 domains aren't available:
- **2 domains**: Source + Target (standard domain adaptation)
- **1 domain**: Source only (no domain adaptation, just feature learning)

## Usage

**Full 3-domain training:**
```bash
python train_dapn_encoder.py --num-samples 1000 --epochs 50
# Automatically uses 3 domains if available
```

**2-domain training (if CW unavailable):**
```bash
python train_dapn_encoder.py --num-samples 1000 --epochs 50 --cbs-only
# Uses source + validation (split from source)
```

## Why This Matters

The 3-domain setup allows DAPN to:
1. **Train** on source domain
2. **Adapt** between source and target domains
3. **Validate** on a held-out domain to ensure generalization

This is more robust than 2-domain training and matches the original DAPN methodology.

