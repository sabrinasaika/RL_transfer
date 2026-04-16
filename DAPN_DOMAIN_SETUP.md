# DAPN Domain Setup

## Domain Configuration

As specified, the DAPN setup uses:

### Domain 1: Source (Cyberwheel)
- **Purpose**: Training domain
- **Environment**: Cyberwheel
- **Usage**: Primary training data for few-shot learning
- **Encoder**: `cw_encoder` (Cyberwheel encoder)

### Domain 2: Target (Normal CyberBattleSim)
- **Purpose**: Adaptation target
- **Environment**: Standard CyberBattleSim (e.g., CyberBattleChain-v0)
- **Usage**: Domain to adapt to from Cyberwheel
- **Encoder**: `cbs_encoder` (CBS encoder)

### Domain 3: Validation (CBS with Cyberwheel Topology)
- **Purpose**: Evaluation/validation
- **Environment**: CyberBattleSim with Cyberwheel-like network topology (CyberBattleCW10-v0)
- **Usage**: Tests generalization on a hybrid domain
- **Encoder**: `cbs_encoder` (same as target, but different topology)

## How It Works

1. **Source → Target Adaptation**: 
   - Trains encoder to align Cyberwheel (source) with normal CBS (target)
   - Uses adversarial domain adaptation

2. **Validation**:
   - Tests on CBS environment that has Cyberwheel's network topology
   - Ensures model generalizes to hybrid scenarios

## Environment Details

### Domain 1: Cyberwheel
```python
cw_env = UnifiedSecEnv("cw", cw_factory=make_cw_env)
```

### Domain 2: Normal CBS
```python
cbs_env = UnifiedSecEnv("cbs", cbs_factory=make_cbs_env)
# Uses default: CyberBattleChain-v0
```

### Domain 3: CBS with CW Topology
```python
os.environ["CBS_ENV"] = "CyberBattleCW10-v0"  # CBS built from Cyberwheel YAML
val_cbs_env = UnifiedSecEnv("cbs", cbs_factory=make_cbs_env)
```

The `CyberBattleCW10-v0` environment is built from Cyberwheel's network YAML file, creating a CBS environment that matches Cyberwheel's network topology.

## Training Flow

```
Cyberwheel (Source)
    ↓
[Domain Adaptation]
    ↓
Normal CBS (Target)
    ↓
[Validation]
    ↓
CBS with CW Topology (Validation)
```

## Usage

```bash
# Full 3-domain training
python train_dapn_encoder.py --num-samples 1000 --epochs 50

# This will:
# 1. Collect from Cyberwheel (source)
# 2. Collect from normal CBS (target)  
# 3. Collect from CBS with CW topology (validation)
```

## Benefits

1. **Source = Cyberwheel**: Train on the more complex Cyberwheel environment
2. **Target = Normal CBS**: Adapt to simpler CBS for transfer
3. **Validation = Hybrid**: Test on a domain that combines both (CBS mechanics + CW topology)

This setup allows learning from Cyberwheel and transferring to CBS, while validating on a hybrid domain that tests generalization.

