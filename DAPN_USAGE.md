# Using DAPN for Observation Handling

This guide explains how to use DAPN (Domain Adaptive Prototypical Network) for handling observations in your RL transfer learning project.

## Overview

DAPN provides domain adaptation capabilities that help align observations between different domains (CBS and Cyberwheel). This is particularly useful for transfer learning scenarios where you want to transfer policies between different environments.

## Quick Start

### 1. Basic Usage

The simplest way to use DAPN is to wrap your environment with `DAPNEnvWrapper`:

```python
from adapters.unified_env import UnifiedSecEnv
from adapters.dapn_env_wrapper import DAPNEnvWrapper
from config.env_builders import make_cbs_env

# Create base environment
base_env = UnifiedSecEnv("cbs", cbs_factory=make_cbs_env)

# Wrap with DAPN
dapn_env = DAPNEnvWrapper(
    base_env,
    encoder_path=None,  # Use random initialization
    feature_size=256,
    use_dapn=True
)

# Use the environment normally
obs, info = dapn_env.reset()
action = dapn_env.action_space.sample()
obs, reward, done, truncated, info = dapn_env.step(action)
```

### 2. Using Pre-trained Encoder

If you have a trained DAPN encoder:

```python
dapn_env = DAPNEnvWrapper(
    base_env,
    encoder_path="artifacts/transfer_models/dapn_encoder.pt",
    feature_size=256,
    use_dapn=True
)
```

### 3. Direct Translator Usage

You can also use the DAPN translator directly:

```python
from adapters.dapn_observation_encoder import DAPNObservationTranslator

translator = DAPNObservationTranslator(
    use_dapn=True,
    encoder_path="path/to/encoder.pt",
    feature_size=256
)

# Encode CBS observation
cbs_obs = {...}  # Your CBS observation dict
encoded = translator.from_cbs(cbs_obs)

# Encode Cyberwheel observation
cw_obs = np.array([...])  # Your Cyberwheel observation vector
encoded = translator.from_cw(cw_obs)
```

## Training a DAPN Encoder

To train your own DAPN encoder for domain adaptation:

### Step 1: Collect Observations

```bash
python train_dapn_encoder.py \
    --num-samples 1000 \
    --save-data artifacts/training_data/dapn_observations.npz
```

### Step 2: Train the Encoder

```bash
python train_dapn_encoder.py \
    --load-data artifacts/training_data/dapn_observations.npz \
    --epochs 50 \
    --batch-size 64 \
    --feature-size 256 \
    --save-encoder artifacts/transfer_models/dapn_encoder.pt
```

Or collect and train in one go:

```bash
python train_dapn_encoder.py \
    --num-samples 1000 \
    --epochs 50 \
    --batch-size 64 \
    --feature-size 256 \
    --save-encoder artifacts/transfer_models/dapn_encoder.pt
```

## Training Policies with DAPN

Once you have a DAPN encoder, you can train policies using DAPN-encoded observations:

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

# Train PPO policy
model = PPO("MultiInputPolicy", dapn_env, verbose=1)
model.learn(total_timesteps=100000)
model.save("artifacts/policies/ppo_dapn")
```

## Integration with Existing Code

### Option 1: Replace Observation Translator

You can replace the observation translator in `UnifiedSecEnv`:

```python
from adapters.dapn_observation_encoder import DAPNObservationTranslator

env = UnifiedSecEnv("cbs", cbs_factory=make_cbs_env)
env.obs_t = DAPNObservationTranslator(
    use_dapn=True,
    encoder_path="path/to/encoder.pt",
    feature_size=256
)
```

### Option 2: Use Wrapper

Use the `DAPNEnvWrapper` as shown in the examples above. This is the recommended approach as it handles observation space updates automatically.

## Architecture Details

### DAPN Encoder

The DAPN encoder uses:
- **Feature extraction layers**: MLP-based feature extractor (adapted from ResNet architecture)
- **Bottleneck layer**: Optional bottleneck for dimensionality reduction
- **Domain adaptation**: Adversarial network for domain alignment

### Feature Space

- **Input dimension**: 8 (unified observation representation)
- **Feature size**: 256 (configurable)
- **Output**: Normalized feature vector

## Parameters

### DAPNEnvWrapper

- `encoder_path`: Path to saved encoder checkpoint (None for random init)
- `feature_size`: Size of feature space (default: 256)
- `use_dapn`: Whether to use DAPN encoding (default: True)
- `device`: Device to run encoder on ('cuda' or 'cpu')

### DAPNObservationTranslator

- `use_dapn`: Whether to use DAPN encoder
- `encoder_path`: Path to saved encoder
- `feature_size`: Size of feature space
- `input_dim`: Input observation dimension (default: 8)
- `device`: Device to run on
- `use_adversarial`: Whether to use adversarial domain adaptation (for training)

## Examples

See `example_use_dapn.py` for complete examples of:
1. Basic DAPN usage
2. Using pre-trained encoders
3. Direct translator usage
4. Training policies with DAPN
5. Comparing observations with/without DAPN

Run the examples:

```bash
python example_use_dapn.py
```

## Troubleshooting

### Import Errors

If you get import errors for DAPN modules, make sure the DAPN-master directory is in your project root and contains the required modules.

### Device Errors

If you encounter CUDA errors, try setting `device='cpu'`:

```python
dapn_env = DAPNEnvWrapper(
    base_env,
    device='cpu'
)
```

### Observation Shape Mismatches

If you get observation shape errors, make sure:
1. The `feature_size` matches between training and usage
2. The observation space is properly updated (handled automatically by `DAPNEnvWrapper`)

## Advanced Usage

### Custom Feature Size

You can use different feature sizes:

```python
dapn_env = DAPNEnvWrapper(
    base_env,
    feature_size=512,  # Larger feature space
    encoder_path="path/to/encoder_512.pt"
)
```

### Training with Domain Adaptation

The training script uses adversarial domain adaptation to align features between CBS and Cyberwheel. You can customize the training by modifying `train_dapn_encoder.py`.

## Files

- `adapters/dapn_observation_encoder.py`: Core DAPN encoder and translator
- `adapters/dapn_env_wrapper.py`: Gym wrapper for DAPN
- `train_dapn_encoder.py`: Training script for DAPN encoder
- `example_use_dapn.py`: Usage examples

## References

DAPN (Domain Adaptive Prototypical Network) is based on domain adaptation techniques for few-shot learning. The implementation adapts DAPN's principles for RL observation handling.

