# What `dapn_observation_encoder.py` Does

## Overview
This file implements **DAPN (Domain Adaptive Prototypical Network)** for converting observations from two different RL environments (CBS and Cyberwheel) into a shared feature space. This enables transfer learning between domains.

---

## Three Main Classes

### 1. `DAPNObservationEncoder` (lines 27-114)
**Purpose**: Neural network that encodes observations into feature vectors

**What it does**:
- Takes 8-dimensional observation vectors as input
- Expands them to 256-dimensional feature vectors (configurable)
- Uses MLP (Multi-Layer Perceptron) architecture with:
  - Linear layers: 8 → 128 → 256 → 512
  - Batch normalization
  - ReLU activation
  - Dropout for regularization
  - Optional bottleneck layer (512 → 256)

**Why**: Creates richer representations that help with domain adaptation

---

### 2. `DAPNDomainAdapter` (lines 117-158)
**Purpose**: Adversarial network for domain alignment during training

**What it does**:
- Takes encoded features from both domains
- Tries to predict which domain (CBS or Cyberwheel) the features came from
- During training, the encoder tries to "fool" this adapter
- This forces features from both domains to become similar

**Why**: Aligns the feature spaces so a policy trained on one domain works on the other

---

### 3. `DAPNObservationTranslator` (lines 161-448)
**Purpose**: Main interface that coordinates everything

**What it does**:

#### Initialization (lines 167-230):
1. Creates two encoders: one for CBS, one for Cyberwheel
2. Optionally creates domain adapter for training
3. Loads pre-trained weights if provided (line 224)
4. Sets models to evaluation mode

#### Observation Conversion Methods:

**`from_cbs(obs)`** (lines 266-284):
- Converts CBS observation → 8-dim unified → normalized → 256-dim features

**`from_cw(obs_vec)`** (lines 286-304):
- Converts Cyberwheel observation → 8-dim unified → normalized → 256-dim features

#### Helper Methods:

**`_cbs_to_unified(obs)`** (lines 306-360):
- Extracts 8 key metrics from CBS observation dict:
  1. discovered_node_count
  2. compromised_hosts
  3. discovered_hosts
  4. known_vulns
  5. credentials (credential_cache_length)
  6. steps_elapsed
  7. dist_to_goal
  8. alerts

**`_cw_to_unified(obs_vec)`** (lines 362-423):
- **Line 415 is here!** This is part of converting Cyberwheel observations
- Parses Cyberwheel's variable-length observation vector
- Extracts same 8 metrics as CBS:
  1. discovered_hosts (from host discovery flags)
  2. compromised_hosts (from on_host/escalated/impacted flags)
  3. discovered_hosts (same as #1)
  4. known_vulns (proxied by scanned_hosts)
  5. credentials_found (proxied by escalated_count)
  6. steps_elapsed (from quadrant information)
  7. dist_to_goal (from impacted hosts)
  8. alerts (escalations + impacts)

**`_normalize(vec)`** (lines 425-427):
- Normalizes 8-dim vector to [0, 1] range using default scales

**`_encode_to_features(obs, domain)`** (lines 429-447):
- Runs normalized observation through the appropriate encoder (CBS or CW)
- Returns 256-dim feature vector

---

## Data Flow

```
CBS Observation (dict)
    ↓
_cbs_to_unified() → [8 numbers]
    ↓
_normalize() → [8 numbers, 0-1 range]
    ↓
DAPN Encoder → [256 numbers]
    ↓
Ready for RL policy
```

```
Cyberwheel Observation (variable-length array)
    ↓
_cw_to_unified() → [8 numbers]  ← Line 415 is here!
    ↓
_normalize() → [8 numbers, 0-1 range]
    ↓
DAPN Encoder → [256 numbers]
    ↓
Ready for RL policy
```

---

## Key Feature: Domain Adaptation

The magic happens because:
1. Both CBS and Cyberwheel observations are converted to the **same 8-dim unified format**
2. Both are then encoded to the **same 256-dim feature space**
3. During training, the domain adapter forces these features to be similar
4. Result: A policy trained on CBS features can work on Cyberwheel features (and vice versa)

---

## Line 415 Context

Line 415 is in `_cw_to_unified()` method, specifically returning the unified 8-dim array:
```python
return np.array([
    discovered_hosts,      # Line 415
    compromised_hosts,    # Line 416
    discovered_hosts,     # Line 417
    known_vulns,          # Line 418
    credentials_found,    # Line 419
    steps_elapsed,        # Line 420
    dist_to_goal,         # Line 421
    alerts               # Line 422
], dtype=np.float32)
```

This creates a standardized representation that matches what CBS produces, enabling domain adaptation.

---

## Usage Example

```python
from adapters.dapn_observation_encoder import DAPNObservationTranslator

# Create translator
translator = DAPNObservationTranslator(
    use_dapn=True,
    encoder_path="path/to/trained/encoder.pt"
)

# Convert CBS observation
cbs_obs = {"discovered_node_count": 5, ...}
features = translator.from_cbs(cbs_obs)  # Returns 256-dim array

# Convert Cyberwheel observation
cw_obs = np.array([1, 1, 1, ...])
features = translator.from_cw(cw_obs)  # Returns 256-dim array

# Both now in same feature space!
```

---

## Summary

This file enables **transfer learning** by:
1. ✅ Converting different observation formats to a unified representation
2. ✅ Encoding to a shared feature space
3. ✅ Using domain adaptation to align features between domains
4. ✅ Making it possible to transfer policies between CBS and Cyberwheel


