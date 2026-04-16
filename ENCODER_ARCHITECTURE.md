# DAPN Full Observation Encoder Architecture

## Overview

The encoder uses a **single shared encoder** for both CyberBattleSim (CBS) and Cyberwheel (CW) domains. Observations from each domain are first preprocessed into a fixed-size unified format, then passed through the same encoder to produce domain-invariant features.

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    ENCODER PIPELINE                              │
└─────────────────────────────────────────────────────────────────┘

CBS Domain (Dict)                    Cyberwheel Domain (Array)
     │                                      │
     │                                      │
     ▼                                      ▼
┌─────────────────┐              ┌──────────────────┐
│  Preprocessor   │              │   Preprocessor   │
│  (CBS → 512D)   │              │   (CW → 512D)    │
└─────────────────┘              └──────────────────┘
     │                                      │
     │                                      │
     └──────────────┬──────────────────────┘
                    │
                    ▼
         ┌──────────────────────┐
         │  Normalization       │
         │  [0, 1] range        │
         └──────────────────────┘
                    │
                    ▼
         ┌──────────────────────┐
         │  Single Shared       │
         │  Encoder (512→256)    │
         └──────────────────────┘
                    │
                    ▼
         ┌──────────────────────┐
         │  Feature Vector      │
         │  [256 dimensions]    │
         └──────────────────────┘
```

---

## Step-by-Step Data Flow

### Step 1: Observation Collection

#### **CBS Observation (Source: CyberBattleSim)**
```python
obs_cbs = {
    "newly_discovered_nodes_count": 2,
    "lateral_move": 0,
    "customer_data_found": 0,
    "probe_result": 1,
    "escalation": 0,
    "credential_cache_length": 3,
    "discovered_nodes_properties": np.array([[1, 0, 1], [0, 1, 0]]),  # (N, 3)
    "nodes_privilegelevel": np.array([0, 1, 2]),  # (N,)
    "credential_cache_matrix": (np.array([0, 80]), np.array([1, 443])),  # Tuple
    "discovered_node_count": 5,
    "_explored_network": NetworkXGraph(...)
}
```

#### **Cyberwheel Observation (Source: Cyberwheel)**
```python
obs_cw = np.array([0.5, -0.3, 1.0, 0.2, ...])  # Variable length (typically 701D)
# Contains: host attributes, network topology, etc.
```

---

### Step 2: Preprocessing (UnifiedFullObsPreprocessor)

#### **CBS Preprocessing (`preprocess_cbs`)**

The CBS dict is flattened into a fixed-size vector, **including ALL available fields**:

```
Input: CBS Dict
  ↓
1. Extract Scalars (6 core values)
   - newly_discovered_nodes_count
   - lateral_move
   - customer_data_found
   - probe_result
   - escalation
   - credential_cache_length
  ↓
2. Extract Node Properties (max_nodes × 3)
   - Pad/truncate to max_nodes=50
   - Flatten: (50, 3) → 150D
  ↓
3. Extract Privilege Levels (max_nodes)
   - Pad/truncate to max_nodes=50
   - Result: 50D
  ↓
4. Extract Credential Cache (max_credentials × 2)
   - Pad/truncate to max_credentials=100
   - Flatten: (100, 2) → 200D
  ↓
5. Extract Graph Statistics (2 values)
   - num_nodes, num_edges
  ↓
6. Additional Fields (3 values)
   - discovered_node_count, probe_result, escalation
  ↓
7. Extract ALL Remaining Fields (automatic)
   - Any other numeric fields in the dict
   - Nested arrays/lists are flattened
   - All values converted to float32
  ↓
Concatenate all parts → Variable length vector
  ↓
Pad/Truncate to unified_dim=512
  ↓
Output: Fixed-size vector [512D]
```

**Key Features:**
- **Comprehensive:** Includes ALL available fields from CBS observation
- **Automatic:** Automatically extracts and flattens any additional numeric fields
- **Robust:** Handles missing fields gracefully (defaults to 0)

**Example Calculation:**
```
6 (scalars) + 150 (node_props) + 50 (privileges) + 200 (credentials) + 2 (graph) + 3 (additional) + X (other fields)
= Variable → padded/truncated to 512D
```

#### **Cyberwheel Preprocessing (`preprocess_cw`)**

The Cyberwheel array is simply padded/truncated:

```
Input: Variable-length array (e.g., 701D)
  ↓
Pad with zeros if < 512
OR
Truncate if > 512
  ↓
Output: Fixed-size vector [512D]
```

---

### Step 3: Normalization

Both preprocessed vectors are normalized to [0, 1] range:

```python
# For CBS (typically [0, max] range)
normalized = unified_vec / max_vals  # Clip to [0, 1]

# For Cyberwheel (typically [-1, 1] range)
normalized = (unified_vec + 1.0) / 2.0  # Convert [-1, 1] → [0, 1]
```

**Output:** Normalized vector [512D] in [0, 1] range

---

### Step 4: Encoding (DAPNObservationEncoder)

The normalized 512D vector passes through the shared encoder:

```
Input: [512D] normalized vector
  ↓
┌─────────────────────────────────────────┐
│  Feature Extraction Layers              │
│                                         │
│  Linear(512 → 128)                      │
│  BatchNorm1d(128)                       │
│  ReLU                                    │
│  Dropout(0.2)                           │
│  ────────────────────────              │
│  Linear(128 → 256)                      │
│  BatchNorm1d(256)                       │
│  ReLU                                    │
│  Dropout(0.2)                           │
│  ────────────────────────              │
│  Linear(256 → 512)                      │
│  BatchNorm1d(512)                       │
│  ReLU                                    │
│                                         │
└─────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────┐
│  Bottleneck Layer (Optional)             │
│                                         │
│  Linear(512 → 256)                      │
│                                         │
└─────────────────────────────────────────┘
  ↓
Output: Feature vector [256D]
```

**Architecture Details:**
- **Input dimension:** 512 (unified_dim)
- **Feature layers:** 3-layer MLP (512→128→256→512)
- **Bottleneck:** Optional (512→256)
- **Output dimension:** 256 (feature_size)
- **Activation:** ReLU
- **Regularization:** BatchNorm + Dropout(0.2)

---

### Step 5: Feature Output

The encoder outputs a **256-dimensional feature vector** that is:
- **Domain-invariant:** Features from CBS and CW are aligned in the same space
- **Semantic:** Contains rich information about the observation
- **Normalized:** Features are typically in a reasonable range

---

## Training Flow (Adversarial Domain Adaptation)

During training, the encoder learns to produce domain-invariant features:

```
┌─────────────────────────────────────────────────────────────┐
│                    TRAINING LOOP                             │
└─────────────────────────────────────────────────────────────┘

For each batch:
  1. Collect observations from both domains
     - Source (Cyberwheel): N samples
     - Target (CBS): M samples
  
  2. Preprocess & Encode (BATCH PROCESSING - Critical!)
     - Preprocess ALL CBS obs → 512D vectors (batch)
     - Preprocess ALL CW obs → 512D vectors (batch)
     - Normalize both batches to [0, 1]
     - Encode BOTH batches through shared encoder → 256D features
     - **Important:** Batch processing is essential for BatchNorm to work correctly
  
  3. Update Discriminator (Domain Adapter)
     - Input: Features (detached, no gradients)
     - Predict: Domain (0=CW, 1=CBS)
     - Loss: BCE(domain_pred, domain_labels)
     - Goal: Learn to distinguish domains
  
  4. Update Encoder
     - Input: Features (with gradients)
     - Predict: Domain (same discriminator)
     - Loss: 1.0 - BCE(domain_pred, domain_labels)
     - Goal: Confuse discriminator (make features domain-invariant)
  
  5. Repeat until convergence
```

### Domain Adapter Architecture

```
Input: Feature vector [256D]
  ↓
Linear(256 → 1024)
  - Weight init: Xavier uniform (gain=0.1) to prevent saturation
ReLU
Dropout(0.5)
  ↓
Linear(1024 → 1024)
  - Weight init: Xavier uniform (gain=0.1)
ReLU
Dropout(0.5)
  ↓
Linear(1024 → 1)
  - Weight init: Xavier uniform (gain=0.1)
Sigmoid
  ↓
Clamp to [1e-7, 1-1e-7] (prevent numerical issues)
  ↓
Output: Domain probability [0, 1]
  (0 = Cyberwheel, 1 = CBS)
```

**Key Improvements:**
- **Small initialization (gain=0.1):** Prevents discriminator saturation
- **Prediction clamping:** Avoids numerical instability
- **Saturation detection:** Auto-resets if discriminator predicts all same value
- **Gradient clipping:** Stabilizes training (max_norm=1.0)

---

## Key Design Decisions

### 1. **Single Shared Encoder**
- Follows DAPN master concept
- Both domains use the same encoder after preprocessing
- Enables domain-invariant feature learning

### 2. **Fixed-Size Preprocessing**
- Converts variable-length observations to fixed 512D
- Preserves all information (padding/truncation)
- Enables batch processing

### 3. **Unified Representation**
- Both domains → 512D → 256D features
- Same preprocessing pipeline
- Same encoder architecture

### 4. **Adversarial Training**
- Discriminator tries to identify domain
- Encoder tries to confuse discriminator
- Results in domain-invariant features

---

## Dimension Summary

| Stage | CBS Input | CW Input | Unified | Normalized | Features |
|-------|-----------|----------|---------|------------|----------|
| **Raw** | Dict (variable) | Array (701D) | - | - | - |
| **Preprocessed** | - | - | 512D | - | - |
| **Normalized** | - | - | - | 512D | - |
| **Encoded** | - | - | - | - | 256D |

---

## Code Flow Example

### Single Observation (Inference)

```python
# 1. Initialize translator
translator = DAPNUnifiedFullObsTranslator(
    use_dapn=True,
    feature_size=256,
    unified_dim=512
)

# 2. Process CBS observation
obs_cbs = {...}  # Dict from CBS
features_cbs = translator.from_cbs(obs_cbs)
# Internally:
#   - preprocessor.preprocess_cbs(obs_cbs) → 512D vector (includes ALL fields)
#   - Normalize to [0, 1]
#   - shared_encoder(512D) → 256D features

# 3. Process Cyberwheel observation
obs_cw = np.array([...])  # Array from CW
features_cw = translator.from_cw(obs_cw)
# Internally:
#   - preprocessor.preprocess_cw(obs_cw) → 512D vector
#   - Normalize to [0, 1]
#   - shared_encoder(512D) → 256D features

# Both features_cbs and features_cw are now in the same 256D space!
```

### Batch Processing (Training - Critical!)

```python
# During training, observations are processed in BATCHES (not one-by-one)
# This is essential for BatchNorm to work correctly!

# 1. Preprocess all observations first
source_unified_vecs = []
for obs in source_obs_batch:  # Batch of Cyberwheel observations
    unified_vec = translator.preprocessor.preprocess_cw(obs)
    normalized = normalize_observation(unified_vec, max_vals)
    source_unified_vecs.append(normalized)

target_unified_vecs = []
for obs in target_obs_batch:  # Batch of CBS observations
    unified_vec = translator.preprocessor.preprocess_cbs(obs)
    normalized = normalize_observation(unified_vec, max_vals)
    target_unified_vecs.append(normalized)

# 2. Convert to batch tensors and encode in BATCH (like 8D version)
source_batch_tensor = torch.from_numpy(np.array(source_unified_vecs)).float().to(device)
source_features = translator.shared_encoder(source_batch_tensor)  # Batch processing!

target_batch_tensor = torch.from_numpy(np.array(target_unified_vecs)).float().to(device)
target_features = translator.shared_encoder(target_batch_tensor)  # Batch processing!

# Now source_features and target_features are batched tensors ready for adversarial training
```

**Why Batch Processing Matters:**
- **BatchNorm requires batches:** BatchNorm layers need multiple samples to compute statistics
- **Much faster:** GPU processes batches in parallel
- **Better gradients:** Batch statistics provide more stable gradients
- **Matches 8D version:** Same approach that worked well for 8D

---

## Benefits of This Architecture

1. **Domain Adaptation:** Single encoder learns domain-invariant features
2. **Information Preservation:** Full observations preserved (not reduced to 8D)
   - **All CBS fields included:** Automatically extracts all available fields
3. **Flexibility:** Works with variable-length inputs from both domains
4. **Efficiency:** Single encoder reduces model size and training time
5. **Transfer Learning:** Features can be used for policy transfer between domains
6. **Batch Processing:** Efficient GPU utilization and proper BatchNorm behavior
7. **Stable Training:** Discriminator initialization and clamping prevent saturation

---

## Training Improvements

### Key Fixes Applied

1. **Batch Processing (Critical Fix)**
   - **Before:** Processed observations one-by-one → BatchNorm issues
   - **After:** Process in batches → Matches working 8D version
   - **Impact:** Fixes BatchNorm, much faster, better gradients

2. **Discriminator Initialization**
   - **Small weights (gain=0.1):** Prevents saturation
   - **Auto-reset:** Detects and resets if discriminator saturates
   - **Impact:** Prevents all-predictions-same issue

3. **Numerical Stability**
   - **Prediction clamping:** [1e-7, 1-1e-7] prevents log(0) errors
   - **Gradient clipping:** max_norm=1.0 for discriminator stability
   - **Impact:** Prevents NaN/inf losses

4. **Comprehensive CBS Preprocessing**
   - **All fields included:** Automatically extracts all numeric fields
   - **Robust handling:** Gracefully handles missing fields
   - **Impact:** Preserves all available information

---

## Troubleshooting

### Common Issues and Solutions

1. **Discriminator Saturation (all predictions = 1.0)**
   - **Symptom:** Loss = -49.0, gradients = 0.0
   - **Solution:** Auto-reset implemented, better initialization prevents it

2. **BatchNorm Issues**
   - **Symptom:** Features collapse, poor training
   - **Solution:** Batch processing ensures BatchNorm receives proper batches

3. **Invalid Loss Values**
   - **Symptom:** Loss > 10.0 or < 0.0
   - **Solution:** Prediction clamping and saturation detection

4. **No Gradients**
   - **Symptom:** Gradients = 0.0, no learning
   - **Solution:** Batch processing + proper feature gradient flow