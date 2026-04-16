# DAPN Implementation Details

## Overview

DAPN (Domain Adaptive Prototypical Network) is used for domain adaptation between CyberBattleSim (CBS) and Cyberwheel (CW) environments. It aligns observations from different domains into a shared feature space to enable transfer learning.

---

## 1. Network Architecture

### **DAPNObservationEncoder**

**Purpose**: Encodes 8D normalized observations into high-dimensional feature vectors.

**Architecture**:
```
Input: 8D normalized observation [0.0, 1.0]
  ↓
Linear(8 → 128)
  ↓
BatchNorm1d(128)
  ↓
ReLU + Dropout(0.2)
  ↓
Linear(128 → 256)
  ↓
BatchNorm1d(256)
  ↓
ReLU + Dropout(0.2)
  ↓
Linear(256 → 512)
  ↓
BatchNorm1d(512)
  ↓
ReLU
  ↓
[Optional Bottleneck]
Linear(512 → feature_size)  # default: 256
  ↓
Output: feature_size-dimensional feature vector
```

**Key Components**:
- **Feature Layers**: 3-layer MLP (8→128→256→512) with BatchNorm and Dropout
- **Bottleneck**: Optional layer (512→256) to reduce dimensionality
- **Initialization**: Xavier normal for Linear layers, normal(1.0, 0.02) for BatchNorm
- **BatchNorm Handling**: Automatically switches to eval mode for batch_size=1 during training

**Two Encoders**:
- `cbs_encoder`: Encodes CBS observations
- `cw_encoder`: Encodes Cyberwheel observations
- Both share the same architecture but have separate parameters

### **DAPNDomainAdapter**

**Purpose**: Adversarial network for domain alignment (DANN method).

**Architecture**:
```
Input: feature_size-dimensional features (default: 256)
  ↓
Linear(feature_size → 1024)
  ↓
ReLU + Dropout(0.5)
  ↓
Linear(1024 → 1024)
  ↓
ReLU + Dropout(0.5)
  ↓
Linear(1024 → 1)
  ↓
Sigmoid
  ↓
Output: Domain prediction (0=source/CW, 1=target/CBS)
```

**Purpose**: Discriminates between source and target domain features. During training, encoders learn to produce features that confuse this discriminator, aligning the feature spaces.

---

## 2. Input

### **Input Format**

**Raw Observations**:
- **CBS**: Dictionary with keys like `discovered_node_count`, `nodes_privilegelevel`, `discovered_nodes_properties`, etc.
- **CW**: NumPy array of variable length (host attributes + standalone attributes)

**Preprocessing Pipeline**:
```
Raw Obs (CBS/CW)
  ↓
Convert to Unified 8D Representation
  ↓
Normalize: obs / [50, 50, 50, 200, 50, 1000, 20.0, 100]
  ↓
Clip to [0.0, 1.0]
  ↓
Input to Encoder (8D normalized vector)
```

**Unified 8D Representation**:
```
[discovered_hosts, compromised_hosts, discovered_hosts, known_vulns,
 credentials_found, steps_elapsed, dist_to_goal, alerts]
```

**CBS → Unified Mapping**:
- `discovered_hosts`: `discovered_node_count`
- `compromised_hosts`: Count of nodes with `privilege_level >= 1`
- `known_vulns`: Sum of discovered node properties
- `credentials_found`: `credential_cache_length`
- `steps_elapsed`: Number of edges in explored network graph
- `dist_to_goal`: 0.0
- `alerts`: `probe_result == 1` + `escalation > 0`

**CW → Unified Mapping**:
- `discovered_hosts`: Count of hosts with `discovered == 1`
- `compromised_hosts`: Count with `on_host + escalated + impacted > 0`
- `known_vulns`: Proxy by `scanned_hosts`
- `credentials_found`: Proxy by `escalated_count`
- `steps_elapsed`: Derived from quadrant attribute
- `dist_to_goal`: `1.0 - (impacted_count / total_hosts)`
- `alerts`: `escalated_count + impacted_count`

---

## 3. Output

### **Output Format**

**Encoder Output**:
- **Shape**: `(feature_size,)` or `(batch_size, feature_size)` where `feature_size=256` (default)
- **Type**: NumPy array (float32)
- **Range**: Unbounded (before L2 normalization if applied)

**Observation Space After Encoding**:
```python
# Without encoder
observation_space = Box(low=0.0, high=1.0, shape=(8,), dtype=np.float32)

# With DAPN encoder
observation_space = Box(low=0.0, high=1.0, shape=(256,), dtype=np.float32)
# OR Dict with mask:
observation_space = Dict({
    "obs": Box(low=0.0, high=1.0, shape=(256,), dtype=np.float32),
    "mask": Box(low=0.0, high=1.0, shape=(7,), dtype=np.float32)
})
```

**Feature Characteristics**:
- High-dimensional representation (256D vs 8D)
- Domain-invariant features (aligned across CBS and CW)
- Rich semantic information for policy learning

---

## 4. Loss Function

### **Training Loss Components**

**1. Adversarial Domain Alignment Loss** (Primary):

```python
# Domain discriminator loss (maximize discrimination)
adv_loss = BCE(domain_adapter(features), domain_labels)

# Encoder loss (minimize discrimination = confuse discriminator)
encoder_loss = 1.0 - adv_loss  # Inverted to align features
```

**Purpose**: Aligns feature spaces by making source and target features indistinguishable to the discriminator.

**2. Feature Matching Loss** (Fallback/Regularization):

```python
# When discriminator is too good (loss < 0.01)
feature_match_loss = MSE(source_features.mean(), target_features.mean())
```

**Purpose**: Directly minimizes distance between domain feature distributions when adversarial training is insufficient.

### **Loss Computation**

**Training Steps**:
1. **Update Discriminator** (maximize domain discrimination):
   ```python
   domain_pred = domain_adapter(features.detach())
   adv_loss = BCE(domain_pred, domain_labels)
   adv_loss.backward()  # Update discriminator
   ```

2. **Update Encoders** (minimize domain discrimination):
   ```python
   domain_pred = domain_adapter(features)  # Non-detached
   encoder_loss = 1.0 - BCE(domain_pred, domain_labels)
   encoder_loss.backward()  # Update encoders
   ```

**Loss Formula**:
```
Total Loss = 1.0 - adv_loss_for_encoder
           + 0.1 * feature_match_loss (if adv_loss < 0.01)
```

---

## 5. Training Procedure

### **Training Setup**

**Domain Configuration** (3-domain setup):
1. **Source Domain**: Cyberwheel observations
2. **Target Domain**: Normal CyberBattleSim observations
3. **Validation Domain**: CBS with Cyberwheel topology (evaluation only)

**Data Collection**:
```python
# Collect observations from each domain
source_obs = collect_from_cyberwheel(num_samples=1000)
target_obs = collect_from_cbs(num_samples=1000)
val_obs = collect_from_cbs_variant(num_samples=200)
```

**Dataset**:
- `ObservationDataset`: Handles multi-domain observations
- Labels: 0=source (CW), 1=target (CBS), 2=validation
- Batch sampling ensures both domains present

### **Training Loop**

```python
for epoch in range(num_epochs):
    for batch in dataloader:
        obs_batch, domain_labels = batch
        
        # Split by domain
        source_mask = domain_labels == 0
        target_mask = domain_labels == 1
        
        # Encode observations
        source_features = cw_encoder(obs_batch[source_mask])
        target_features = cbs_encoder(obs_batch[target_mask])
        
        # Step 1: Update discriminator
        domain_pred = domain_adapter(cat([source_features.detach(), 
                                          target_features.detach()]))
        adv_loss = BCE(domain_pred, domain_labels)
        adv_loss.backward()
        optimizer_adversarial.step()
        
        # Step 2: Update encoders (confuse discriminator)
        domain_pred = domain_adapter(cat([source_features, target_features]))
        encoder_loss = 1.0 - BCE(domain_pred, domain_labels)
        encoder_loss.backward()
        optimizer_encoder.step()
```

### **Hyperparameters**

**Default Values**:
- `feature_size`: 256
- `num_epochs`: 50
- `batch_size`: 64
- `learning_rate`: 0.001 (encoder), 0.0001 (discriminator, 10x slower)
- `input_dim`: 8

**Optimizers**:
- **Encoder**: Adam(lr=0.001)
- **Discriminator**: Adam(lr=0.0001) - slower to allow encoder to align features

**Training Strategy**:
- Discriminator learning rate is 10x slower than encoder
- Prevents discriminator from becoming too good too fast
- Allows encoders time to align features before discriminator becomes perfect

### **Checkpointing**

**Save Format**:
```python
checkpoint = {
    'cbs_encoder_state_dict': cbs_encoder.state_dict(),
    'cw_encoder_state_dict': cw_encoder.state_dict(),
    'domain_adapter_state_dict': domain_adapter.state_dict(),
    'feature_size': 256,
    'input_dim': 8
}
torch.save(checkpoint, 'dapn_encoder.pt')
```

**Load Format**:
- Supports loading shared encoder (if only `encoder_state_dict` present)
- Supports loading separate encoders
- Handles missing components gracefully

---

## 6. Integration with RL Component

### **Environment Wrapper**

**DAPNEnvWrapper**:
```python
class DAPNEnvWrapper(gym.Wrapper):
    def __init__(self, env, encoder_path, feature_size=256):
        # Load DAPN translator
        self.translator = DAPNObservationTranslator(
            use_dapn=True,
            encoder_path=encoder_path,
            feature_size=feature_size
        )
        
        # Update observation space
        self.observation_space = Box(
            low=0.0, high=1.0, shape=(feature_size,), dtype=np.float32
        )
    
    def reset(self):
        obs, info = self.env.reset()
        return self.translator.from_cbs(obs) if CBS else self.translator.from_cw(obs)
    
    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        encoded_obs = self.translator.from_cbs(obs) if CBS else self.translator.from_cw(obs)
        return encoded_obs, reward, done, truncated, info
```

### **Usage with RL Algorithms**

**Training Policy with DAPN**:
```python
from stable_baselines3 import PPO
from adapters.dapn_env_wrapper import DAPNEnvWrapper

# Create base environment
base_env = UnifiedSecEnv("cbs", cbs_factory=make_cbs_env)

# Wrap with DAPN
dapn_env = DAPNEnvWrapper(
    base_env,
    encoder_path="artifacts/transfer_models/dapn_encoder.pt",
    feature_size=256
)

# Train policy (uses 256D features instead of 8D)
model = PPO("MultiInputPolicy", dapn_env, verbose=1)
model.learn(total_timesteps=100000)
```

**Transfer Learning Flow**:
```
1. Train DAPN encoder on source (CW) + target (CBS) observations
   → Saves: dapn_encoder.pt

2. Train policy on source domain (CW) with DAPN encoder
   → Policy learns on 256D aligned features

3. Transfer policy to target domain (CBS) with same DAPN encoder
   → Policy works because features are domain-aligned
```

### **Observation Flow**

**During RL Training/Evaluation**:
```
Environment Step
  ↓
Raw Observation (CBS dict or CW array)
  ↓
DAPNObservationTranslator.from_cbs() or .from_cw()
  ↓
Convert to Unified 8D
  ↓
Normalize [0, 1]
  ↓
DAPNObservationEncoder (8D → 256D)
  ↓
Encoded Features (256D)
  ↓
Policy Input
```

### **Integration Points**

1. **Observation Translation**: Replaces `ObservationTranslator` in `UnifiedSecEnv`
2. **Environment Wrapper**: `DAPNEnvWrapper` handles encoding automatically
3. **Policy Training**: Works with any RL algorithm (PPO, DQN, etc.)
4. **Transfer**: Same encoder used for both source and target domains

### **Advantages**

- **Domain Alignment**: Features from different domains become similar
- **Transfer Learning**: Policy trained on one domain works on another
- **Rich Representations**: 256D features vs 8D raw observations
- **Automatic**: Wrapper handles encoding transparently

---

## Summary

**DAPN Architecture**:
- Two encoders (CBS + CW): 8D → 256D MLP with BatchNorm/Dropout
- Domain adapter: 256D → 1D discriminator (DANN method)

**Training**:
- Adversarial domain alignment (confuse discriminator)
- 3-domain setup (source, target, validation)
- Separate optimizers with different learning rates

**Integration**:
- `DAPNEnvWrapper` for automatic encoding
- Works with any RL algorithm
- Enables transfer learning between domains


