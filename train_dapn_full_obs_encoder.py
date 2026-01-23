"""
Train DAPN encoder using FULL raw observations (not 8D unified format).
Uses full observation encoders with adversarial domain adaptation.
"""

import os
import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "cyberwheel"))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from adapters.unified_env import UnifiedSecEnv
from adapters.dapn_unified_full_obs_translator import DAPNUnifiedFullObsTranslator
from adapters.dapn_observation_encoder import DAPNDomainAdapter
from adapters.unified_full_obs_preprocessor import UnifiedFullObsPreprocessor
from config.env_builders import make_cbs_env, make_cw_env


class FullObservationDataset(Dataset):
    """Dataset for full raw observations from multiple domains."""
    
    def __init__(self, source_obs_list, target_obs_list, val_obs_list=None):
        self.source_obs = source_obs_list  # Raw CW observations (arrays)
        self.target_obs = target_obs_list  # Raw CBS observations (dicts)
        self.val_obs = val_obs_list or []  # Raw CBS validation observations (dicts)
        self.total_samples = len(source_obs_list) + len(target_obs_list) + len(self.val_obs)
    
    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx):
        if idx < len(self.source_obs):
            return self.source_obs[idx], 0  # 0 = Source domain (Cyberwheel)
        elif idx < len(self.source_obs) + len(self.target_obs):
            target_idx = idx - len(self.source_obs)
            return self.target_obs[target_idx], 1  # 1 = Target domain (Normal CyberBattleSim)
        else:
            val_idx = idx - len(self.source_obs) - len(self.target_obs)
            return self.val_obs[val_idx], 2  # 2 = Validation domain


def collect_full_observations(
    num_samples=1000,
    save_path=None,
    cw_available=True,
    use_3_domains=True,
    cw_agent_path=None,
    cbs_agent_path=None
):
    """
    Collect FULL raw observations from domains (no 8D conversion).
    
    Args:
        num_samples: Number of samples to collect per domain
        save_path: Optional path to save collected observations
        cw_available: Whether to try collecting Cyberwheel observations
        use_3_domains: Whether to collect from 3 domains
        cw_agent_path: Optional path to Cyberwheel agent checkpoint
        cbs_agent_path: Optional path to CBS agent checkpoint
    """
    # Load agents if provided
    cw_agent = None
    cbs_agent = None
    
    if cw_agent_path and os.path.exists(cw_agent_path):
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            from cyberwheel.utils import RLPolicy
            from eval.eval_cw_checkpoints_on_cbs import infer_cyberwheel_config
            print(f"Loading Cyberwheel agent from {cw_agent_path}...")
            action_space_size, obs_space_shape = infer_cyberwheel_config(cw_agent_path)
            cw_agent = RLPolicy(action_space_shape=action_space_size, obs_space_shape=obs_space_shape).to(device)
            state_dict = torch.load(cw_agent_path, map_location=device)
            cw_agent.load_state_dict(state_dict)
            cw_agent.eval()
            print(f"✓ Loaded Cyberwheel agent")
        except Exception as e:
            print(f"Warning: Could not load Cyberwheel agent: {e}")
            cw_agent = None
    
    if cbs_agent_path and os.path.exists(cbs_agent_path):
        try:
            from stable_baselines3 import PPO
            print(f"Loading CBS agent from {cbs_agent_path}...")
            cbs_agent = PPO.load(cbs_agent_path)
            print(f"✓ Loaded CBS agent")
        except Exception as e:
            print(f"Warning: Could not load CBS agent: {e}")
            cbs_agent = None
    
    # Domain 1: Source (Cyberwheel) - collect RAW observations
    source_obs_list = []
    if cw_available:
        agent_type = "Cyberwheel agent" if cw_agent else "random actions"
        print(f"Collecting FULL raw observations from Source domain (Cyberwheel) using {agent_type}...")
        try:
            cw_env = UnifiedSecEnv("cw", cw_factory=make_cw_env)
            
            for _ in tqdm(range(num_samples)):
                obs, _ = cw_env.reset()
                raw_obs = getattr(cw_env, '_last_raw_obs', None)
                
                # Cyberwheel returns dict with "blue" and "red" keys, each containing obs_vec
                # Extract the red agent's observation (attacker perspective)
                if isinstance(raw_obs, dict):
                    if "red" in raw_obs and raw_obs["red"] is not None:
                        raw_obs = raw_obs["red"]
                    elif "blue" in raw_obs and raw_obs["blue"] is not None:
                        raw_obs = raw_obs["blue"]
                    else:
                        raw_obs = None
                
                if raw_obs is None:
                    raw_obs = obs if isinstance(obs, np.ndarray) else np.array([])
                
                # Ensure it's a numpy array
                if not isinstance(raw_obs, np.ndarray):
                    raw_obs = np.asarray(raw_obs, dtype=np.float32)
                
                # Store RAW observation (no conversion to 8D!)
                source_obs_list.append(raw_obs.copy())
                
                # Use agent if available, else random
                if cw_agent is not None:
                    device = next(cw_agent.parameters()).device
                    obs_tensor = torch.FloatTensor(raw_obs).unsqueeze(0).to(device)
                    with torch.no_grad():
                        action_probs = cw_agent(obs_tensor)
                        action = torch.multinomial(action_probs, 1).item()
                else:
                    action = cw_env.action_space.sample()
                cw_env.step(action)
        except Exception as e:
            print(f"Warning: Could not collect Cyberwheel observations: {e}")
    
    # Domain 2: Target (Normal CyberBattleSim) - collect RAW observations
    target_obs_list = []
    agent_type = "CBS agent" if cbs_agent else "random actions"
    print(f"Collecting FULL raw observations from Target domain (Normal CyberBattleSim) using {agent_type}...")
    target_cbs_env = UnifiedSecEnv("cbs", cbs_factory=make_cbs_env)
    
    for _ in tqdm(range(num_samples)):
        obs, _ = target_cbs_env.reset()
        raw_obs = getattr(target_cbs_env, '_last_raw_cbs_obs', None) or getattr(target_cbs_env, '_last_raw_obs', None)
        if raw_obs is None:
            raw_obs = obs if isinstance(obs, dict) else {}
        
        # Store RAW observation dict (no conversion to 8D!)
        if isinstance(raw_obs, dict):
            # Make a copy of the dict
            raw_obs_copy = {}
            for k, v in raw_obs.items():
                if isinstance(v, np.ndarray):
                    raw_obs_copy[k] = v.copy()
                else:
                    raw_obs_copy[k] = v
            target_obs_list.append(raw_obs_copy)
        else:
            target_obs_list.append(raw_obs)
        
        # Use agent if available, else random
        if cbs_agent is not None:
            action, _ = cbs_agent.predict(obs, deterministic=False)
        else:
            action = target_cbs_env.action_space.sample()
        target_cbs_env.step(action)
    
    # Domain 3: Validation (CBS with Cyberwheel topology) - optional
    val_obs_list = []
    if use_3_domains:
        print(f"Collecting FULL raw observations from Validation domain (CBS with Cyberwheel topology)...")
        try:
            # Set environment to use Cyberwheel topology
            os.environ["CBS_USE_CW_TOPOLOGY"] = "1"
            val_cbs_env = UnifiedSecEnv("cbs", cbs_factory=make_cbs_env)
            
            for _ in tqdm(range(min(num_samples // 5, 200))):  # Fewer validation samples
                obs, _ = val_cbs_env.reset()
                raw_obs = getattr(val_cbs_env, '_last_raw_cbs_obs', None) or getattr(val_cbs_env, '_last_raw_obs', None)
                if raw_obs is None:
                    raw_obs = obs if isinstance(obs, dict) else {}
                
                # Store RAW observation dict
                if isinstance(raw_obs, dict):
                    raw_obs_copy = {}
                    for k, v in raw_obs.items():
                        if isinstance(v, np.ndarray):
                            raw_obs_copy[k] = v.copy()
                        else:
                            raw_obs_copy[k] = v
                    val_obs_list.append(raw_obs_copy)
                else:
                    val_obs_list.append(raw_obs)
                
                action = val_cbs_env.action_space.sample()
                val_cbs_env.step(action)
        except Exception as e:
            print(f"Warning: Could not collect validation observations: {e}")
            val_obs_list = []
    
    print(f"\nCollected observations:")
    print(f"  Source (Cyberwheel): {len(source_obs_list)} samples")
    print(f"  Target (Normal CBS): {len(target_obs_list)} samples")
    print(f"  Validation (CBS with CW topology): {len(val_obs_list)} samples")
    
    # Save if requested
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        np.savez(
            save_path,
            source_obs=source_obs_list,
            target_obs=target_obs_list,
            val_obs=val_obs_list
        )
        print(f"Saved observations to {save_path}")
    
    return source_obs_list, target_obs_list, val_obs_list


def compute_normalization_stats(translator, source_obs_list, target_obs_list, sample_size=100):
    """
    Compute normalization statistics from a sample of observations.
    Returns max values for each dimension to use for normalization.
    """
    print("Computing normalization statistics from sample observations...")
    unified_vecs = []
    
    # Sample from source
    for obs in source_obs_list[:sample_size]:
        try:
            unified_vec = translator.preprocessor.preprocess_cw(obs)
            unified_vecs.append(unified_vec)
        except:
            pass
    
    # Sample from target
    for obs in target_obs_list[:sample_size]:
        try:
            unified_vec = translator.preprocessor.preprocess_cbs(obs)
            unified_vecs.append(unified_vec)
        except:
            pass
    
    if len(unified_vecs) > 0:
        all_vecs = np.stack(unified_vecs)
        # Compute actual max values per dimension
        max_vals = np.max(np.abs(all_vecs), axis=0)
        
        # Check if values are already very small (normalized or mostly zeros)
        actual_max = max_vals.max()
        actual_mean = np.abs(all_vecs).mean()
        
        print(f"  Input value ranges: min={all_vecs.min():.4f}, max={all_vecs.max():.4f}, mean={all_vecs.mean():.4f}, abs_max={actual_max:.4f}")
        
        # Check if values are in [-1, 1] range (already normalized but with negatives)
        if all_vecs.min() >= -1.1 and all_vecs.max() <= 1.1:
            # Values are in [-1, 1] range, convert to [0, 1] for encoder
            # Use standard normalization: (x - min) / (max - min) -> [0, 1]
            # But we'll use a simpler approach: (x + 1) / 2 -> [0, 1]
            print(f"  Values are in [-1, 1] range, will convert to [0, 1] during normalization")
            # Set max_vals to 2.0 so that (x + 1) / 2 maps [-1, 1] -> [0, 1]
            max_vals = np.ones_like(max_vals) * 2.0
        elif actual_max < 2.0:
            # Values are likely already normalized in [0, 1] or very small
            # Use 99th percentile to be robust to outliers
            percentile_99 = np.percentile(np.abs(all_vecs), 99, axis=0)
            max_vals = np.maximum(max_vals, percentile_99)
            # Scale up if values are too small
            if actual_max < 0.1:
                # Values are very small, scale them up
                scale_factor = 10.0 / actual_max if actual_max > 0 else 100.0
                max_vals = max_vals * scale_factor
                print(f"  Values are very small, scaling by {scale_factor:.2f}x")
        else:
            # Values are in natural range, use them as-is
            pass
        
        # Ensure minimum value to avoid division by zero, but preserve scale
        # Only set minimum if the actual max is very small
        if actual_max < 0.01:
            max_vals = np.maximum(max_vals, 1.0)  # Force to 1.0 only if values are tiny
        else:
            max_vals = np.maximum(max_vals, actual_max * 0.01)  # At least 1% of actual max
        
        print(f"  Computed max values: min={max_vals.min():.4f}, max={max_vals.max():.4f}, mean={max_vals.mean():.4f}")
        return max_vals.astype(np.float32)
    else:
        print("  Warning: Could not compute stats, using default normalization")
        return np.ones(translator.unified_dim, dtype=np.float32) * 100.0


def normalize_observation(unified_vec, max_vals):
    """
    Normalize observation vector to [0, 1] range.
    Handles both [-1, 1] -> [0, 1] conversion and standard normalization.
    """
    # Handle [-1, 1] -> [0, 1] conversion if max_vals is 2.0
    if len(max_vals) > 0 and max_vals[0] == 2.0:  # Detected [-1, 1] range
        normalized = np.clip((unified_vec + 1.0) / 2.0, 0.0, 1.0)
    else:
        normalized = np.clip(unified_vec / max_vals, 0.0, 1.0)
    return normalized


def train_dapn_full_obs_encoder(
    source_obs_list,
    target_obs_list,
    val_obs_list=None,
    feature_size=256,
    num_epochs=50,
    batch_size=32,  # Smaller batch size for full observations
    learning_rate=0.001,
    device=None,
    save_path="artifacts/transfer_models/dapn_full_obs_encoder.pt",
    use_3_domains=True
):
    """
    Train DAPN encoder using full raw observations with adversarial domain adaptation.
    
    Args:
        source_obs_list: List of raw Cyberwheel observations (arrays)
        target_obs_list: List of raw CBS observations (dicts)
        val_obs_list: List of raw CBS validation observations (dicts)
        feature_size: Size of feature space
        num_epochs: Number of training epochs
        batch_size: Batch size (smaller for full observations)
        learning_rate: Learning rate
        device: Device to train on
        save_path: Path to save trained encoder
        use_3_domains: Whether to use 3-domain setup
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    # Use SINGLE encoder with unified preprocessing (follows DAPN master)
    print("Using SINGLE shared encoder with unified preprocessing (follows DAPN master concept)")
    unified_dim = 512  # Fixed size for unified representation
    
    translator = DAPNUnifiedFullObsTranslator(
        use_dapn=True,
        feature_size=feature_size,
        unified_dim=unified_dim,
        device=device,
        use_adversarial=True
    )
    
    # Set to training mode
    translator.shared_encoder.train()
    if translator.domain_adapter:
        translator.domain_adapter.train()
    
    print(f"\nFull Observation DAPN Training Setup:")
    print(f"  Source domain (Cyberwheel): {len(source_obs_list)} samples")
    print(f"  Target domain (Normal CyberBattleSim): {len(target_obs_list)} samples")
    print(f"  Validation domain: {len(val_obs_list)} samples")
    print(f"  Feature size: {feature_size}")
    print(f"  Method: Adversarial Domain Adaptation (DANN)\n")
    
    # Compute normalization statistics from data
    normalization_max_vals = compute_normalization_stats(translator, source_obs_list, target_obs_list)
    
    # Create dataset
    dataset = FullObservationDataset(source_obs_list, target_obs_list, val_obs_list)
    
    # Custom collate function to handle variable-length observations
    def collate_fn(batch):
        """Collate function that handles variable-length observations."""
        obs_list, domain_labels = zip(*batch)
        return list(obs_list), torch.tensor(domain_labels, dtype=torch.float32)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    
    # Optimizers
    encoder_params = list(translator.shared_encoder.parameters())
    domain_adapter = translator.domain_adapter
    
    # Use slightly lower learning rate for encoder to stabilize training
    encoder_lr = learning_rate
    optimizer_encoder = optim.Adam(encoder_params, lr=encoder_lr, weight_decay=1e-5)
    
    optimizer_adversarial = None
    if domain_adapter is not None:
        # Use slower learning rate for discriminator to prevent it from becoming too good too fast
        # This allows the encoder time to align features before discriminator becomes perfect
        discriminator_lr = learning_rate * 0.1  # 10x slower than encoder (same as working 8D version)
        optimizer_adversarial = optim.Adam(
            domain_adapter.parameters(),
            lr=discriminator_lr,
            weight_decay=1e-5
        )
        print(f"Discriminator learning rate: {discriminator_lr:.6f} (encoder: {encoder_lr:.6f}, 10x slower)")
    
    # Loss functions
    bce_loss = nn.BCELoss()
    mse_loss = nn.MSELoss()
    
    # Check initial feature statistics to verify domains are distinguishable
    print("\nChecking initial feature statistics...")
    sample_size = min(50, len(source_obs_list), len(target_obs_list))
    if sample_size > 0:
        sample_source = source_obs_list[:sample_size]
        sample_target = target_obs_list[:sample_size]
        
        source_feats = []
        target_feats = []
        for obs in sample_source:
            try:
                unified_vec = translator.preprocessor.preprocess_cw(obs)
                max_vals = normalization_max_vals if normalization_max_vals is not None else np.ones(len(unified_vec), dtype=np.float32) * 100.0
                normalized = normalize_observation(unified_vec, max_vals)
                obs_tensor = torch.from_numpy(normalized).float().to(device)
                with torch.no_grad():
                    feat = translator.shared_encoder(obs_tensor)
                    source_feats.append(feat.cpu().numpy())
            except:
                pass
        
        for obs in sample_target:
            try:
                unified_vec = translator.preprocessor.preprocess_cbs(obs)
                max_vals = normalization_max_vals if normalization_max_vals is not None else np.ones(len(unified_vec), dtype=np.float32) * 100.0
                normalized = normalize_observation(unified_vec, max_vals)
                obs_tensor = torch.from_numpy(normalized).float().to(device)
                with torch.no_grad():
                    feat = translator.shared_encoder(obs_tensor)
                    target_feats.append(feat.cpu().numpy())
            except:
                pass
        
        if len(source_feats) > 0 and len(target_feats) > 0:
            source_feats = np.array(source_feats)
            target_feats = np.array(target_feats)
            src_mean = source_feats.mean(axis=0)
            tgt_mean = target_feats.mean(axis=0)
            src_std = source_feats.std(axis=0)
            tgt_std = target_feats.std(axis=0)
            
            # Check if features are distinguishable
            mean_diff = np.abs(src_mean - tgt_mean).mean()
            std_diff = np.abs(src_std - tgt_std).mean()
            
            print(f"  Source features: μ={src_mean.mean():.4f}, σ={src_std.mean():.4f}")
            print(f"  Target features: μ={tgt_mean.mean():.4f}, σ={tgt_std.mean():.4f}")
            print(f"  Mean difference: {mean_diff:.4f}, Std difference: {std_diff:.4f}")
            
            if mean_diff < 0.01 and std_diff < 0.01:
                print("  WARNING: Features are very similar! Domains may not be distinguishable.")
            else:
                print("  Features appear distinguishable between domains.")
    
    print(f"\nTraining for {num_epochs} epochs...")
    print(f"Total batches per epoch: ~{len(dataloader)}")
    print("-" * 80)
    import sys
    sys.stdout.flush()  # Ensure output is flushed before training starts
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        total_adv_loss = 0.0
        total_discriminator_loss = 0.0
        total_adversarial_loss = 0.0
        total_discriminator_acc = 0.0
        num_batches = 0
        num_discriminator_updates = 0
        last_source_features = None
        last_target_features = None
        skipped_batches = 0
        encoding_errors = 0
        
        # Accumulate observations (not features) to avoid in-place modification issues
        accumulated_source_obs = []
        accumulated_target_obs = []
        
        # Progress bar for epoch (enabled to show progress)
        from tqdm import tqdm
        epoch_pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False, ncols=80, disable=False)
        
        for batch_idx, (obs_batch, domain_labels) in enumerate(epoch_pbar):
            domain_labels = domain_labels.to(device)
            
            # Collect observations by domain
            batch_source_obs = []
            batch_target_obs = []
            
            for obs, domain_label in zip(obs_batch, domain_labels):
                domain_label_int = int(domain_label.item())
                
                try:
                    if domain_label_int == 0:  # Source (Cyberwheel)
                        batch_source_obs.append(obs)
                    elif domain_label_int == 1:  # Target (CBS)
                        batch_target_obs.append(obs)
                    # Skip validation domain for adversarial training
                except Exception as e:
                    encoding_errors += 1
                    if encoding_errors <= 3:
                        import traceback
                        print(f"\nObservation collection error #{encoding_errors} (domain={domain_label_int}): {type(e).__name__}: {e}", flush=True)
                    continue
            
            # Accumulate observations across batches
            accumulated_source_obs.extend(batch_source_obs)
            accumulated_target_obs.extend(batch_target_obs)
            
            # Update progress bar to show accumulation status
            epoch_pbar.set_postfix({
                'src': len(accumulated_source_obs),
                'tgt': len(accumulated_target_obs)
            })
            
            # Process when we have both source and target observations
            # Limit batch size to prevent memory issues (process in chunks of max 64)
            max_batch_size = 64
            if len(accumulated_source_obs) > 0 and len(accumulated_target_obs) > 0:
                # Take up to max_batch_size from each domain
                num_to_process = min(
                    max_batch_size,
                    len(accumulated_source_obs),
                    len(accumulated_target_obs)
                )
                
                # Extract batch of observations
                source_obs_batch = accumulated_source_obs[:num_to_process]
                target_obs_batch = accumulated_target_obs[:num_to_process]
                
                # Remove processed observations
                accumulated_source_obs = accumulated_source_obs[num_to_process:]
                accumulated_target_obs = accumulated_target_obs[num_to_process:]
                
                # Encode observations in BATCHES (like working 8D version) - much faster and fixes BatchNorm
                # Preprocess all observations first
                source_unified_vecs = []
                for obs in source_obs_batch:
                    unified_vec = translator.preprocessor.preprocess_cw(obs)
                    max_vals = normalization_max_vals if normalization_max_vals is not None else np.ones(len(unified_vec), dtype=np.float32) * 100.0
                    normalized = normalize_observation(unified_vec, max_vals)
                    source_unified_vecs.append(normalized)
                
                target_unified_vecs = []
                for obs in target_obs_batch:
                    unified_vec = translator.preprocessor.preprocess_cbs(obs)
                    max_vals = normalization_max_vals if normalization_max_vals is not None else np.ones(len(unified_vec), dtype=np.float32) * 100.0
                    normalized = normalize_observation(unified_vec, max_vals)
                    target_unified_vecs.append(normalized)
                
                # Convert to batch tensors and encode in batch (like 8D version)
                if len(source_unified_vecs) > 0:
                    source_batch_tensor = torch.from_numpy(np.array(source_unified_vecs)).float().to(device)
                    source_features_encoder = translator.shared_encoder(source_batch_tensor)
                else:
                    source_features_encoder = None
                
                if len(target_unified_vecs) > 0:
                    target_batch_tensor = torch.from_numpy(np.array(target_unified_vecs)).float().to(device)
                    target_features_encoder = translator.shared_encoder(target_batch_tensor)
                else:
                    target_features_encoder = None
            else:
                # Skip this iteration if we don't have both domains yet
                continue
            
            # Store for statistics
            last_source_features = source_features_encoder
            last_target_features = target_features_encoder
            
            # Adversarial domain adaptation - EXACTLY like working 8D version
            if domain_adapter is not None:
                # Detach features for adversarial network update (to avoid gradient issues)
                # We'll use detached features for adversarial network, but original for encoder update
                all_features_detached = torch.cat([source_features_encoder.detach(), target_features_encoder.detach()], dim=0)
                all_features_for_encoder = torch.cat([source_features_encoder, target_features_encoder], dim=0)
                
                # Create domain labels: 0 for source (Cyberwheel), 1 for target (CBS)
                domain_targets = torch.cat([
                    torch.zeros(source_features_encoder.size(0), 1).to(device),
                    torch.ones(target_features_encoder.size(0), 1).to(device)
                ], dim=0)
                
                # Step 1: Update adversarial network to maximize domain discrimination
                if optimizer_adversarial is not None:
                    optimizer_adversarial.zero_grad()
                    domain_pred_adv = domain_adapter(all_features_detached)
                    
                    # Clamp predictions to prevent numerical issues
                    domain_pred_adv = torch.clamp(domain_pred_adv, 1e-7, 1.0 - 1e-7)
                    
                    adv_loss = bce_loss(domain_pred_adv, domain_targets)
                    
                    # Check if loss is valid
                    if adv_loss.item() > 10.0 or adv_loss.item() < 0.0:
                        print(f"WARNING: Invalid discriminator loss: {adv_loss.item()}")
                        print(f"  domain_pred_adv range: [{domain_pred_adv.min().item():.4f}, {domain_pred_adv.max().item():.4f}]")
                        # Reset discriminator if loss is invalid
                        for m in domain_adapter.modules():
                            if isinstance(m, nn.Linear):
                                nn.init.xavier_uniform_(m.weight, gain=0.1)
                                if m.bias is not None:
                                    nn.init.zeros_(m.bias)
                        # Recompute with reset discriminator
                        domain_pred_adv = domain_adapter(all_features_detached)
                        domain_pred_adv = torch.clamp(domain_pred_adv, 1e-7, 1.0 - 1e-7)
                        adv_loss = bce_loss(domain_pred_adv, domain_targets)
                    
                    adv_loss.backward()
                    # Gradient clipping for discriminator
                    torch.nn.utils.clip_grad_norm_(domain_adapter.parameters(), max_norm=1.0)
                    optimizer_adversarial.step()
                    
                    # Calculate discriminator accuracy for logging
                    pred_labels = (domain_pred_adv > 0.5).float()
                    disc_acc = (pred_labels == domain_targets).float().mean().item()
                    total_discriminator_acc += disc_acc
                    total_discriminator_loss += adv_loss.item()
                    num_discriminator_updates += 1
                else:
                    adv_loss = None
                
                # Step 2: Compute adversarial loss for encoder (to confuse discriminator)
                # Use non-detached features so gradients flow to encoders
                # Note: Discriminator was just updated, so we need to recompute predictions
                domain_pred_encoder = domain_adapter(all_features_for_encoder)
                
                # Clamp predictions to prevent numerical issues (shouldn't be needed but safety check)
                domain_pred_encoder = torch.clamp(domain_pred_encoder, 1e-7, 1.0 - 1e-7)
                
                adv_loss_for_encoder = bce_loss(domain_pred_encoder, domain_targets)
                
                # Debug: Check if loss is valid and discriminator is saturated
                if adv_loss_for_encoder.item() > 10.0 or adv_loss_for_encoder.item() < 0.0:
                    print(f"WARNING: Invalid adversarial loss: {adv_loss_for_encoder.item()}")
                    print(f"  domain_pred_encoder range: [{domain_pred_encoder.min().item():.4f}, {domain_pred_encoder.max().item():.4f}]")
                    print(f"  domain_targets range: [{domain_targets.min().item():.4f}, {domain_targets.max().item():.4f}]")
                    print(f"  Features require_grad: {all_features_for_encoder.requires_grad}")
                    print(f"  Encoder training mode: {translator.shared_encoder.training}")
                    print(f"  Feature stats - Source: mean={source_features_encoder.mean().item():.4f}, std={source_features_encoder.std().item():.4f}")
                    print(f"  Feature stats - Target: mean={target_features_encoder.mean().item():.4f}, std={target_features_encoder.std().item():.4f}")
                
                # Check if discriminator is saturated (all predictions same)
                pred_std = domain_pred_encoder.std().item()
                if pred_std < 1e-6:
                    print(f"WARNING: Discriminator saturated! All predictions={domain_pred_encoder[0].item():.4f}, std={pred_std:.6f}")
                    # Reset discriminator to break saturation
                    for m in domain_adapter.modules():
                        if isinstance(m, nn.Linear):
                            nn.init.xavier_uniform_(m.weight, gain=0.1)
                            if m.bias is not None:
                                nn.init.zeros_(m.bias)
                
                # Total loss: Only adversarial (inverted for domain confusion) - EXACTLY like 8D version
                # We want encoders to produce features that confuse the domain discriminator
                total_batch_loss = 1.0 - adv_loss_for_encoder
                
                # Add a small regularization to prevent the loss from being exactly 1.0
                # This ensures gradients flow even when discriminator is very good
                if adv_loss_for_encoder.item() < 0.01:
                    # If discriminator is too good, add a small penalty to encourage feature alignment
                    total_batch_loss = total_batch_loss + 0.1 * mse_loss(
                        source_features_encoder.mean(dim=0), target_features_encoder.mean(dim=0)
                    )
                
                # Update encoder
                optimizer_encoder.zero_grad()
                total_batch_loss.backward()
                
                # Check if gradients exist before stepping
                has_gradients = False
                for p in encoder_params:
                    if p.grad is not None and p.grad.abs().sum() > 0:
                        has_gradients = True
                        break
                
                if not has_gradients:
                    print(f"WARNING: No gradients detected! Loss={total_batch_loss.item():.4f}, adv_loss={adv_loss_for_encoder.item():.4f}")
                    print(f"  Features require_grad: {all_features_for_encoder.requires_grad}")
                    print(f"  Encoder training: {translator.shared_encoder.training}")
                
                optimizer_encoder.step()
                
                # Track losses for logging
                total_loss += total_batch_loss.item()
                # Use discriminator loss for adversarial loss tracking (what discriminator sees)
                if adv_loss is not None:
                    total_adversarial_loss += adv_loss.item()
                else:
                    # Fallback: use encoder adversarial loss (but this is what encoder sees, not discriminator)
                    total_adversarial_loss += adv_loss_for_encoder.item()
                
                num_batches += 1
            elif source_features_encoder is not None and target_features_encoder is not None:
                # Fallback: if no domain adapter, use a simple feature matching loss
                optimizer_encoder.zero_grad()
                match_loss = mse_loss(
                    source_features_encoder.mean(dim=0, keepdim=False), 
                    target_features_encoder.mean(dim=0, keepdim=False)
                )
                match_loss.backward()
                optimizer_encoder.step()
                total_loss += match_loss.item()
                num_batches += 1
        
        # Process any remaining accumulated observations at the end of the epoch
        if len(accumulated_source_obs) > 0 and len(accumulated_target_obs) > 0:
            # Preprocess all observations first
            source_unified_vecs = []
            for obs in accumulated_source_obs:
                unified_vec = translator.preprocessor.preprocess_cw(obs)
                max_vals = normalization_max_vals if normalization_max_vals is not None else np.ones(len(unified_vec), dtype=np.float32) * 100.0
                normalized = normalize_observation(unified_vec, max_vals)
                source_unified_vecs.append(normalized)
            
            target_unified_vecs = []
            for obs in accumulated_target_obs:
                unified_vec = translator.preprocessor.preprocess_cbs(obs)
                max_vals = normalization_max_vals if normalization_max_vals is not None else np.ones(len(unified_vec), dtype=np.float32) * 100.0
                normalized = normalize_observation(unified_vec, max_vals)
                target_unified_vecs.append(normalized)
            
            # Convert to batch tensors and encode in batch (like 8D version)
            if len(source_unified_vecs) > 0:
                source_batch_tensor = torch.from_numpy(np.array(source_unified_vecs)).float().to(device)
                source_features_enc = translator.shared_encoder(source_batch_tensor)
            else:
                source_features_enc = None
            
            if len(target_unified_vecs) > 0:
                target_batch_tensor = torch.from_numpy(np.array(target_unified_vecs)).float().to(device)
                target_features_enc = translator.shared_encoder(target_batch_tensor)
            else:
                target_features_enc = None
            
            # Process this final batch - EXACTLY like working 8D version
            if domain_adapter is not None:
                # Detach features for adversarial network update
                all_features_detached = torch.cat([source_features_enc.detach(), target_features_enc.detach()], dim=0)
                all_features_for_encoder = torch.cat([source_features_enc, target_features_enc], dim=0)
                
                domain_targets = torch.cat([
                    torch.zeros(source_features_enc.size(0), 1).to(device),
                    torch.ones(target_features_enc.size(0), 1).to(device)
                ], dim=0)
                
                # Step 1: Update adversarial network
                if optimizer_adversarial is not None:
                    optimizer_adversarial.zero_grad()
                    domain_pred_adv = domain_adapter(all_features_detached)
                    adv_loss = bce_loss(domain_pred_adv, domain_targets)
                    adv_loss.backward()
                    optimizer_adversarial.step()
                    
                    disc_acc = ((domain_pred_adv > 0.5).float() == domain_targets).float().mean().item()
                    total_discriminator_acc += disc_acc
                    total_discriminator_loss += adv_loss.item()
                    num_discriminator_updates += 1
                else:
                    adv_loss = None
                
                # Step 2: Compute adversarial loss for encoder
                domain_pred_encoder = domain_adapter(all_features_for_encoder)
                adv_loss_for_encoder = bce_loss(domain_pred_encoder, domain_targets)
                
                # Total loss: Only adversarial (inverted for domain confusion) - EXACTLY like 8D version
                total_batch_loss = 1.0 - adv_loss_for_encoder
                
                if adv_loss_for_encoder.item() < 0.01:
                    total_batch_loss = total_batch_loss + 0.1 * mse_loss(
                        source_features_enc.mean(dim=0), target_features_enc.mean(dim=0)
                    )
                
                # Update encoder
                optimizer_encoder.zero_grad()
                total_batch_loss.backward()
                optimizer_encoder.step()
                
                total_loss += total_batch_loss.item()
                if adv_loss is not None:
                    total_adversarial_loss += adv_loss.item()
                elif adv_loss_for_encoder is not None:
                    total_adversarial_loss += adv_loss_for_encoder.item()
                num_batches += 1
        
        # Calculate averages (handle case where num_batches might be 0)
        if num_batches > 0:
            avg_encoder_loss = total_loss / num_batches
            avg_discriminator_loss = total_discriminator_loss / num_discriminator_updates if num_discriminator_updates > 0 else 0.0
            # Adversarial loss should be averaged over batches (it's computed for each batch)
            avg_adversarial_loss = total_adversarial_loss / num_batches if num_batches > 0 else 0.0
            avg_disc_acc = total_discriminator_acc / num_discriminator_updates if num_discriminator_updates > 0 else 0.0
        else:
            # If no batches processed, use zeros (shouldn't happen but handle gracefully)
            avg_encoder_loss = 0.0
            avg_adversarial_loss = 0.0
        
        # Print every 10 epochs (and always print the final epoch)
        if (epoch + 1) % 10 == 0 or (epoch + 1) == num_epochs:
            import sys
            if num_batches == 0:
                # Debug info if no batches were processed
                msg = f"Epoch {epoch+1}/{num_epochs}: Loss=0.0000, Adv=0.0000, Disc=0.0000, Acc=0.00% (WARNING: {skipped_batches} batches skipped, {encoding_errors} encoding errors)"
            else:
                # Add feature statistics if available
                feature_stats = ""
                if last_source_features is not None and last_target_features is not None:
                    src_mean = last_source_features.mean().item()
                    tgt_mean = last_target_features.mean().item()
                    src_std = last_source_features.std().item()
                    tgt_std = last_target_features.std().item()
                    mean_diff = abs(src_mean - tgt_mean)
                    feature_stats = f" | FeatDiff={mean_diff:.4f}"
                
                # Add gradient norm info if available
                grad_info = ""
                if num_batches > 0:
                    try:
                        encoder_grad_norm = 0.0
                        disc_grad_norm = 0.0
                        for p in encoder_params:
                            if p.grad is not None:
                                encoder_grad_norm += p.grad.data.norm(2).item() ** 2
                        encoder_grad_norm = encoder_grad_norm ** 0.5
                        
                        if domain_adapter is not None:
                            for p in domain_adapter.parameters():
                                if p.grad is not None:
                                    disc_grad_norm += p.grad.data.norm(2).item() ** 2
                            disc_grad_norm = disc_grad_norm ** 0.5
                        
                        grad_info = f" | Grads: Enc={encoder_grad_norm:.3f}, Disc={disc_grad_norm:.3f}"
                    except:
                        pass
                
                msg = f"Epoch {epoch+1}/{num_epochs}: Loss={avg_encoder_loss:.4f}, Adv={avg_adversarial_loss:.4f}, Disc={avg_discriminator_loss:.4f}, Acc={avg_disc_acc*100:.2f}%{feature_stats}{grad_info}"
            print(msg, flush=True)
            sys.stdout.flush()  # Force flush to ensure output appears
    
    # Save encoder
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    translator.save_encoder(save_path)
    print(f"Saved trained encoder to {save_path}")
    
    return translator


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train DAPN encoder with FULL raw observations")
    parser.add_argument("--num-samples", type=int, default=1000, help="Number of samples per domain")
    parser.add_argument("--feature-size", type=int, default=256, help="Feature space size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (smaller for full obs)")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--load-data", type=str, default=None, help="Path to load pre-collected observations")
    parser.add_argument("--save-data", type=str, default=None, help="Path to save collected observations")
    parser.add_argument("--save-encoder", type=str,
                       default="artifacts/transfer_models/dapn_full_obs_encoder.pt",
                       help="Path to save trained encoder")
    parser.add_argument("--cw-agent", type=str, default=None,
                       help="Path to Cyberwheel agent checkpoint")
    parser.add_argument("--cbs-agent", type=str, default=None,
                       help="Path to CBS agent checkpoint")
    
    args = parser.parse_args()
    
    # Load or collect observations
    if args.load_data and os.path.exists(args.load_data):
        print(f"Loading observations from {args.load_data}")
        data = np.load(args.load_data, allow_pickle=True)
        source_obs_list = data['source_obs'].tolist()
        target_obs_list = data['target_obs'].tolist()
        val_obs_list = data['val_obs'].tolist() if 'val_obs' in data else []
    else:
        # Collect observations
        source_obs_list, target_obs_list, val_obs_list = collect_full_observations(
            num_samples=args.num_samples,
            save_path=args.save_data,
            cw_available=True,
            use_3_domains=True,
            cw_agent_path=args.cw_agent,
            cbs_agent_path=args.cbs_agent
        )
    
    # Guard: DAPN needs both source and target domains
    if len(source_obs_list) == 0 or len(target_obs_list) == 0:
        print("\nERROR: Missing observations for training.")
        print(f"  Source (Cyberwheel): {len(source_obs_list)} samples")
        print(f"  Target (CBS): {len(target_obs_list)} samples")
        print("DAPN training requires both domains. Please fix Cyberwheel collection and rerun.")
        sys.exit(1)

    # Train encoder
    train_dapn_full_obs_encoder(
        source_obs_list=source_obs_list,
        target_obs_list=target_obs_list,
        val_obs_list=val_obs_list,
        feature_size=args.feature_size,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        save_path=args.save_encoder,
        use_3_domains=True
    )
    
    print("\n✓ Training complete!")
    print(f"  Encoder saved to: {args.save_encoder}")
    print(f"  Encoder type: Single shared encoder (follows DAPN master)")
    print(f"\nTo use this encoder, set:")
    print(f"  export DAPN_USE_FULL_OBS=1")
    print(f"  export DAPN_ENCODER_PATH={args.save_encoder}")