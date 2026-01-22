

import os
import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from adapters.unified_env import UnifiedSecEnv
from adapters.dapn_observation_encoder import DAPNObservationTranslator, DAPNDomainAdapter
from config.env_builders import make_cbs_env, make_cw_env


class ObservationDataset(Dataset):
    """Dataset for observations from multiple domains (DAPN requires 3)."""
    
    def __init__(self, source_obs_list, target_obs_list, val_obs_list=None):
        self.source_obs = source_obs_list
        self.target_obs = target_obs_list
        self.val_obs = val_obs_list or []
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
            return self.val_obs[val_idx], 2  # 2 = Validation domain (CBS with Cyberwheel topology)


def collect_observations(num_samples=1000, save_path=None, cw_available=True, use_3_domains=True,
                         cw_agent_path=None, cbs_agent_path=None):
    """
    Collect observations from domains for domain alignment training.
    
    Collects from 3 domains:
    1. Source domain (Cyberwheel) - source features
    2. Target domain (Normal CyberBattleSim) - target features to align with source
    3. Validation domain (CBS with Cyberwheel topology) - evaluation only
    
    Args:
        num_samples: Number of samples to collect per domain
        save_path: Optional path to save collected observations
        cw_available: Whether to try collecting Cyberwheel observations
        use_3_domains: Whether to collect from 3 domains (True) or 2 (False)
        cw_agent_path: Optional path to Cyberwheel agent checkpoint (uses agent if provided, else random)
        cbs_agent_path: Optional path to CBS agent checkpoint (uses agent if provided, else random)
    """
    translator = DAPNObservationTranslator(use_dapn=False)
    
    # Load agents if provided
    cw_agent = None
    cbs_agent = None
    
    if cw_agent_path and cw_agent_path != "path/to/cyberwheel_agent.pt" and os.path.exists(cw_agent_path):
        try:
            import torch
            from cyberwheel.utils import RLPolicy
            from eval.eval_cw_checkpoints_on_cbs import infer_cyberwheel_config
            print(f"Loading Cyberwheel agent from {cw_agent_path}...")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            action_space_size, obs_space_shape = infer_cyberwheel_config(cw_agent_path)
            cw_agent = RLPolicy(action_space_shape=action_space_size, obs_space_shape=obs_space_shape).to(device)
            state_dict = torch.load(cw_agent_path, map_location=device)
            cw_agent.load_state_dict(state_dict)
            cw_agent.eval()
            print(f"✓ Loaded Cyberwheel agent (action_space={action_space_size}, obs_shape={obs_space_shape})")
        except Exception as e:
            print(f"Warning: Could not load Cyberwheel agent: {e}")
            print("  Falling back to random actions")
            cw_agent = None
    
    if cbs_agent_path and cbs_agent_path != "path/to/cbs_agent.zip" and os.path.exists(cbs_agent_path):
        try:
            from stable_baselines3 import PPO
            print(f"Loading CBS agent from {cbs_agent_path}...")
            cbs_agent = PPO.load(cbs_agent_path)
            print(f"✓ Loaded CBS agent")
        except Exception as e:
            print(f"Warning: Could not load CBS agent: {e}")
            print("  Falling back to random actions")
            cbs_agent = None
    
    # Domain 1: Source (Cyberwheel - training)
    source_obs_list = []
    if cw_available:
        agent_type = "Cyberwheel agent" if cw_agent else "random actions"
        print(f"Collecting observations from Source domain (Cyberwheel) using {agent_type}...")
        try:
            cw_env = UnifiedSecEnv("cw", cw_factory=make_cw_env)
            
            for _ in tqdm(range(num_samples)):
                obs, _ = cw_env.reset()
                raw_obs = getattr(cw_env, '_last_raw_obs', None)
                if raw_obs is None:
                    raw_obs = obs if isinstance(obs, np.ndarray) else np.array([])
                
                unified_obs = translator._cw_to_unified(raw_obs)
                normalized_obs = translator._normalize(unified_obs)
                source_obs_list.append(normalized_obs)
                
                # Use agent if available, else random
                if cw_agent is not None:
                    # Cyberwheel agent expects raw observation vector
                    # Convert to tensor and get action
                    device = next(cw_agent.parameters()).device
                    obs_tensor = torch.FloatTensor(raw_obs).unsqueeze(0).to(device)
                    with torch.no_grad():
                        action_probs = cw_agent(obs_tensor)
                        action = torch.multinomial(action_probs, 1).item()
                else:
                    action = cw_env.action_space.sample()
                cw_env.step(action)
        except (ModuleNotFoundError, ImportError, Exception) as e:
            print(f"Warning: Could not collect Cyberwheel observations: {e}")
            print("Falling back to CBS as source domain...")
            # Fallback: use CBS as source if Cyberwheel unavailable
            cbs_env = UnifiedSecEnv("cbs", cbs_factory=make_cbs_env)
            for _ in tqdm(range(num_samples)):
                obs, _ = cbs_env.reset()
                raw_obs = getattr(cbs_env, '_last_raw_cbs_obs', None) or getattr(cbs_env, '_last_raw_obs', None)
                if raw_obs is None:
                    raw_obs = obs if isinstance(obs, dict) else {}
                unified_obs = translator._cbs_to_unified(raw_obs)
                normalized_obs = translator._normalize(unified_obs)
                source_obs_list.append(normalized_obs)
                action = cbs_env.action_space.sample()
                cbs_env.step(action)
    else:
        print("Skipping Source domain (Cyberwheel) collection")
    
    # Domain 2: Target (Normal CyberBattleSim - adaptation)
    agent_type = "CBS agent" if cbs_agent else "random actions"
    print(f"Collecting observations from Target domain (Normal CyberBattleSim) using {agent_type}...")
    target_cbs_env = UnifiedSecEnv("cbs", cbs_factory=make_cbs_env)
    target_obs_list = []
    
    for _ in tqdm(range(num_samples)):
        obs, _ = target_cbs_env.reset()
        raw_obs = getattr(target_cbs_env, '_last_raw_cbs_obs', None) or getattr(target_cbs_env, '_last_raw_obs', None)
        if raw_obs is None:
            raw_obs = obs if isinstance(obs, dict) else {}
        
        unified_obs = translator._cbs_to_unified(raw_obs)
        normalized_obs = translator._normalize(unified_obs)
        target_obs_list.append(normalized_obs)
        
        # Use agent if available, else random
        if cbs_agent is not None:
            action, _ = cbs_agent.predict(obs, deterministic=False)
        else:
            action = target_cbs_env.action_space.sample()
        target_cbs_env.step(action)
    
    # Domain 3: Validation (CBS with Cyberwheel-like topology - evaluation)
    val_obs_list = []
    if use_3_domains:
        print("Collecting observations from Validation domain (CBS with Cyberwheel topology)...")
        try:
            # Use CBS environment with Cyberwheel-like topology
            # This uses CyberBattleCW10-v0 which builds CBS from Cyberwheel YAML
            original_env = os.environ.get("CBS_ENV", "CyberBattleChain-v0")
            os.environ["CBS_ENV"] = "CyberBattleCW10-v0"  # CBS with CW topology
            
            val_cbs_env = UnifiedSecEnv("cbs", cbs_factory=make_cbs_env)
            os.environ["CBS_ENV"] = original_env  # Restore
            
            for _ in tqdm(range(num_samples // 2)):  # Fewer samples for validation
                obs, _ = val_cbs_env.reset()
                raw_obs = getattr(val_cbs_env, '_last_raw_cbs_obs', None) or getattr(val_cbs_env, '_last_raw_obs', None)
                if raw_obs is None:
                    raw_obs = obs if isinstance(obs, dict) else {}
                
                unified_obs = translator._cbs_to_unified(raw_obs)
                normalized_obs = translator._normalize(unified_obs)
                val_obs_list.append(normalized_obs)
                
                action = val_cbs_env.action_space.sample()
                val_cbs_env.step(action)
        except Exception as e:
            print(f"Warning: Could not collect validation domain observations: {e}")
            print("Using target domain split for validation instead...")
            # Fallback: use a split of target domain for validation
            split_idx = len(target_obs_list) // 5  # Use 20% for validation
            val_obs_list = target_obs_list[:split_idx]
            target_obs_list = target_obs_list[split_idx:]
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        np.savez(save_path, 
                 source_obs=source_obs_list, 
                 target_obs=target_obs_list,
                 val_obs=val_obs_list)
        print(f"Saved observations to {save_path}")
        print(f"  Source: {len(source_obs_list)} samples")
        print(f"  Target: {len(target_obs_list)} samples")
        print(f"  Validation: {len(val_obs_list)} samples")
    
    return source_obs_list, target_obs_list, val_obs_list


def train_dapn_encoder(
    source_obs_list,
    target_obs_list,
    val_obs_list=None,
    feature_size=256,
    num_epochs=50,
    batch_size=64,
    learning_rate=0.001,
    device=None,
    save_path="artifacts/transfer_models/dapn_encoder.pt",
    use_3_domains=True
):
    """
    Train encoder with domain alignment (adversarial domain adaptation) only.
    Uses domain alignment to align features from different domains without prototypical networks.
    
    Domain setup:
    1. Source domain (Cyberwheel) - source features
    2. Target domain (Normal CyberBattleSim) - target features to align with source
    3. Validation domain (CBS with Cyberwheel topology) - evaluation only
    
    Args:
        source_obs_list: List of normalized source domain (Cyberwheel) observations
        target_obs_list: List of normalized target domain (Normal CBS) observations
        val_obs_list: List of normalized validation domain (CBS with CW topology) observations (optional)
        feature_size: Size of feature space
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        device: Device to train on
        save_path: Path to save trained encoder
        use_3_domains: Whether to use 3-domain setup (True) or 2-domain (False)
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    # Create translator with DAPN
    # Use shared encoder to follow original DAPN concept (single encoder for both domains)
    use_shared = os.environ.get("DAPN_USE_SHARED_ENCODER", "1") == "1"
    if use_shared:
        print("Using SINGLE shared encoder (follows original DAPN concept)")
    else:
        print("Using separate encoders for each domain")
    
    translator = DAPNObservationTranslator(
        use_dapn=True,
        feature_size=feature_size,
        device=device,
        use_adversarial=True  # Enable adversarial training
    )
    
    # Set to training mode
    if translator.use_shared_encoder:
        translator.shared_encoder.train()
    else:
        translator.cbs_encoder.train()
        translator.cw_encoder.train()
    translator.domain_adapter.train()
    
    # Handle domain availability
    if len(source_obs_list) == 0:
        print("Warning: No source domain (Cyberwheel) observations available.")
        print("Note: Source domain should be Cyberwheel for proper DAPN setup.")
    
    if len(target_obs_list) == 0:
        print("Warning: No target domain (CBS) observations available. Training source-only encoder.")
        if translator.domain_adapter:
            translator.domain_adapter = None
        use_3_domains = False
    
    if val_obs_list is None or len(val_obs_list) == 0:
        print("Warning: No validation domain observations. Using 2-domain setup.")
        use_3_domains = False
        val_obs_list = []
    
    print(f"\nDomain Alignment Training Setup:")
    print(f"  Source domain (Cyberwheel): {len(source_obs_list)} samples")
    print(f"  Target domain (Normal CyberBattleSim): {len(target_obs_list)} samples")
    print(f"  Validation domain (CBS with Cyberwheel topology): {len(val_obs_list)} samples")
    print(f"  Method: Adversarial Domain Adaptation (DANN)\n")
    
    # Create dataset and dataloader
    # Use a custom sampler to ensure batches always have both source and target domains
    dataset = ObservationDataset(source_obs_list, target_obs_list, val_obs_list)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Verify we have samples from both domains
    if len(source_obs_list) == 0 or len(target_obs_list) == 0:
        print("Error: Need samples from both source and target domains for domain alignment!")
        if len(source_obs_list) == 0:
            print("  Source domain (Cyberwheel) has 0 samples")
        if len(target_obs_list) == 0:
            print("  Target domain (CBS) has 0 samples")
        return translator
    
    # Optimizers
    if translator.use_shared_encoder:
        encoder_params = list(translator.shared_encoder.parameters())
    else:
        encoder_params = list(translator.cbs_encoder.parameters())
        if translator.cw_encoder is not None:
            encoder_params += list(translator.cw_encoder.parameters())
    
    optimizer_encoder = optim.Adam(encoder_params, lr=learning_rate)
    
    optimizer_adversarial = None
    if translator.domain_adapter is not None:
        # Use slower learning rate for discriminator to prevent it from becoming too good too fast
        # This allows the encoder time to align features before discriminator becomes perfect
        discriminator_lr = learning_rate * 0.1  # 10x slower than encoder
        optimizer_adversarial = optim.Adam(
            translator.domain_adapter.parameters(),
            lr=discriminator_lr
        )
        print(f"Discriminator learning rate: {discriminator_lr:.6f} (encoder: {learning_rate:.6f})")
    
    # Loss functions (only adversarial for domain alignment)
    mse_loss = nn.MSELoss()  # Only used as fallback if no domain adapter
    bce_loss = nn.BCELoss()  # For adversarial domain discriminator
    
    print(f"Training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        total_adv_loss = 0.0
        
        for batch_idx, (obs_batch, domain_labels) in enumerate(dataloader):
            obs_batch = obs_batch.to(device)
            domain_labels = domain_labels.to(device).float()
            
            # Split batch into domains (0=source, 1=target, 2=validation)
            batch_size = obs_batch.size(0)
            source_mask = domain_labels == 0
            target_mask = domain_labels == 1
            val_mask = domain_labels == 2
            
            # Encode observations
            # Source = Cyberwheel, Target = Normal CBS, Validation = CBS with CW topology
            if translator.use_shared_encoder:
                # Use single shared encoder for all domains
                source_features = translator.shared_encoder(obs_batch[source_mask]) if source_mask.any() else None
                target_features = translator.shared_encoder(obs_batch[target_mask]) if target_mask.any() else None
                val_features = translator.shared_encoder(obs_batch[val_mask]) if val_mask.any() else None
            else:
                # Use separate encoders
                source_features = translator.cw_encoder(obs_batch[source_mask]) if source_mask.any() and translator.cw_encoder else None
                target_features = translator.cbs_encoder(obs_batch[target_mask]) if target_mask.any() else None
                val_features = translator.cbs_encoder(obs_batch[val_mask]) if val_mask.any() else None
            
            # Domain alignment: Only use adversarial domain adaptation (no reconstruction loss)
            # The goal is to align source (Cyberwheel) and target (CBS) feature spaces
            # For adversarial training, we need to:
            # 1. First update adversarial network to maximize domain discrimination
            # 2. Then update encoders to minimize domain discrimination (confuse discriminator)
            
            # Initialize losses
            adv_loss = None
            adv_loss_for_encoder = None
            
            # Check if we have samples from both domains in this batch
            has_source = source_mask.any()
            has_target = target_mask.any()
            
            if has_source and has_target and translator.domain_adapter is not None:
                # Detach features for adversarial network update (to avoid gradient issues)
                # We'll use detached features for adversarial network, but original for encoder update
                all_features_detached = torch.cat([source_features.detach(), target_features.detach()], dim=0)
                all_features_for_encoder = torch.cat([source_features, target_features], dim=0)
                
                # Create domain labels: 0 for source (Cyberwheel), 1 for target (CBS)
                domain_targets = torch.cat([
                    torch.zeros(source_features.size(0), 1).to(device),
                    torch.ones(target_features.size(0), 1).to(device)
                ], dim=0)
                
                # Step 1: Update adversarial network to maximize domain discrimination
                if optimizer_adversarial is not None:
                    optimizer_adversarial.zero_grad()
                    domain_pred_adv = translator.domain_adapter(all_features_detached)
                    adv_loss = bce_loss(domain_pred_adv, domain_targets)
                    adv_loss.backward()
                    optimizer_adversarial.step()
                
                # Step 2: Compute adversarial loss for encoder (to confuse discriminator)
                # Use non-detached features so gradients flow to encoders
                # Note: Discriminator was just updated, so we need to recompute predictions
                domain_pred_encoder = translator.domain_adapter(all_features_for_encoder)
                adv_loss_for_encoder = bce_loss(domain_pred_encoder, domain_targets)
            
            # Total loss: Only adversarial (inverted for domain confusion)
            # We want encoders to produce features that confuse the domain discriminator
            if adv_loss_for_encoder is not None:
                # Invert adversarial loss: we want to confuse the discriminator
                # This aligns the feature spaces of source and target domains
                # Use the adversarial loss directly (minimize it = maximize confusion)
                # But we want to maximize confusion, so we minimize: -log(1 - pred) for correct labels
                # Actually, we want to minimize: -adv_loss_for_encoder (maximize adv_loss_for_encoder)
                # But that's unstable. Instead, use: 1.0 - adv_loss_for_encoder
                # This gives us a loss that decreases as the discriminator gets more confused
                total_batch_loss = 1.0 - adv_loss_for_encoder
                
                # Add a small regularization to prevent the loss from being exactly 1.0
                # This ensures gradients flow even when discriminator is very good
                if adv_loss_for_encoder.item() < 0.01:
                    # If discriminator is too good, add a small penalty to encourage feature alignment
                    total_batch_loss = total_batch_loss + 0.1 * mse_loss(
                        source_features.mean(dim=0), target_features.mean(dim=0)
                    )
            elif source_features is not None and target_features is not None:
                # Fallback: if no domain adapter, use a simple feature matching loss
                # Minimize distance between source and target feature distributions
                total_batch_loss = mse_loss(source_features.mean(dim=0, keepdim=False), 
                                           target_features.mean(dim=0, keepdim=False))
            else:
                # Skip this batch if no features available
                continue
            
            # Update encoders (loss should always require grad at this point)
            optimizer_encoder.zero_grad()
            total_batch_loss.backward()
            optimizer_encoder.step()
            
            total_loss += total_batch_loss.item()
            # Track both discriminator loss and encoder adversarial loss
            if adv_loss is not None:
                total_adv_loss += adv_loss.item()
            elif adv_loss_for_encoder is not None:
                # If discriminator wasn't updated this batch, use encoder loss for tracking
                total_adv_loss += adv_loss_for_encoder.item()
            else:
                total_adv_loss += 0.0
        
        avg_loss = total_loss / len(dataloader)
        avg_adv = total_adv_loss / len(dataloader)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}: "
                  f"Loss={avg_loss:.4f}, Adv={avg_adv:.4f}")
    
    # Save encoder
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    translator.save_encoder(save_path)
    print(f"Saved trained encoder to {save_path}")
    
    return translator


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train DAPN encoder for observation handling")
    parser.add_argument("--num-samples", type=int, default=1000, help="Number of samples per domain")
    parser.add_argument("--feature-size", type=int, default=256, help="Feature space size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--load-data", type=str, default=None, help="Path to load pre-collected observations")
    parser.add_argument("--save-data", type=str, default=None, help="Path to save collected observations")
    parser.add_argument("--save-encoder", type=str, 
                       default="artifacts/transfer_models/dapn_encoder.pt",
                       help="Path to save trained encoder")
    parser.add_argument("--cbs-only", action="store_true",
                       help="Train with CBS observations only (skip Cyberwheel)")
    parser.add_argument("--cw-agent", type=str, default=None,
                       help="Path to Cyberwheel agent checkpoint (uses agent if provided, else random)")
    parser.add_argument("--cbs-agent", type=str, default=None,
                       help="Path to CBS agent checkpoint (uses agent if provided, else random)")
    
    args = parser.parse_args()
    
    # Load or collect observations
    if args.load_data and os.path.exists(args.load_data):
        print(f"Loading observations from {args.load_data}")
        data = np.load(args.load_data, allow_pickle=True)
        # Support both old format (cbs_obs, cw_obs) and new format (source_obs, target_obs, val_obs)
        if 'source_obs' in data:
            source_obs_list = data['source_obs'].tolist()
            target_obs_list = data['target_obs'].tolist() if 'target_obs' in data else []
            val_obs_list = data['val_obs'].tolist() if 'val_obs' in data else []
        else:
            # Legacy format
            source_obs_list = data['cbs_obs'].tolist()
            target_obs_list = data['cw_obs'].tolist() if 'cw_obs' in data else []
            val_obs_list = []
    else:
        source_obs_list, target_obs_list, val_obs_list = collect_observations(
            num_samples=args.num_samples,
            save_path=args.save_data,
            cw_available=not args.cbs_only,  # Need Cyberwheel for source domain
            use_3_domains=not args.cbs_only,  # Use 3 domains if CW is available
            cw_agent_path=args.cw_agent,
            cbs_agent_path=args.cbs_agent
        )
    
    # Convert to tensors
    source_obs_list = [torch.FloatTensor(obs) for obs in source_obs_list]
    target_obs_list = [torch.FloatTensor(obs) for obs in target_obs_list]
    val_obs_list = [torch.FloatTensor(obs) for obs in val_obs_list] if val_obs_list else []
    
    if len(source_obs_list) == 0:
        print("Error: No source domain observations collected!")
        sys.exit(1)
    
    # Train encoder
    translator = train_dapn_encoder(
        source_obs_list,
        target_obs_list,
        val_obs_list=val_obs_list,
        feature_size=args.feature_size,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        save_path=args.save_encoder,
        use_3_domains=(len(val_obs_list) > 0 and len(target_obs_list) > 0)
    )
    
    print("Training complete!")

