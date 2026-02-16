"""
Train DAPN encoder with episodic training structure (following DAPN paper).
Implements:
- Episodic training from Ds (Nsc=20 classes, k=5 shots)
- Episodic training from Dd (Ndc=5 classes, k=5 shots)
- Prototypical network loss + Domain adversarial loss

Validation note:
- "Target FSL Acc" is computed using dataset.labels, which are synthetic
  (from clustering, situation_action, or action-based labels). So it measures
  encoder transfer on those constructed labels, NOT on a real downstream task.
- For CW → CBS transfer for policy, evaluate instead with:
  - reward improvement or success rate (e.g. run policy on CBS with this encoder),
  - or mapping consistency across domains.
  See evaluate_transfer.py / run_transfer_evaluation.py for policy-level evaluation.
"""

import os
import sys
import random
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "cyberwheel"))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from adapters.unified_env import UnifiedSecEnv
from adapters.dapn_observation_encoder import DAPNObservationTranslator, DAPNDomainAdapter
from adapters.unified_full_obs_preprocessor import UnifiedFullObsPreprocessor
from adapters.episodic_training import (
    create_episodic_dataloaders,
    CategoriesSampler,
    euclidean_metric,
    compute_prototypes,
    count_acc
)
from config.env_builders import make_cbs_env, make_cw_env


class GradientReversalFunction(torch.autograd.Function):
    """Gradient reversal layer for adversarial training."""

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


def grad_reverse(x, lambda_=1.0):
    return GradientReversalFunction.apply(x, lambda_)


class ConditionalDomainDiscriminator(nn.Module):
    """CDAN-style discriminator conditioned on class predictions."""

    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        hidden_size: int = 1024,
        conditioning: str = "randomized",
        conditioning_dim: int = 1024
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.conditioning = conditioning
        if conditioning not in ("outer", "randomized"):
            raise ValueError(f"Unknown conditioning mode: {conditioning}")
        if conditioning == "randomized":
            self.conditioner = RandomizedConditioning(
                f_dim=feature_dim,
                g_dim=num_classes,
                output_dim=conditioning_dim
            )
            input_dim = conditioning_dim
        else:
            self.conditioner = None
            input_dim = feature_dim * num_classes
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, features: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
        # probs: (B, C), features: (B, F)
        if self.conditioning == "randomized":
            conditioned = self.conditioner(features, probs)
            return self.net(conditioned)
        batch_size = features.size(0)
        op_out = torch.bmm(probs.unsqueeze(2), features.unsqueeze(1))  # (B, C, F)
        return self.net(op_out.view(batch_size, -1))


class RandomizedConditioning(nn.Module):
    """Approximate multilinear map for CDAN-style conditioning."""

    def __init__(self, f_dim: int, g_dim: int, output_dim: int = 1024):
        super().__init__()
        self.output_dim = output_dim
        self.register_buffer("Rf", torch.randn(output_dim, f_dim))
        self.register_buffer("Rg", torch.randn(output_dim, g_dim))

    def forward(self, f: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        f_proj = F.linear(f, self.Rf)
        g_proj = F.linear(g, self.Rg)
        return (f_proj * g_proj) / (self.output_dim ** 0.5)


def pad_probs(probs: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Pad probabilities to a fixed number of classes."""
    if probs.size(1) == num_classes:
        return probs
    if probs.size(1) > num_classes:
        return probs[:, :num_classes]
    pad = torch.zeros(probs.size(0), num_classes - probs.size(1), device=probs.device, dtype=probs.dtype)
    return torch.cat([probs, pad], dim=1)


def cdan_loss(
    discriminator: nn.Module,
    features: torch.Tensor,
    probs: torch.Tensor,
    domain_labels: torch.Tensor,
    entropy: torch.Tensor,
    n_source: int
) -> torch.Tensor:
    """CDAN loss with entropy conditioning."""
    domain_pred = discriminator(features, probs)
    bce = nn.BCELoss(reduction="none")(domain_pred, domain_labels)
    # Entropy-aware reweighting (CDAN+E)
    entropy = 1.0 + torch.exp(-entropy)
    source_mask = torch.zeros_like(entropy)
    target_mask = torch.zeros_like(entropy)
    source_mask[:n_source] = 1.0
    target_mask[n_source:] = 1.0
    source_weight = entropy * source_mask
    target_weight = entropy * target_mask
    weight = source_weight / (source_weight.sum() + 1e-6) + target_weight / (target_weight.sum() + 1e-6)
    return (weight * bce).sum() / (weight.sum() + 1e-6)


def evaluate_fsl_target(
    encoder,
    dataset,
    n_way: int,
    k: int,
    query: int,
    episodes: int,
    device: torch.device
):
    """Evaluate few-shot accuracy on target domain with episodic sampling."""
    sampler = CategoriesSampler(labels=dataset.labels, n_batch=episodes, n_cls=n_way, n_per=k + query)
    accs = []
    for batch_indices in sampler:
        batch_indices_np = batch_indices.numpy()
        batch_obs = torch.stack([dataset[int(idx)][0] for idx in batch_indices_np]).to(device)
        batch_labels = torch.LongTensor(dataset.labels[batch_indices_np]).to(device)
        
        n_per = k + query
        batch_obs_reshaped = batch_obs.reshape(n_per, n_way, -1)
        batch_labels_reshaped = batch_labels.reshape(n_per, n_way)
        
        support_obs = batch_obs_reshaped[:k].reshape(-1, batch_obs.shape[-1])
        query_obs = batch_obs_reshaped[k:].reshape(-1, batch_obs.shape[-1])
        support_labels = batch_labels_reshaped[:k].reshape(-1)
        query_labels = batch_labels_reshaped[k:].reshape(-1)
        
        # Encode and normalize
        features = encoder(support_obs)
        support_features = F.normalize(features, p=2, dim=1)
        query_features = F.normalize(encoder(query_obs), p=2, dim=1)
        
        unique_classes = torch.unique(support_labels)
        label_map = {orig.item(): i for i, orig in enumerate(unique_classes)}
        mapped_support_labels = torch.tensor(
            [label_map[label.item()] for label in support_labels],
            dtype=torch.long, device=device
        )
        
        prototypes = []
        for class_idx in range(len(unique_classes)):
            mask = mapped_support_labels == class_idx
            if mask.sum() == 0:
                continue
            prototypes.append(support_features[mask].mean(dim=0))
        
        if len(prototypes) == 0:
            continue
        
        prototypes = torch.stack(prototypes)
        mapped_query_labels = torch.tensor(
            [label_map[label.item()] if label.item() in label_map else -1 for label in query_labels],
            dtype=torch.long, device=device
        )
        valid_mask = mapped_query_labels >= 0
        if valid_mask.sum() == 0:
            continue
        
        query_features = query_features[valid_mask]
        mapped_query_labels = mapped_query_labels[valid_mask]
        
        logits = euclidean_metric(query_features, prototypes)
        accs.append(count_acc(logits, mapped_query_labels))
    
    if len(accs) == 0:
        return 0.0, 0.0
    accs = np.array(accs, dtype=np.float32)
    return float(accs.mean()), float(accs.std())

def train_dapn_encoder_episodic(
    source_obs_list,
    target_obs_list,
    val_obs_list=None,
    feature_size=256,
    num_iterations=10000,
    n_sc=20,  # Number of classes for Ds episodes
    n_dc=5,   # Number of classes for Dd episodes
    k=5,      # Shots per class
    query=15, # Query samples per class
    learning_rate=0.001,
    device=None,
    save_path="artifacts/transfer_models/dapn_encoder_episodic.pt",
    test_interval=100,
    snapshot_interval=5000,
    eval_episodes=1000,
    fsl_loss_weight=2.0,
):
    """
    Train DAPN encoder with episodic training following the paper.
    
    Args:
        source_obs_list: List of source domain (Ds) observations
        target_obs_list: List of target domain (Dd) observations
        val_obs_list: Optional validation observations
        feature_size: Size of feature space
        num_iterations: Number of training iterations
        n_sc: Number of classes for source domain episodes (Nsc)
        n_dc: Number of classes for target domain episodes (Ndc)
        k: Number of support samples per class
        query: Number of query samples per class
        learning_rate: Learning rate
        device: Device to train on
        save_path: Path to save trained encoder
        test_interval: Interval for validation testing
        snapshot_interval: Interval for saving snapshots
        eval_episodes: Number of target-domain evaluation episodes
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    # Create preprocessor for full observations (512D unified representation)
    preprocessor = UnifiedFullObsPreprocessor(unified_dim=512)
    
    # Create translator with DAPN
    use_shared = os.environ.get("DAPN_USE_SHARED_ENCODER", "1") == "1"
    translator = DAPNObservationTranslator(
        use_dapn=True,
        feature_size=feature_size,
        input_dim=512,  # Use 512D for full observations instead of 8D
        device=device,
        use_adversarial=True
    )
    
    # Set to training mode
    if translator.use_shared_encoder:
        translator.shared_encoder.train()
    else:
        translator.cbs_encoder.train()
        translator.cw_encoder.train()
    translator.domain_adapter.train()
    
    print(f"\n{'='*80}")
    print("EPISODIC TRAINING SETUP")
    print(f"{'='*80}")
    print(f"Source domain (Ds): {len(source_obs_list)} samples")
    print(f"Target domain (Dd): {len(target_obs_list)} samples")
    print(f"Episodic parameters:")
    print(f"  - Nsc (Ds classes): {n_sc}")
    print(f"  - Ndc (Dd classes): {n_dc}")
    print(f"  - k (shots): {k}")
    print(f"  - Query samples: {query}")
    
    # Get actions if available (for action-based clustering)
    source_actions = getattr(train_dapn_encoder_episodic, '_source_actions', None)
    target_actions = getattr(train_dapn_encoder_episodic, '_target_actions', None)
    
    # Create episodic dataloaders (pass preprocessor to handle raw observations)
    source_sampler, target_sampler, source_dataset, target_dataset, source_labels, target_labels = \
        create_episodic_dataloaders(
            source_obs=source_obs_list,
            target_obs=target_obs_list,
            source_actions=source_actions,
            target_actions=target_actions,
            n_sc=n_sc,
            n_dc=n_dc,
            k=k,
            query=query,
            n_batch_ds=1000,  # Generate many episodes
            n_batch_dd=1000,
            device=device,
            preprocessor=preprocessor,
            label_mode=getattr(train_dapn_encoder_episodic, "_label_mode", "cluster"),
            label_bins=getattr(train_dapn_encoder_episodic, "_label_bins", None)
        )
    
    # Create iterators
    source_iter = iter(source_sampler)
    target_iter = iter(target_sampler)
    
    # Optimizers
    if translator.use_shared_encoder:
        encoder_params = list(translator.shared_encoder.parameters())
    else:
        encoder_params = list(translator.cbs_encoder.parameters())
        if translator.cw_encoder is not None:
            encoder_params += list(translator.cw_encoder.parameters())
    
    # Learnable uncertainty weights for multi-loss optimization
    # [Lps, Lpt, Ldc, Lrec]  (Lds is discriminator-only; no encoder loss)
    loss_log_vars = nn.Parameter(torch.zeros(4, device=device))
    encoder_params += [loss_log_vars]
    
    # Use weight decay for regularization (reduced to allow more learning)
    optimizer_encoder = optim.Adam(encoder_params, lr=learning_rate, weight_decay=1e-5)
    
    # Learning rate scheduler for encoder (less aggressive, but still reduces LR)
    scheduler_encoder = optim.lr_scheduler.StepLR(optimizer_encoder, step_size=num_iterations//2, gamma=0.8)
    
    # Discriminators for full DAPN:
    # - Ldc: domain confusion (CDAN+E) after embedding
    # - Lds: domain discrimination before embedding
    discriminator_lr = learning_rate * 0.1
    num_classes_cond = max(n_sc, n_dc)
    cond_method = os.environ.get("DAPN_COND_METHOD", "randomized")
    domain_confusion_disc = ConditionalDomainDiscriminator(
        feature_dim=feature_size,
        num_classes=num_classes_cond,
        conditioning=cond_method
    ).to(device)
    domain_discriminator = DAPNDomainAdapter(feature_dim=feature_size, method="DANN").to(device)
    
    optimizer_confusion = optim.Adam(
        domain_confusion_disc.parameters(),
        lr=discriminator_lr,
        weight_decay=1e-5
    )
    optimizer_discriminator = optim.Adam(
        domain_discriminator.parameters(),
        lr=discriminator_lr,
        weight_decay=1e-5
    )
    
    scheduler_confusion = optim.lr_scheduler.StepLR(
        optimizer_confusion, step_size=num_iterations // 3, gamma=0.5
    )
    scheduler_discriminator = optim.lr_scheduler.StepLR(
        optimizer_discriminator, step_size=num_iterations // 3, gamma=0.5
    )
    
    # Loss functions
    ce_loss = nn.CrossEntropyLoss()
    bce_loss = nn.BCELoss()
    mse_loss = nn.MSELoss()
    
    print(f"\n{'='*80}")
    print("STARTING EPISODIC TRAINING")
    print(f"{'='*80}")
    print("Progress: Iter | Lps | Lpt | Ldc | Lds | Lrec | Total | Acc_src | Acc_tgt | Val=Target FSL Acc (synthetic labels)")
    print(f"{'='*80}")

    best_acc = 0.0
    ave_acc = Averager()
    
    from tqdm import tqdm
    
    for iteration in tqdm(range(num_iterations), desc="Training", unit="iter"):
        # Get next episode from source domain (Ds)
        try:
            source_indices = next(source_iter)
        except StopIteration:
            source_iter = iter(source_sampler)
            source_indices = next(source_iter)
        
        # Get next episode from target domain (Dd)
        try:
            target_indices = next(target_iter)
        except StopIteration:
            target_iter = iter(target_sampler)
            target_indices = next(target_iter)
        
        # Prepare source episode data (use dataset's __getitem__ to handle raw observations)
        source_indices_np = source_indices.numpy()
        source_batch_obs = torch.stack([
            source_dataset[int(idx)][0]  # __getitem__ returns (obs_tensor, label)
            for idx in source_indices_np
        ]).to(device)
        source_batch_labels = torch.LongTensor(source_dataset.labels[source_indices_np]).to(device)
        
        # Split into support and query sets
        # CategoriesSampler returns samples ordered by shot position:
        # [shot0_class0, shot0_class1, ..., shot0_classN, shot1_class0, shot1_class1, ..., shot1_classN, ...]
        # So we need to reshape: (n_per, n_sc) where n_per = k + query
        expected_total = n_sc * (k + query)
        
        # Adjust n_sc if we don't have enough samples
        actual_n_sc = n_sc
        if len(source_batch_obs) < expected_total:
            # Recalculate based on what we have
            actual_n_sc = len(source_batch_obs) // (k + query)
            if actual_n_sc < 1:
                actual_n_sc = 1
            print(f"Warning: Got {len(source_batch_obs)} samples, expected {expected_total}. "
                  f"Using {actual_n_sc} classes instead of {n_sc}. "
                  f"(Only {actual_n_sc} classes have >= {k + query} samples; use smaller --n-sc or more data.)")
        
        # Reshape to (n_per, actual_n_sc) where n_per = k + query
        n_per = k + query
        if len(source_batch_obs) < n_per * actual_n_sc:
            # Not enough samples, skip
            if iteration % 100 == 0:
                print(f"Skipping iteration {iteration}: insufficient samples ({len(source_batch_obs)} < {n_per * actual_n_sc})")
            continue
        
        # Reshape: (n_per, actual_n_sc) - each row is one shot position across all classes
        batch_obs_reshaped = source_batch_obs[:n_per * actual_n_sc].reshape(n_per, actual_n_sc, -1)
        batch_labels_reshaped = source_batch_labels[:n_per * actual_n_sc].reshape(n_per, actual_n_sc)
        
        # Support: first k rows (shots 0 to k-1), Query: remaining rows (shots k to n_per-1)
        source_support_obs = batch_obs_reshaped[:k].reshape(-1, source_batch_obs.shape[-1])  # (k * actual_n_sc, obs_dim)
        source_query_obs = batch_obs_reshaped[k:].reshape(-1, source_batch_obs.shape[-1])   # (query * actual_n_sc, obs_dim)
        source_support_labels = batch_labels_reshaped[:k].reshape(-1)  # (k * actual_n_sc,)
        source_query_labels = batch_labels_reshaped[k:].reshape(-1)    # (query * actual_n_sc,)
        
        # Prepare target episode data (use dataset's __getitem__ to handle raw observations)
        target_indices_np = target_indices.numpy()
        target_batch_obs = torch.stack([
            target_dataset[int(idx)][0]  # __getitem__ returns (obs_tensor, label)
            for idx in target_indices_np
        ]).to(device)
        target_batch_labels = torch.LongTensor(target_dataset.labels[target_indices_np]).to(device)
        
        # Split target into support/query sets (same structure as source)
        expected_total_t = n_dc * (k + query)
        actual_n_dc = n_dc
        if len(target_batch_obs) < expected_total_t:
            actual_n_dc = len(target_batch_obs) // (k + query)
            if actual_n_dc < 1:
                actual_n_dc = 1
            print(f"Warning: Got {len(target_batch_obs)} target samples, expected {expected_total_t}. "
                  f"Using {actual_n_dc} classes instead of {n_dc}. "
                  f"(Only {actual_n_dc} classes have >= {k + query} samples; use smaller --n-dc or more data.)")
        
        if len(target_batch_obs) < n_per * actual_n_dc:
            if iteration % 100 == 0:
                print(f"Skipping iteration {iteration}: insufficient target samples ({len(target_batch_obs)} < {n_per * actual_n_dc})")
            continue
        
        target_obs_reshaped = target_batch_obs[:n_per * actual_n_dc].reshape(n_per, actual_n_dc, -1)
        target_labels_reshaped = target_batch_labels[:n_per * actual_n_dc].reshape(n_per, actual_n_dc)
        
        target_support_obs = target_obs_reshaped[:k].reshape(-1, target_batch_obs.shape[-1])
        target_query_obs = target_obs_reshaped[k:].reshape(-1, target_batch_obs.shape[-1])
        target_support_labels = target_labels_reshaped[:k].reshape(-1)
        target_query_labels = target_labels_reshaped[k:].reshape(-1)
        
        # Encode observations (full DAPN: autoencoder + attention)
        if translator.use_shared_encoder:
            src_sup_emb, src_sup_pre, src_sup_recon, _ = translator.shared_encoder(source_support_obs, return_all=True)
            src_q_emb, src_q_pre, src_q_recon, _ = translator.shared_encoder(source_query_obs, return_all=True)
            tgt_sup_emb, tgt_sup_pre, tgt_sup_recon, _ = translator.shared_encoder(target_support_obs, return_all=True)
            tgt_q_emb, tgt_q_pre, tgt_q_recon, _ = translator.shared_encoder(target_query_obs, return_all=True)
        else:
            src_sup_emb, src_sup_pre, src_sup_recon, _ = translator.cw_encoder(source_support_obs, return_all=True)
            src_q_emb, src_q_pre, src_q_recon, _ = translator.cw_encoder(source_query_obs, return_all=True)
            tgt_sup_emb, tgt_sup_pre, tgt_sup_recon, _ = translator.cbs_encoder(target_support_obs, return_all=True)
            tgt_q_emb, tgt_q_pre, tgt_q_recon, _ = translator.cbs_encoder(target_query_obs, return_all=True)
        
        # Normalize embedded features for prototypical networks
        source_support_features = F.normalize(src_sup_emb, p=2, dim=1)
        source_query_features = F.normalize(src_q_emb, p=2, dim=1)
        target_support_features = F.normalize(tgt_sup_emb, p=2, dim=1)
        target_query_features = F.normalize(tgt_q_emb, p=2, dim=1)
        
        # Compute prototypes from support set using actual labels
        # Map support labels to 0..actual_n_sc-1
        unique_support_classes = torch.unique(source_support_labels)
        support_label_map = {orig_label.item(): new_label for new_label, orig_label in enumerate(unique_support_classes)}
        mapped_support_labels = torch.tensor([support_label_map[label.item()] for label in source_support_labels], 
                                            dtype=torch.long, device=device)
        
        # Compute prototypes by averaging features for each class
        actual_n_sc = len(unique_support_classes)
        source_prototypes = []
        for class_idx in range(actual_n_sc):
            class_mask = (mapped_support_labels == class_idx)
            if class_mask.sum() == 0:
                # Skip if no samples for this class
                continue
            class_features = source_support_features[class_mask]
            prototype = class_features.mean(dim=0)
            source_prototypes.append(prototype)
        
        if len(source_prototypes) == 0:
            if iteration % 100 == 0:
                print(f"Skipping iteration {iteration}: no valid prototypes")
            continue
        
        source_prototypes = torch.stack(source_prototypes)  # (actual_n_sc, feature_dim)
        actual_n_sc = len(source_prototypes)  # Update to actual number of prototypes
        
        # Map query labels to match prototype indices (same mapping as support)
        source_query_labels_mapped = torch.tensor([
            support_label_map[label.item()] if label.item() in support_label_map else -1
            for label in source_query_labels
        ], dtype=torch.long, device=device)
        
        # Filter out any query samples with labels not in support set (shouldn't happen, but safety check)
        valid_query_mask = source_query_labels_mapped >= 0
        if valid_query_mask.sum() < len(source_query_labels_mapped):
            if iteration % 100 == 0:
                print(f"Warning: {len(source_query_labels_mapped) - valid_query_mask.sum()} query samples have labels not in support set")
            source_query_features = source_query_features[valid_query_mask]
            source_query_labels_mapped = source_query_labels_mapped[valid_query_mask]
        
        if len(source_query_features) == 0:
            if iteration % 100 == 0:
                print(f"Skipping iteration {iteration}: no valid query samples")
            continue
        
        # Compute prototypical network loss (FSL loss)
        query_logits = euclidean_metric(source_query_features, source_prototypes)
        
        # Temperature: use 1.0 so logits stay informative (T>1 flattens softmax and weakens FSL gradient)
        temperature = 1.0
        query_logits = query_logits / temperature
        
        # Debug: Check if features are collapsing
        if iteration % 500 == 0:
            feature_std = source_query_features.std().item()
            prototype_std = source_prototypes.std().item()
            logits_std = query_logits.std().item()
            print(f"  [Debug] Feature std: {feature_std:.6f}, Prototype std: {prototype_std:.6f}, Logits std: {logits_std:.6f}, Temp: {temperature:.3f}")
            if feature_std < 0.01:
                print(f"  [Warning] Features are collapsing (std={feature_std:.6f})!")
        
        fsl_loss_source = ce_loss(query_logits, source_query_labels_mapped)
        fsl_acc_source = count_acc(query_logits, source_query_labels_mapped)
        
        # Target prototypes and FSL loss (Lpt)
        unique_target_classes = torch.unique(target_support_labels)
        target_label_map = {orig_label.item(): new_label for new_label, orig_label in enumerate(unique_target_classes)}
        mapped_target_support_labels = torch.tensor(
            [target_label_map[label.item()] for label in target_support_labels],
            dtype=torch.long, device=device
        )
        
        target_prototypes = []
        for class_idx in range(len(unique_target_classes)):
            class_mask = (mapped_target_support_labels == class_idx)
            if class_mask.sum() == 0:
                continue
            class_features = target_support_features[class_mask]
            target_prototypes.append(class_features.mean(dim=0))
        
        if len(target_prototypes) == 0:
            if iteration % 100 == 0:
                print(f"Skipping iteration {iteration}: no valid target prototypes")
            continue
        
        target_prototypes = torch.stack(target_prototypes)
        target_query_labels_mapped = torch.tensor(
            [target_label_map[label.item()] if label.item() in target_label_map else -1
             for label in target_query_labels],
            dtype=torch.long, device=device
        )
        valid_tgt_mask = target_query_labels_mapped >= 0
        if valid_tgt_mask.sum() < len(target_query_labels_mapped):
            target_query_features = target_query_features[valid_tgt_mask]
            target_query_labels_mapped = target_query_labels_mapped[valid_tgt_mask]
        
        if len(target_query_features) == 0:
            if iteration % 100 == 0:
                print(f"Skipping iteration {iteration}: no valid target query samples")
            continue
        
        target_query_logits = euclidean_metric(target_query_features, target_prototypes)
        fsl_loss_target = ce_loss(target_query_logits, target_query_labels_mapped)
        fsl_acc_target = count_acc(target_query_logits, target_query_labels_mapped)
        
        fsl_loss = fsl_loss_source + fsl_loss_target
        fsl_acc = 0.5 * (fsl_acc_source + fsl_acc_target)
        
        # Domain confusion loss (Ldc) with CDAN+E after embedding
        source_all_features = torch.cat([source_support_features, source_query_features], dim=0)
        target_all_features = torch.cat([target_support_features, target_query_features], dim=0)
        source_all_logits = euclidean_metric(source_all_features, source_prototypes)
        target_all_logits = euclidean_metric(target_all_features, target_prototypes)
        source_probs = F.softmax(source_all_logits, dim=1)
        target_probs = F.softmax(target_all_logits, dim=1)
        
        source_probs = pad_probs(source_probs, num_classes_cond)
        target_probs = pad_probs(target_probs, num_classes_cond)
        
        all_features = torch.cat([source_all_features, target_all_features], dim=0)
        all_probs = torch.cat([source_probs, target_probs], dim=0)
        domain_labels = torch.cat([
            torch.zeros(source_all_features.size(0), 1).to(device),
            torch.ones(target_all_features.size(0), 1).to(device)
        ], dim=0)
        
        entropy = -torch.sum(all_probs * torch.log(all_probs + 1e-6), dim=1, keepdim=True)
        n_source = source_all_features.size(0)
        
        # Update domain confusion discriminator (CDAN)
        optimizer_confusion.zero_grad()
        cdan_disc_loss = cdan_loss(
            domain_confusion_disc,
            all_features.detach(),
            all_probs.detach(),
            domain_labels,
            entropy.detach(),
            n_source
        )
        cdan_disc_loss.backward()
        optimizer_confusion.step()
        
        # Encoder confusion loss with gradient reversal
        grl_features = grad_reverse(all_features, lambda_=1.0)
        cdan_enc_loss = cdan_loss(
            domain_confusion_disc,
            grl_features,
            all_probs,
            domain_labels,
            entropy,
            n_source
        )
        
        # Domain discrimination loss (Lds) before embedding
        pre_features_all = torch.cat([src_sup_pre, src_q_pre, tgt_sup_pre, tgt_q_pre], dim=0)
        optimizer_discriminator.zero_grad()
        lds_disc_loss = bce_loss(domain_discriminator(pre_features_all.detach()), domain_labels)
        lds_disc_loss.backward()
        optimizer_discriminator.step()
        
        # Lds is discriminator-only (do NOT confuse encoder here)
        lds_enc_loss = None
        
        # Autoencoder reconstruction loss
        recon_loss = (
            mse_loss(src_sup_recon, source_support_obs) +
            mse_loss(src_q_recon, source_query_obs) +
            mse_loss(tgt_sup_recon, target_support_obs) +
            mse_loss(tgt_q_recon, target_query_obs)
        ) / 4.0
        
        # Adaptive re-weighting of losses; boost FSL so classification signal is not drowned by domain/recon
        # Optionally ramp down domain confusion early so encoder learns discriminative features first
        ramp = min(1.0, iteration / max(1, num_iterations // 5))
        cdan_weighted = cdan_enc_loss * ramp
        losses = [fsl_loss_weight * fsl_loss_source, fsl_loss_weight * fsl_loss_target, cdan_weighted, recon_loss]
        total_loss = 0.0
        for i, loss_val in enumerate(losses):
            total_loss = total_loss + 0.5 * loss_log_vars[i] + torch.exp(-loss_log_vars[i]) * loss_val
        
        # Update encoder with gradient clipping for stability
        optimizer_encoder.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder_params, max_norm=1.0)  # Clip gradients
        optimizer_encoder.step()
        
        # Update learning rate schedulers
        if (iteration + 1) % 10 == 0:  # Step scheduler every 10 iterations
            scheduler_encoder.step()
            scheduler_confusion.step()
            scheduler_discriminator.step()
        
        # Logging: every 10 iters (and always at 0) so accuracy and losses are visible
        if iteration % 10 == 0:
            current_lr = optimizer_encoder.param_groups[0]['lr']
            acc_src_pct = fsl_acc_source * 100
            acc_tgt_pct = fsl_acc_target * 100
            print(
                f"Iter {iteration:5d} | "
                f"Lps: {fsl_loss_source.item():.4f} | Lpt: {fsl_loss_target.item():.4f} | "
                f"Ldc: {cdan_enc_loss.item():.4f} | Lds: {lds_disc_loss.item():.4f} | "
                f"Lrec: {recon_loss.item():.4f} | Total: {total_loss.item():.4f} | "
                f"Acc_src: {acc_src_pct:.1f}% | Acc_tgt: {acc_tgt_pct:.1f}% | LR: {current_lr:.6f}"
            )
        
        # Validation: few-shot accuracy on target using *synthetic* labels (dataset.labels).
        # This is a proxy for encoder transfer quality, not a real downstream task metric.
        # For policy transfer (CW→CBS), use reward/success-rate evaluation (e.g. evaluate_transfer.py).
        if iteration % test_interval == test_interval - 1:
            eval_encoder = translator.shared_encoder if translator.use_shared_encoder else translator.cbs_encoder
            eval_encoder.eval()
            mean_acc, std_acc = evaluate_fsl_target(
                eval_encoder,
                target_dataset,
                n_way=n_dc,
                k=k,
                query=query,
                episodes=eval_episodes,
                device=device
            )
            ave_acc.add(mean_acc)
            best_acc = max(best_acc, mean_acc)
            print(
                f"\n  >> VALIDATION iter {iteration}: Target FSL Acc (synthetic labels) = {mean_acc*100:.2f}% ± {std_acc*100:.2f}% "
                f"(episodes={eval_episodes}) | Running avg: {ave_acc.item()*100:.2f}% | Best: {best_acc*100:.2f}%"
            )
            eval_encoder.train()
        
        # Save snapshot
        if iteration % snapshot_interval == 0:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            translator.save_encoder(save_path.replace('.pt', f'_iter_{iteration:05d}.pt'))
    
    # Save final model
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    translator.save_encoder(save_path)
    print(f"\n{'='*80}")
    print("TRAINING COMPLETE – SUMMARY")
    print(f"{'='*80}")
    print(f"  Best target FSL accuracy (synthetic labels): {best_acc*100:.2f}%")
    print(f"  Running avg target accuracy: {ave_acc.item()*100:.2f}%")
    print(f"  Encoder saved to: {save_path}")
    print(f"  Note: For real CW→CBS transfer, evaluate policy reward/success rate (e.g. evaluate_transfer.py).")
    print(f"{'='*80}")
    
    return translator


class Averager:
    """Running average calculator."""
    def __init__(self):
        self.n = 0
        self.v = 0
    
    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1
    
    def item(self):
        return self.v


def _load_cw_policy(cw_policy_path):
    """Load CW policy: PPO .zip or Cyberwheel .pt. Returns (policy, 'ppo'|'cyberwheel', device_or_none)."""
    if not cw_policy_path or not os.path.exists(cw_policy_path):
        return None, None, None
    try:
        if cw_policy_path.endswith(".pt"):
            from collect_episodic_data_with_policies import load_cyberwheel_policy
            policy, device = load_cyberwheel_policy(cw_policy_path)
            return (policy, "cyberwheel", device) if policy is not None else (None, None, None)
        else:
            from collect_episodic_data_with_policies import load_ppo_policy
            policy = load_ppo_policy(cw_policy_path)
            return (policy, "ppo", None) if policy is not None else (None, None, None)
    except Exception as e:
        print(f"  Warning: could not load CW policy from {cw_policy_path}: {e}")
        return None, None, None


def _load_cbs_policy(cbs_policy_path):
    """Load CBS PPO policy. Returns policy or None."""
    if not cbs_policy_path or not os.path.exists(cbs_policy_path):
        return None
    try:
        from collect_episodic_data_with_policies import load_ppo_policy
        return load_ppo_policy(cbs_policy_path)
    except Exception as e:
        print(f"  Warning: could not load CBS policy from {cbs_policy_path}: {e}")
        return None


def _cw_policy_action(cw_policy, cw_policy_type, cw_device, obs, raw, deterministic=False):
    """Get action from CW policy given env obs and raw obs."""
    if cw_policy_type == "ppo":
        # Env returns 8D unified obs; use it (or convert raw to 8D if policy expects 8D and we have raw)
        policy_obs = obs.get("obs", obs) if isinstance(obs, dict) else obs
        policy_obs = np.asarray(policy_obs, dtype=np.float32)
        if policy_obs.size != 8 and raw is not None:
            from adapters.observation_translator import ObservationTranslator
            policy_obs = ObservationTranslator().from_cw(np.asarray(raw, dtype=np.float32))
        action, _ = cw_policy.predict(policy_obs, deterministic=deterministic)
        return int(action)
    else:
        import torch
        obs_t = torch.from_numpy(np.asarray(raw, dtype=np.float32)).float().unsqueeze(0).to(cw_device)
        with torch.no_grad():
            action_t, _, _, _ = cw_policy.get_action_and_value(obs_t)
        return int(action_t.cpu().numpy()[0])


def _cbs_policy_action(cbs_policy, obs, deterministic=False):
    """Get action from CBS policy given env obs."""
    policy_obs_space = getattr(cbs_policy, "observation_space", None)
    if policy_obs_space is not None and hasattr(policy_obs_space, "spaces"):
        policy_obs = obs if isinstance(obs, dict) else {"obs": np.asarray(obs, dtype=np.float32), "mask": np.ones(7, dtype=np.float32)}
    else:
        policy_obs = obs.get("obs", obs) if isinstance(obs, dict) else obs
        if not isinstance(policy_obs, np.ndarray):
            policy_obs = np.array(policy_obs, dtype=np.float32)
    action, _ = cbs_policy.predict(policy_obs, deterministic=deterministic)
    return int(action)


def collect_observations_episodic(
    num_samples=1000,
    save_path=None,
    val_fraction=0.2,
    seed=None,
    cw_policy_path=None,
    cbs_policy_path=None,
    max_steps_per_episode=200,
    deterministic_policy=False,
):
    """
    Collect raw observations from Cyberwheel (source) and CyberBattleSim (target).
    Uses trained policies per env when paths are provided; otherwise random actions.
    Saves with source=CW, target=CBS.
    Returns: (source_obs_list, target_obs_list, val_obs_list, source_actions_list, target_actions_list).
    """
    from config.env_builders import make_cbs_env, make_cw_env

    print("Collecting observations (source=Cyberwheel, target=CyberBattleSim)...")
    if cw_policy_path or cbs_policy_path:
        print("  Using policy per env: CW={}, CBS={}".format(cw_policy_path or "random", cbs_policy_path or "random"))
    if deterministic_policy and (cw_policy_path or cbs_policy_path):
        print("  Policy mode: deterministic (greedy action)")
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # ---- Source (Cyberwheel) ----
    source_obs_list = []
    source_actions_list = []
    cw_policy, cw_ptype, cw_device = _load_cw_policy(cw_policy_path)
    try:
        cw_env = UnifiedSecEnv("cw", cw_factory=make_cw_env)
        if cw_policy is not None:
            # Policy rollout: run episodes until we have at least num_samples
            pbar = tqdm(desc="CW (source, policy)", unit="samples")
            while len(source_obs_list) < num_samples:
                obs, _ = cw_env.reset()
                done, truncated = False, False
                step = 0
                while not (done or truncated) and step < max_steps_per_episode and len(source_obs_list) < num_samples:
                    raw = getattr(cw_env, "_last_raw_obs", None) or (obs if isinstance(obs, np.ndarray) else np.array([], dtype=np.float32))
                    if not isinstance(raw, np.ndarray):
                        raw = np.array([], dtype=np.float32)
                    source_obs_list.append(raw)
                    action = _cw_policy_action(cw_policy, cw_ptype, cw_device, obs, raw, deterministic=deterministic_policy)
                    source_actions_list.append(action)
                    obs, _, done, truncated, _ = cw_env.step(action)
                    step += 1
                    pbar.update(1)
            pbar.close()
        else:
            for i in tqdm(range(num_samples), desc="CW (source, random)"):
                obs, _ = cw_env.reset()
                raw = getattr(cw_env, "_last_raw_obs", None)
                if raw is None:
                    raw = obs if isinstance(obs, np.ndarray) else np.array([], dtype=np.float32)
                source_obs_list.append(raw)
                action = cw_env.action_space.sample()
                source_actions_list.append(int(action))
                cw_env.step(action)
        print(f"  Collected {len(source_obs_list)} source (CW) samples.")
    except Exception as e:
        print(f"  Warning: could not collect Cyberwheel obs: {e}")
        source_obs_list = []
        source_actions_list = []

    # ---- Target (CyberBattleSim) ----
    target_obs_list = []
    target_actions_list = []
    cbs_policy = _load_cbs_policy(cbs_policy_path)
    try:
        cbs_env = UnifiedSecEnv("cbs", cbs_factory=make_cbs_env)
        if cbs_policy is not None:
            pbar = tqdm(desc="CBS (target, policy)", unit="samples")
            while len(target_obs_list) < num_samples:
                obs, _ = cbs_env.reset()
                done, truncated = False, False
                step = 0
                while not (done or truncated) and step < max_steps_per_episode and len(target_obs_list) < num_samples:
                    raw = getattr(cbs_env, "_last_raw_cbs_obs", None) or getattr(cbs_env, "_last_raw_obs", None)
                    if raw is None:
                        raw = obs if isinstance(obs, dict) else {}
                    target_obs_list.append(raw)
                    action = _cbs_policy_action(cbs_policy, obs, deterministic=deterministic_policy)
                    target_actions_list.append(action)
                    obs, _, done, truncated, _ = cbs_env.step(action)
                    step += 1
                    pbar.update(1)
            pbar.close()
        else:
            for i in tqdm(range(num_samples), desc="CBS (target, random)"):
                obs, _ = cbs_env.reset()
                raw = getattr(cbs_env, "_last_raw_cbs_obs", None) or getattr(cbs_env, "_last_raw_obs", None)
                if raw is None:
                    raw = obs if isinstance(obs, dict) else {}
                target_obs_list.append(raw)
                action = cbs_env.action_space.sample()
                target_actions_list.append(int(action))
                cbs_env.step(action)
        print(f"  Collected {len(target_obs_list)} target (CBS) samples.")
    except Exception as e:
        print(f"  Warning: could not collect CBS obs: {e}")
        target_obs_list = []
        target_actions_list = []

    # ---- Validation split from target ----
    val_obs_list = []
    if val_fraction > 0 and len(target_obs_list) > 0:
        n_val = max(1, int(len(target_obs_list) * val_fraction))
        val_obs_list = target_obs_list[:n_val]
        target_obs_list = target_obs_list[n_val:]
        target_actions_list = target_actions_list[n_val:]
        print(f"  Validation: {len(val_obs_list)}; target train: {len(target_obs_list)}")

    source_actions_list = source_actions_list[:len(source_obs_list)]
    target_actions_list = target_actions_list[:len(target_obs_list)]

    # Shuffle
    if len(source_obs_list) > 0:
        idxs = list(range(len(source_obs_list)))
        random.shuffle(idxs)
        source_obs_list = [source_obs_list[i] for i in idxs]
        source_actions_list = [source_actions_list[i] for i in idxs]
    if len(target_obs_list) > 0:
        idxs = list(range(len(target_obs_list)))
        random.shuffle(idxs)
        target_obs_list = [target_obs_list[i] for i in idxs]
        target_actions_list = [target_actions_list[i] for i in idxs]

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        out = {
            "source_obs": source_obs_list,
            "target_obs": target_obs_list,
            "val_obs": val_obs_list,
            "source_actions": source_actions_list if source_actions_list else [],
            "target_actions": target_actions_list if target_actions_list else [],
        }
        np.savez(save_path, **out, allow_pickle=True)
        print(f"Saved observations to {save_path} (source=CW, target=CBS, includes actions for --label-mode situation_action)")

    return (
        source_obs_list,
        target_obs_list,
        val_obs_list,
        source_actions_list if source_actions_list else None,
        target_actions_list if target_actions_list else None,
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train DAPN encoder with episodic training")
    parser.add_argument("--num-samples", type=int, default=1000, help="Number of samples per domain")
    parser.add_argument("--feature-size", type=int, default=256, help="Feature space size")
    parser.add_argument("--iterations", type=int, default=10000, help="Number of training iterations")
    parser.add_argument("--n-sc", type=int, default=20, help="Number of classes for source domain (Nsc)")
    parser.add_argument("--n-dc", type=int, default=5, help="Number of classes for target domain (Ndc)")
    parser.add_argument("--k", type=int, default=5, help="Number of shots per class")
    parser.add_argument("--query", type=int, default=15, help="Number of query samples per class")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--fsl-loss-weight", type=float, default=2.0,
                        help="Weight for few-shot (Lps+Lpt) loss; increase if acc stays near random (default 2.0)")
    parser.add_argument("--load-data", type=str, default=None, help="Path to load pre-collected observations")
    parser.add_argument("--save-data", type=str, default=None, help="Path to save collected observations")
    parser.add_argument("--cw-policy", type=str, default=None,
                        help="Path to Cyberwheel policy (.zip PPO or .pt) for policy-based collection; if not set, use random actions")
    parser.add_argument("--cbs-policy", type=str, default=None,
                        help="Path to CBS policy (.zip PPO) for policy-based collection; if not set, use random actions")
    parser.add_argument("--max-steps", type=int, default=200,
                        help="Max steps per episode when using policies (default 200)")
    parser.add_argument("--deterministic-policy", action="store_true",
                        help="Use deterministic (greedy) policy when collecting with --cw-policy/--cbs-policy")
    parser.add_argument("--eval-episodes", type=int, default=1000, help="Target FSL eval episodes")
    parser.add_argument("--test-interval", type=int, default=100, help="Run validation every N iterations")
    parser.add_argument(
        "--label-mode",
        type=str,
        default="situation_action",
        choices=["situation_action", "action", "cluster"],
        help="Class label mode: situation_action (needs actions in data, best semantics); action (cluster by action); cluster (K-means on obs, more samples often don't help)"
    )
    parser.add_argument("--save-encoder", type=str,
                       default="artifacts/transfer_models/dapn_encoder_episodic.pt",
                       help="Path to save trained encoder")
    parser.add_argument("--gpu", action="store_true", help="Force GPU (cuda); error if not available. Default: auto (GPU if available).")
    parser.add_argument("--device", type=str, default=None, choices=["cuda", "cpu"],
                       help="Device to use: cuda or cpu (overrides auto/default)")
    
    args = parser.parse_args()
    
    # Load or collect observations. Source = Cyberwheel, target = CyberBattleSim.
    if args.load_data and os.path.exists(args.load_data):
        print(f"Loading observations from {args.load_data}")
        data = np.load(args.load_data, allow_pickle=True)
        if 'source_obs' in data:
            # Generic keys: assume file saved with source=CW, target=CBS
            source_obs_list = data['source_obs'].tolist()
            target_obs_list = data['target_obs'].tolist() if 'target_obs' in data else []
            val_obs_list = data['val_obs'].tolist() if 'val_obs' in data else []
            source_actions_list = data['source_actions'].tolist() if 'source_actions' in data else None
            target_actions_list = data['target_actions'].tolist() if 'target_actions' in data else None
        else:
            # Legacy keys: CW = source, CBS = target (no actions in legacy format)
            source_obs_list = data['cw_obs'].tolist() if 'cw_obs' in data else []
            target_obs_list = data['cbs_obs'].tolist() if 'cbs_obs' in data else []
            val_obs_list = []
            source_actions_list = None
            target_actions_list = None
        if args.label_mode == "situation_action" and (source_actions_list is None or target_actions_list is None or
            len(source_actions_list or []) == 0 or len(target_actions_list or []) == 0):
            print("Note: --label-mode situation_action requested but loaded file has no action data.")
            print("  Labels will fall back to cluster. To use situation_action, collect data without --load-data")
            print("  (or use a file saved with --save-data so it contains source_actions and target_actions).")
    else:
        source_obs_list, target_obs_list, val_obs_list, source_actions_list, target_actions_list = (
            collect_observations_episodic(
                num_samples=args.num_samples,
                save_path=args.save_data,
                val_fraction=0.2,
                seed=42,
                cw_policy_path=args.cw_policy,
                cbs_policy_path=args.cbs_policy,
                max_steps_per_episode=args.max_steps,
                deterministic_policy=args.deterministic_policy,
            )
        )
    
    print("Source domain: Cyberwheel | Target domain: CyberBattleSim")
    
    # Prefer action-as-class when available
    if source_actions_list is not None:
        train_dapn_encoder_episodic._source_actions = source_actions_list
    if target_actions_list is not None:
        train_dapn_encoder_episodic._target_actions = target_actions_list
    train_dapn_encoder_episodic._label_mode = args.label_mode
    
    # Convert to numpy arrays/tensors
    source_obs_list = [np.array(obs) if not isinstance(obs, np.ndarray) else obs for obs in source_obs_list]
    target_obs_list = [np.array(obs) if not isinstance(obs, np.ndarray) else obs for obs in target_obs_list]
    
    if len(source_obs_list) == 0 or len(target_obs_list) == 0:
        print("Error: Need samples from both domains!")
        sys.exit(1)
    
    # Device: explicit --device wins, then --gpu, else auto (cuda if available)
    if args.device is not None:
        train_device = torch.device(args.device)
    elif args.gpu:
        if not torch.cuda.is_available():
            print("Error: --gpu requested but CUDA is not available.")
            sys.exit(1)
        train_device = torch.device("cuda")
    else:
        train_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {train_device}")
    
    # Train with episodic structure
    translator = train_dapn_encoder_episodic(
        source_obs_list=source_obs_list,
        target_obs_list=target_obs_list,
        val_obs_list=val_obs_list if val_obs_list else None,
        feature_size=args.feature_size,
        num_iterations=args.iterations,
        n_sc=args.n_sc,
        n_dc=args.n_dc,
        k=args.k,
        query=args.query,
        learning_rate=args.lr,
        save_path=args.save_encoder,
        test_interval=args.test_interval,
        eval_episodes=args.eval_episodes,
        device=train_device,
        fsl_loss_weight=args.fsl_loss_weight
    )
    
    print("\n✓ Episodic training complete!")
