"""
DAPN translator using FULL observations with SINGLE shared encoder.
Converts both domains to fixed-size vectors first, then uses one encoder.
Follows DAPN master concept while preserving all observation information.
"""

import os
import torch
import torch.nn as nn
import numpy as np
from typing import Optional
from adapters.unified_full_obs_preprocessor import UnifiedFullObsPreprocessor
from adapters.dapn_observation_encoder import DAPNObservationEncoder, DAPNDomainAdapter


class DAPNUnifiedFullObsTranslator:
    """
    DAPN translator that uses full observations with a SINGLE shared encoder.
    Preprocesses both domains to fixed-size vectors, then uses one encoder.
    This follows DAPN master's single encoder concept while preserving all information.
    """
    
    def __init__(
        self,
        use_dapn: bool = True,
        encoder_path: Optional[str] = None,
        feature_size: int = 256,
        unified_dim: int = 512,  # Fixed size for unified representation
        device: Optional[torch.device] = None,
        use_adversarial: bool = False
    ):
        """
        Initialize unified full observation DAPN translator.
        
        Args:
            use_dapn: Whether to use DAPN encoding
            encoder_path: Path to saved encoder checkpoint
            feature_size: Size of feature space
            unified_dim: Fixed dimension for unified representation
            device: Device to run on
            use_adversarial: Whether to use adversarial domain adaptation (for training)
        """
        self.use_dapn = use_dapn
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_size = feature_size
        self.unified_dim = unified_dim
        self.use_adversarial = use_adversarial
        
        # Create preprocessor to convert both domains to fixed-size vectors
        self.preprocessor = UnifiedFullObsPreprocessor(unified_dim=unified_dim)

        # Normalization stats (loaded from checkpoint; None = fallback /100 clipping)
        self.norm_mean: Optional[np.ndarray] = None
        self.norm_std:  Optional[np.ndarray] = None
        self.clip_z: float = 5.0

        # Create SINGLE shared encoder (follows DAPN master concept)
        self.shared_encoder = None
        self.domain_adapter = None
        
        if use_dapn:
            # Single encoder that works on unified fixed-size vectors
            self.shared_encoder = DAPNObservationEncoder(
                input_dim=unified_dim,  # Takes unified fixed-size vectors
                feature_size=feature_size,
                device=self.device
            ).to(self.device)
            
            # Create domain adapter for adversarial training
            if use_adversarial:
                self.domain_adapter = DAPNDomainAdapter(
                    feature_dim=feature_size,
                    method="DANN"
                ).to(self.device)
            
            # Load pre-trained weights if provided
            if encoder_path:
                self.load_encoder(encoder_path)
            
            # Set to eval mode by default
            self.shared_encoder.eval()
            if self.domain_adapter:
                self.domain_adapter.eval()
    
    def load_encoder(self, encoder_path: str):
        """Load encoder weights from checkpoint."""
        try:
            checkpoint = torch.load(encoder_path, map_location=self.device, weights_only=False)
            
            if 'shared_encoder_state_dict' in checkpoint:
                self.shared_encoder.load_state_dict(checkpoint['shared_encoder_state_dict'])
            elif 'encoder_state_dict' in checkpoint:
                self.shared_encoder.load_state_dict(checkpoint['encoder_state_dict'])
            else:
                print(f"Warning: No encoder found in checkpoint")
            
            if 'domain_adapter_state_dict' in checkpoint and self.domain_adapter:
                self.domain_adapter.load_state_dict(checkpoint['domain_adapter_state_dict'])
            
            if 'unified_dim' in checkpoint:
                self.unified_dim = checkpoint['unified_dim']
                self.preprocessor = UnifiedFullObsPreprocessor(unified_dim=self.unified_dim)

            # Load z-score normalization stats saved by train_dapn_encoder.py
            if 'norm_mean' in checkpoint and 'norm_std' in checkpoint:
                self.norm_mean = np.asarray(checkpoint['norm_mean'], dtype=np.float32)
                self.norm_std  = np.asarray(checkpoint['norm_std'],  dtype=np.float32)
                self.clip_z    = float(checkpoint.get('clip_z', 5.0))
                print(f"  Loaded z-score norm stats (mean/std shape={self.norm_mean.shape})")
            else:
                print(f"  Warning: no norm stats in checkpoint — using /100 fallback")

            print(f"Loaded unified full observation DAPN encoder from {encoder_path}")
        except Exception as e:
            print(f"Warning: Could not load encoder from {encoder_path}: {e}")
    
    def save_encoder(self, save_path: str):
        """Save encoder weights to checkpoint."""
        checkpoint = {
            'shared_encoder_state_dict': self.shared_encoder.state_dict() if self.shared_encoder else None,
            'domain_adapter_state_dict': self.domain_adapter.state_dict() if self.domain_adapter else None,
            'feature_size': self.feature_size,
            'unified_dim': self.unified_dim
        }
        torch.save(checkpoint, save_path)
        print(f"Saved unified full observation DAPN encoder to {save_path}")
    
    def from_cbs(self, obs) -> np.ndarray:
        """
        Translate CBS observation using unified preprocessor + single encoder.
        
        Args:
            obs: CBS observation (dict with all fields)
        
        Returns:
            Encoded features [feature_size]
        """
        if not self.use_dapn or self.shared_encoder is None:
            return np.zeros(self.feature_size, dtype=np.float32)
        
        # Step 1: Preprocess full CBS dict to fixed-size vector
        unified_vec = self.preprocessor.preprocess_cbs(obs)

        # Step 2: Normalize — z-score if stats available, else /100 fallback
        normalized = self._normalize(unified_vec)

        # Step 3: Encode using single shared encoder
        with torch.no_grad():
            obs_tensor = torch.from_numpy(normalized).float().to(self.device)
            features = self.shared_encoder(obs_tensor)
            if isinstance(features, torch.Tensor):
                features = features.cpu().numpy()

        return features.astype(np.float32)
    
    def from_cw(self, obs_vec: np.ndarray) -> np.ndarray:
        """
        Translate Cyberwheel observation using unified preprocessor + single encoder.
        
        Args:
            obs_vec: Cyberwheel observation vector (variable length)
        
        Returns:
            Encoded features [feature_size]
        """
        if not self.use_dapn or self.shared_encoder is None:
            return np.zeros(self.feature_size, dtype=np.float32)
        
        # Step 1: Preprocess full Cyberwheel array to fixed-size vector
        unified_vec = self.preprocessor.preprocess_cw(obs_vec)

        # Step 2: Normalize — z-score if stats available, else /100 fallback
        normalized = self._normalize(unified_vec)

        # Step 3: Encode using single shared encoder
        with torch.no_grad():
            obs_tensor = torch.from_numpy(normalized).float().to(self.device)
            features = self.shared_encoder(obs_tensor)
            if isinstance(features, torch.Tensor):
                features = features.cpu().numpy()

        return features.astype(np.float32)

    def _normalize(self, vec: np.ndarray) -> np.ndarray:
        """Z-score normalize if stats loaded, else clip to [0,1] with /100 fallback."""
        vec = np.asarray(vec, dtype=np.float32)
        if self.norm_mean is not None and self.norm_std is not None:
            z = (vec - self.norm_mean) / self.norm_std
            return np.clip(z, -self.clip_z, self.clip_z).astype(np.float32)
        max_vals = np.ones(self.unified_dim, dtype=np.float32) * 100.0
        return np.clip(vec / max_vals, 0.0, 1.0).astype(np.float32)
