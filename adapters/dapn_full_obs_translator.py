"""
DAPN observation translator that uses FULL raw observations (not 8D unified).
Combines full observation encoders with DAPN's adversarial domain adaptation.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict
from adapters.full_observation_encoder import (
    CBSFullObservationEncoder,
    CWFullObservationEncoder
)
from adapters.dapn_observation_encoder import DAPNDomainAdapter


class DAPNFullObservationTranslator:
    """
    DAPN translator that uses full raw observations instead of 8D unified format.
    Uses full observation encoders with adversarial domain adaptation.
    """
    
    def __init__(
        self,
        use_dapn: bool = True,
        encoder_path: Optional[str] = None,
        feature_size: int = 256,
        device: Optional[torch.device] = None,
        use_adversarial: bool = False,
        # CBS encoder params
        cbs_max_nodes: int = 50,
        cbs_max_credentials: int = 100,
        # CW encoder params
        cw_max_obs_size: int = 701,
        cw_host_attr_size: int = 7
    ):
        """
        Initialize DAPN full observation translator.
        
        Args:
            use_dapn: Whether to use DAPN encoding
            encoder_path: Path to saved encoder checkpoint
            feature_size: Size of feature space
            device: Device to run on
            use_adversarial: Whether to use adversarial domain adaptation (for training)
            cbs_max_nodes: Max nodes for CBS encoder
            cbs_max_credentials: Max credentials for CBS encoder
            cw_max_obs_size: Max observation size for CW encoder
            cw_host_attr_size: Host attribute size for CW encoder
        """
        self.use_dapn = use_dapn
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_size = feature_size
        self.use_adversarial = use_adversarial
        
        # Create full observation encoders (not 8D encoders!)
        self.cbs_encoder = None
        self.cw_encoder = None
        self.domain_adapter = None
        
        if use_dapn:
            # Create CBS encoder for full dict observations
            self.cbs_encoder = CBSFullObservationEncoder(
                max_nodes=cbs_max_nodes,
                max_credentials=cbs_max_credentials,
                feature_size=feature_size,
                use_graph=False  # Can enable if torch_geometric available
            ).to(self.device)
            
            # Create CW encoder for full variable-length arrays
            self.cw_encoder = CWFullObservationEncoder(
                max_obs_size=cw_max_obs_size,
                feature_size=feature_size,
                host_attr_size=cw_host_attr_size
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
            self.cbs_encoder.eval()
            self.cw_encoder.eval()
            if self.domain_adapter:
                self.domain_adapter.eval()
    
    def load_encoder(self, encoder_path: str):
        """Load encoder weights from checkpoint."""
        try:
            checkpoint = torch.load(encoder_path, map_location=self.device)
            
            if 'cbs_encoder_state_dict' in checkpoint:
                self.cbs_encoder.load_state_dict(checkpoint['cbs_encoder_state_dict'])
            elif 'encoder_state_dict' in checkpoint:
                # Try loading as shared encoder
                self.cbs_encoder.load_state_dict(checkpoint['encoder_state_dict'])
                self.cw_encoder.load_state_dict(checkpoint['encoder_state_dict'])
            
            if 'cw_encoder_state_dict' in checkpoint:
                self.cw_encoder.load_state_dict(checkpoint['cw_encoder_state_dict'])
            
            if 'domain_adapter_state_dict' in checkpoint and self.domain_adapter:
                self.domain_adapter.load_state_dict(checkpoint['domain_adapter_state_dict'])
            
            print(f"Loaded DAPN full observation encoder from {encoder_path}")
        except Exception as e:
            print(f"Warning: Could not load encoder from {encoder_path}: {e}")
    
    def save_encoder(self, save_path: str):
        """Save encoder weights to checkpoint."""
        checkpoint = {
            'cbs_encoder_state_dict': self.cbs_encoder.state_dict() if self.cbs_encoder else None,
            'cw_encoder_state_dict': self.cw_encoder.state_dict() if self.cw_encoder else None,
            'domain_adapter_state_dict': self.domain_adapter.state_dict() if self.domain_adapter else None,
            'feature_size': self.feature_size
        }
        torch.save(checkpoint, save_path)
        print(f"Saved DAPN full observation encoder to {save_path}")
    
    def from_cbs(self, obs) -> np.ndarray:
        """
        Translate CBS observation using full observation encoder + DAPN.
        
        Args:
            obs: CBS observation (dict with all fields)
        
        Returns:
            Encoded features [feature_size]
        """
        if not self.use_dapn or self.cbs_encoder is None:
            # Fallback: return zeros
            return np.zeros(self.feature_size, dtype=np.float32)
        
        # Use full observation encoder directly (no 8D conversion!)
        with torch.no_grad():
            features = self.cbs_encoder(obs)
            if isinstance(features, torch.Tensor):
                features = features.cpu().numpy()
        
        return features.astype(np.float32)
    
    def from_cw(self, obs_vec: np.ndarray) -> np.ndarray:
        """
        Translate Cyberwheel observation using full observation encoder + DAPN.
        
        Args:
            obs_vec: Cyberwheel observation vector (variable length)
        
        Returns:
            Encoded features [feature_size]
        """
        if not self.use_dapn or self.cw_encoder is None:
            # Fallback: return zeros
            return np.zeros(self.feature_size, dtype=np.float32)
        
        # Use full observation encoder directly (no 8D conversion!)
        with torch.no_grad():
            features = self.cw_encoder(obs_vec)
            if isinstance(features, torch.Tensor):
                features = features.cpu().numpy()
        
        return features.astype(np.float32)
