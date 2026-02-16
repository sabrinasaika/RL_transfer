"""
Observation translator that uses full observations instead of reduced 8-dim.
"""

import numpy as np
import torch
from typing import Dict, Optional

from adapters.full_observation_encoder import (
    CBSFullObservationEncoder,
    CWFullObservationEncoder
)


class FullObservationTranslator:
    """
    Translator that preserves full observations and uses encoders
    to map them to a shared feature space.
    """
    
    def __init__(
        self,
        use_transfer: bool = False,
        encoder_path: Optional[str] = None,
        feature_size: int = 64,
        device: Optional[torch.device] = None
    ):
        self.use_transfer = use_transfer
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.cbs_encoder = None
        self.cw_encoder = None
        
        if use_transfer and encoder_path:
            try:
                checkpoint = torch.load(encoder_path, map_location=self.device)
                
                self.cbs_encoder = CBSFullObservationEncoder(feature_size=feature_size, use_graph=False)
                self.cw_encoder = CWFullObservationEncoder(feature_size=feature_size)
                
                if 'cbs_encoder_state_dict' in checkpoint:
                    self.cbs_encoder.load_state_dict(checkpoint['cbs_encoder_state_dict'])
                if 'cw_encoder_state_dict' in checkpoint:
                    self.cw_encoder.load_state_dict(checkpoint['cw_encoder_state_dict'])
                
                self.cbs_encoder = self.cbs_encoder.to(self.device)
                self.cw_encoder = self.cw_encoder.to(self.device)
                self.cbs_encoder.eval()
                self.cw_encoder.eval()
                
                print(f"Loaded full observation encoders from {encoder_path}")
            except Exception as e:
                print(f"Warning: Could not load encoders: {e}")
                self.use_transfer = False
    
    def from_cbs(self, obs: Dict, return_raw: bool = False) -> np.ndarray:
        """
        Translate CBS observation.
        
        Args:
            obs: Full CBS observation dict
            return_raw: If True, return raw dict instead of encoding
            
        Returns:
            If return_raw: raw observation dict
            If use_transfer: encoded features [feature_size]
            Otherwise: placeholder (full obs not reduced)
        """
        if return_raw:
            return obs
        
        if self.use_transfer and self.cbs_encoder is not None:
            with torch.no_grad():
                features = self.cbs_encoder(obs)
                return features.cpu().numpy()
        else:
            # Return a placeholder - in practice you'd use the full obs
            # For now, return zeros as placeholder
            return np.zeros(64, dtype=np.float32)
    
    def from_cw(self, obs_vec: np.ndarray, return_raw: bool = False) -> np.ndarray:
        """
        Translate Cyberwheel observation.
        
        Args:
            obs_vec: Full Cyberwheel observation vector
            return_raw: If True, return raw vector instead of encoding
            
        Returns:
            If return_raw: raw observation vector
            If use_transfer: encoded features [feature_size]
            Otherwise: placeholder
        """
        if return_raw:
            return obs_vec
        
        if self.use_transfer and self.cw_encoder is not None:
            with torch.no_grad():
                features = self.cw_encoder(obs_vec)
                return features.cpu().numpy()
        else:
            # Return placeholder
            return np.zeros(64, dtype=np.float32)

