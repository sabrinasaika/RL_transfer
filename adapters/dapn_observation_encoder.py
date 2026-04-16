"""
DAPN-based observation encoder for domain adaptation between CBS and Cyberwheel.
Uses DAPN's domain adaptation principles to align observations from different domains.
"""

import os
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict
import sys
from pathlib import Path

# Add DAPN to path
dapn_path = Path(__file__).parent.parent / "DAPN-master"
if str(dapn_path) not in sys.path:
    sys.path.insert(0, str(dapn_path))

try:
    import domain_adaptive_module.network as dapn_network
    import domain_adaptive_module.loss as dapn_loss
    DAPN_AVAILABLE = True
except ImportError:
    print("Warning: DAPN modules not available. Install DAPN dependencies.")
    DAPN_AVAILABLE = False


class DAPNObservationEncoder(nn.Module):
    """
    DAPN-based encoder for observations that uses domain adaptation.
    Adapts DAPN's ResNet architecture for vector observations.
    """
    
    def __init__(
        self,
        input_dim: int = 8,
        feature_size: int = 256,
        use_bottleneck: bool = True,
        bottleneck_dim: int = 256,
        device: Optional[torch.device] = None
    ):
        super().__init__()
        self.input_dim = input_dim
        self.feature_size = feature_size
        self.use_bottleneck = use_bottleneck
        self.bottleneck_dim = bottleneck_dim
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Feature extraction layers (similar to ResNet feature layers)
        # Since we have vector inputs, we use MLPs instead of conv layers
        # REMOVED BatchNorm to prevent feature collapse - using LayerNorm instead
        self.feature_layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),  # LayerNorm instead of BatchNorm (works with any batch size)
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),  # Reduced dropout
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
        )
        
        # Bottleneck layer (optional)
        if use_bottleneck:
            self.bottleneck = nn.Linear(512, bottleneck_dim)
            self.bottleneck.apply(self._init_weights)
            self.__in_features = bottleneck_dim
        else:
            self.__in_features = 512
        
        # Attention module to suppress domain-specific information
        # Produces a gating vector in [0, 1] for each feature dimension
        self.attention = nn.Sequential(
            nn.Linear(self.__in_features, self.__in_features),
            nn.ReLU(inplace=True),
            nn.Linear(self.__in_features, self.__in_features),
            nn.Sigmoid()
        )
        
        # Autoencoder decoder to reconstruct input from embedded features
        # Helps preserve semantic information while suppressing domain-specific noise
        decoder_hidden = max(128, self.__in_features)
        self.decoder = nn.Sequential(
            nn.Linear(self.__in_features, decoder_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(decoder_hidden, input_dim)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        """Initialize weights similar to DAPN's initialization."""
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            # Use kaiming init for ReLU (better than xavier for ReLU)
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif classname.find('LayerNorm') != -1:
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor, return_all: bool = False):
        """
        Forward pass through the encoder.
        
        Args:
            x: Input observation tensor of shape (batch_size, input_dim) or (input_dim,)
            return_all: If True, return (embedded, pre_embed, recon, attention)
        
        Returns:
            features: Encoded features of shape (batch_size, feature_size) or (feature_size,)
        """
        was_1d = x.dim() == 1
        if was_1d:
            x = x.unsqueeze(0)
        
        # Extract features (LayerNorm works with any batch size, no special handling needed)
        pre_features = self.feature_layers(x)
        
        # Apply bottleneck if enabled
        if self.use_bottleneck:
            pre_features = self.bottleneck(pre_features)
        
        # Apply attention gating
        attention = self.attention(pre_features)
        features = pre_features * attention
        
        # Autoencoder reconstruction (from embedded features)
        recon = self.decoder(features)
        
        if was_1d:
            features = features.squeeze(0)
            pre_features = pre_features.squeeze(0)
            recon = recon.squeeze(0)
            attention = attention.squeeze(0)
        
        if return_all:
            return features, pre_features, recon, attention
        return features
    
    def output_num(self):
        """Return output feature dimension."""
        return self.__in_features


class DAPNDomainAdapter(nn.Module):
    """
    Domain adapter using DAPN's adversarial network for domain alignment.
    """
    
    def __init__(
        self,
        feature_dim: int,
        hidden_size: int = 1024,
        method: str = "DANN"
    ):
        super().__init__()
        self.method = method
        self.feature_dim = feature_dim
        
        # Always use fallback implementation to avoid numpy compatibility issues
        # The DAPN AdversarialNetwork has numpy compatibility issues with newer versions
        self.ad_net = nn.Sequential(
            nn.Linear(feature_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, 1)
            # No Sigmoid here — BCEWithLogitsLoss expects raw logits
        )
        
        # Initialize weights properly to prevent saturation
        self._init_weights()
    
    def _init_weights(self):
        """Initialize discriminator weights to prevent saturation."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Use smaller initialization to prevent saturation
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through adversarial network.
        
        Args:
            features: Feature tensor of shape (batch_size, feature_dim)
        
        Returns:
            domain_pred: Domain prediction (0=source, 1=target)
        """
        return self.ad_net(features)


class DAPNObservationTranslator:
    """
    Observation translator using DAPN for domain adaptation.
    Can be used as a drop-in replacement for ObservationTranslator.
    
    NOTE: This class is for 8D observations only (legacy approach).
    For full observations, use DAPNUnifiedFullObsTranslator instead.
    """
    
    def __init__(
        self,
        use_dapn: bool = True,
        encoder_path: Optional[str] = None,
        feature_size: int = 256,
        input_dim: int = 8,
        device: Optional[torch.device] = None,
        use_adversarial: bool = False
    ):
        """
        Initialize DAPN observation translator.
        
        Args:
            use_dapn: Whether to use DAPN encoder
            encoder_path: Path to saved encoder checkpoint
            feature_size: Size of feature space
            input_dim: Input observation dimension (should be 8 for this legacy class)
            device: Device to run on
            use_adversarial: Whether to use adversarial domain adaptation (for training)
        
        NOTE: This translator converts observations to 8D vectors first.
        For full observation support, use DAPNUnifiedFullObsTranslator.
        """
        self.use_dapn = use_dapn
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_size = feature_size
        self.input_dim = input_dim
        self.use_adversarial = use_adversarial
        
        # Default scales for normalization (8D-specific: [discovered_nodes, compromised_hosts, 
        # discovered_hosts, known_vulns, creds, steps_elapsed, dist_goal, alerts])
        self.default_scales = np.array([50, 50, 50, 200, 50, 1000, 20.0, 100], dtype=np.float32)
        
        # Create encoder(s) for domain adaptation
        # Option: Use single shared encoder (follows original DAPN) or separate encoders
        self.use_shared_encoder = os.environ.get("DAPN_USE_SHARED_ENCODER", "1") == "1"
        
        self.shared_encoder = None
        self.cbs_encoder = None
        self.cw_encoder = None
        self.domain_adapter = None
        
        if use_dapn:
            if self.use_shared_encoder:
                # Use SINGLE shared encoder (follows original DAPN concept)
                # Both domains are converted to unified 8D format first, so one encoder works
                self.shared_encoder = DAPNObservationEncoder(
                    input_dim=input_dim,
                    feature_size=feature_size,
                    device=self.device
                ).to(self.device)
                # For compatibility, also set cbs_encoder and cw_encoder to point to shared
                self.cbs_encoder = self.shared_encoder
                self.cw_encoder = self.shared_encoder
            else:
                # Use separate encoders (original implementation)
                self.cbs_encoder = DAPNObservationEncoder(
                    input_dim=input_dim,
                    feature_size=feature_size,
                    device=self.device
                ).to(self.device)
                
                self.cw_encoder = DAPNObservationEncoder(
                    input_dim=input_dim,
                    feature_size=feature_size,
                    device=self.device
                ).to(self.device)
            
            # Create domain adapter if using adversarial training
            if use_adversarial:
                self.domain_adapter = DAPNDomainAdapter(
                    feature_dim=feature_size,
                    method="DANN"
                ).to(self.device)
            
            # Load pre-trained weights if provided
            if encoder_path:
                self.load_encoder(encoder_path)
            
            # Set to eval mode by default
            if self.use_shared_encoder:
                self.shared_encoder.eval()
            else:
                self.cbs_encoder.eval()
                self.cw_encoder.eval()
            if self.domain_adapter:
                self.domain_adapter.eval()
    
    def load_encoder(self, encoder_path: str):
        """Load encoder weights from checkpoint."""
        try:
            checkpoint = torch.load(encoder_path, map_location=self.device, weights_only=False)
            
            if self.use_shared_encoder:
                # Load shared encoder
                if 'shared_encoder_state_dict' in checkpoint:
                    self.shared_encoder.load_state_dict(checkpoint['shared_encoder_state_dict'])
                elif 'encoder_state_dict' in checkpoint:
                    self.shared_encoder.load_state_dict(checkpoint['encoder_state_dict'])
                elif 'cbs_encoder_state_dict' in checkpoint:
                    # If only one encoder saved, use it for shared
                    self.shared_encoder.load_state_dict(checkpoint['cbs_encoder_state_dict'])
                else:
                    print(f"Warning: No shared encoder found in checkpoint")
            else:
                # Load separate encoders
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
            
            print(f"Loaded DAPN encoder from {encoder_path} (shared={self.use_shared_encoder})")
        except Exception as e:
            print(f"Warning: Could not load encoder from {encoder_path}: {e}")
    
    def save_encoder(self, save_path: str):
        """Save encoder weights to checkpoint."""
        if self.use_shared_encoder:
            checkpoint = {
                'shared_encoder_state_dict': self.shared_encoder.state_dict() if self.shared_encoder else None,
                'domain_adapter_state_dict': self.domain_adapter.state_dict() if self.domain_adapter else None,
                'feature_size': self.feature_size,
                'input_dim': self.input_dim,
                'use_shared_encoder': True
            }
        else:
            checkpoint = {
                'cbs_encoder_state_dict': self.cbs_encoder.state_dict() if self.cbs_encoder else None,
                'cw_encoder_state_dict': self.cw_encoder.state_dict() if self.cw_encoder else None,
                'domain_adapter_state_dict': self.domain_adapter.state_dict() if self.domain_adapter else None,
                'feature_size': self.feature_size,
                'input_dim': self.input_dim,
                'use_shared_encoder': False
            }
        torch.save(checkpoint, save_path)
        print(f"Saved DAPN encoder to {save_path} (shared={self.use_shared_encoder})")
    
    def from_cbs(self, obs) -> np.ndarray:
        """
        Translate CBS observation using DAPN encoder.
        
        Args:
            obs: CBS observation (dict or normalized vector)
        
        Returns:
            Encoded features or normalized vector
        """
        # First normalize to unified representation
        vec = self._cbs_to_unified(obs)
        vec_normalized = self._normalize(vec)
        
        # Encode using DAPN if enabled
        if self.use_dapn:
            encoder = self.shared_encoder if self.use_shared_encoder else self.cbs_encoder
            if encoder is not None:
                return self._encode_to_features(vec_normalized, encoder=encoder)
        
        return vec_normalized
    
    def from_cw(self, obs_vec: np.ndarray) -> np.ndarray:
        """
        Translate Cyberwheel observation using DAPN encoder.
        
        Args:
            obs_vec: Cyberwheel observation vector
        
        Returns:
            Encoded features or normalized vector
        """
        # First normalize to unified representation
        vec = self._cw_to_unified(obs_vec)
        vec_normalized = self._normalize(vec)
        
        # Encode using DAPN if enabled
        if self.use_dapn:
            encoder = self.shared_encoder if self.use_shared_encoder else self.cw_encoder
            if encoder is not None:
                return self._encode_to_features(vec_normalized, encoder=encoder)
        
        return vec_normalized
    
    def _cbs_to_unified(self, obs) -> np.ndarray:
        """Convert CBS observation to unified 8-dim representation."""
        import numpy as _np
        
        discovered_node_count = int(obs.get("discovered_node_count", 0) if isinstance(obs, dict) else 0)
        
        priv = obs.get("nodes_privilegelevel", _np.array([], dtype=_np.int32)) if isinstance(obs, dict) else _np.array([], dtype=_np.int32)
        if not isinstance(priv, _np.ndarray):
            priv = _np.array(priv, dtype=_np.int32) if priv is not None else _np.array([], dtype=_np.int32)
        compromised_hosts = int((priv >= 1).sum())
        
        discovered_hosts = discovered_node_count
        
        known_vulns = 0
        if isinstance(obs, dict):
            props = obs.get("discovered_nodes_properties")
            if isinstance(props, _np.ndarray) and props.size > 0:
                try:
                    if props.ndim == 2 and props.shape[1] > 0:
                        col = props[:, props.shape[1] - 1]
                        vals = _np.asarray(col, dtype=_np.float32)
                        if (vals > 5).any():
                            known_vulns = int(_np.maximum(vals, 0).sum())
                        else:
                            known_vulns = int((vals > 0).sum())
                except Exception:
                    known_vulns = 0
        
        creds = int(obs.get("credential_cache_length", 0) if isinstance(obs, dict) else 0)
        
        steps_elapsed = 0
        if isinstance(obs, dict):
            explored = obs.get("_explored_network")
            try:
                if explored is not None and hasattr(explored, "number_of_edges"):
                    steps_elapsed = int(explored.number_of_edges())
            except Exception:
                steps_elapsed = 0
        
        if discovered_node_count > 0:
            dist_goal = (1.0 - compromised_hosts / float(discovered_node_count)) * self.default_scales[6]
        else:
            dist_goal = float(self.default_scales[6])

        probe_result = int(obs.get("probe_result", 0) or 0 if isinstance(obs, dict) else 0)
        escalation_val = int(obs.get("escalation", 0) or 0 if isinstance(obs, dict) else 0)
        alerts = int((probe_result == 1)) + int(escalation_val > 0)
        
        return _np.array([
            discovered_node_count,
            compromised_hosts,
            discovered_hosts,
            known_vulns,
            creds,
            steps_elapsed,
            float(dist_goal),
            alerts
        ], dtype=_np.float32)
    
    def _cw_to_unified(self, obs_vec: np.ndarray) -> np.ndarray:
        """Convert Cyberwheel observation to unified 8-dim representation."""
        HOST_ATTRS = 7
        if not isinstance(obs_vec, np.ndarray):
            obs_vec = np.asarray(obs_vec)
        n = int(obs_vec.size)
        standalone_len = n % HOST_ATTRS
        max_hosts = (n - standalone_len) // HOST_ATTRS if n >= HOST_ATTRS else 0
        
        total_hosts_present = 0
        compromised_hosts = 0
        discovered_hosts = 0
        scanned_hosts = 0
        escalated_count = 0
        impacted_count = 0
        
        for i in range(max_hosts):
            base = i * HOST_ATTRS
            chunk = obs_vec[base : base + HOST_ATTRS]
            if np.all(chunk == -1):
                continue
            total_hosts_present += 1
            discovered = int(chunk[3] == 1)
            on_host = int(chunk[4] == 1)
            escalated = int(chunk[5] == 1)
            impacted = int(chunk[6] == 1)
            scanned = int(chunk[2] == 1)
            discovered_hosts += discovered
            compromised_hosts += int((on_host + escalated + impacted) > 0)
            scanned_hosts += scanned
            escalated_count += escalated
            impacted_count += impacted
        
        known_vulns = scanned_hosts
        credentials_found = escalated_count
        steps_fraction = 0.0
        if standalone_len > 0:
            try:
                quadrant = int(obs_vec[-standalone_len])
                quadrant = min(max(quadrant, 1), 4)
                steps_fraction = (quadrant - 0.5) / 4.0
            except Exception:
                steps_fraction = 0.0
        steps_elapsed = steps_fraction * float(self.default_scales[5])
        
        if total_hosts_present > 0:
            goal_fraction_remaining = 1.0 - (impacted_count / float(total_hosts_present))
        else:
            goal_fraction_remaining = 1.0
        dist_to_goal = goal_fraction_remaining * float(self.default_scales[6])
        alerts = escalated_count + impacted_count
        
        return np.array([
            discovered_hosts,
            compromised_hosts,
            total_hosts_present,
            known_vulns,
            credentials_found,
            steps_elapsed,
            dist_to_goal,
            alerts
        ], dtype=np.float32)
    
    def _normalize(self, vec: np.ndarray) -> np.ndarray:
        """Normalize observation vector."""
        return np.clip(vec / self.default_scales, 0.0, 1.0)
    
    def _encode_to_features(self, obs: np.ndarray, domain: str = 'cbs', encoder=None) -> np.ndarray:
        """
        Encode observation to feature space using DAPN encoder.
        
        Args:
            obs: Normalized observation vector [input_dim]
            domain: 'cbs' or 'cw' (used if encoder not provided)
            encoder: Specific encoder to use (if None, selects based on domain)
        
        Returns:
            features: Encoded features [feature_size]
        """
        if encoder is None:
            # Fallback to domain-based selection
            if self.use_shared_encoder:
                encoder = self.shared_encoder
            else:
                encoder = self.cbs_encoder if domain == 'cbs' else self.cw_encoder
        
        if encoder is None:
            return obs
        
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            features = encoder(obs_tensor)
            return features.squeeze(0).cpu().numpy()

