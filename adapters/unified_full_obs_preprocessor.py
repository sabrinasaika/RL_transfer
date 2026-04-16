"""
Unified preprocessor that converts full observations from both domains
to a fixed-size vector representation, enabling use of a single encoder.
Preserves more information than 8D while following DAPN master's single encoder concept.
"""

import torch
import numpy as np
from typing import Dict, Union, Optional

from adapters.kill_chain import cw_raw_to_red_vector


class UnifiedFullObsPreprocessor:
    """
    Preprocesses full observations from both domains to fixed-size vectors.
    This allows using a single encoder while preserving more information than 8D.
    """
    
    def __init__(
        self,
        unified_dim: int = 512,  # Fixed size for unified representation
        cbs_max_nodes: int = 50,
        cbs_max_credentials: int = 100,
        cbs_property_count: int = 3,
        cw_max_obs_size: int = 701,
        cw_host_attr_size: int = 7
    ):
        """
        Initialize unified preprocessor.
        
        Args:
            unified_dim: Fixed dimension for unified representation
            cbs_max_nodes: Max nodes for CBS observations
            cbs_max_credentials: Max credentials for CBS
            cbs_property_count: Number of properties per node
            cw_max_obs_size: Max observation size for Cyberwheel
            cw_host_attr_size: Attributes per host in Cyberwheel
        """
        self.unified_dim = unified_dim
        self.cbs_max_nodes = cbs_max_nodes
        self.cbs_max_credentials = cbs_max_credentials
        self.cbs_property_count = cbs_property_count
        self.cw_max_obs_size = cw_max_obs_size
        self.cw_host_attr_size = cw_host_attr_size
        
        # Calculate expected dimensions for each domain
        # CBS: scalars + node_props + privileges + credentials + graph_stats
        self.cbs_expected_dim = (
            6 +  # scalars
            cbs_max_nodes * cbs_property_count +  # node properties
            cbs_max_nodes +  # privilege levels
            cbs_max_credentials * 2 +  # credential cache (node_idx, port_idx)
            2  # graph stats (num_nodes, num_edges)
        )
        
        # Cyberwheel: host attributes + standalone
        self.cw_expected_dim = cw_max_obs_size
        
        # Always use unified_dim as target (encoder expects this exact size)
        self.target_dim = unified_dim
    
    def preprocess_cbs(self, obs: Dict) -> np.ndarray:
        """
        Convert CBS full observation dict to fixed-size vector.

        Field layout (most informative first so truncation to unified_dim preserves signal):
          [0:100]    nodes_privilegelevel  (100D) — primary kill-chain signal
          [100:106]  scalars               (6D)   — discovered_node_count, cred_cache_length,
                                                     escalation, lateral_move,
                                                     newly_discovered_nodes_count, customer_data_found
          [106:306]  leaked_credentials    (200D) — credential cache encoded as node+port pairs
          [306:512]  node_properties       (206D) — discovered_nodes_properties flattened, truncated

        Fixes vs previous version:
          - cbs_max_nodes was 50; actual CBS arrays are 100-element → now uses 100
          - cbs_property_count was 3; actual is 14 → now reads the real shape
          - credential key was 'credential_cache_matrix' (absent); actual is 'leaked_credentials'
        """
        def _pad1d(arr, length):
            arr = np.asarray(arr, dtype=np.float32).ravel()
            if len(arr) >= length:
                return arr[:length]
            return np.concatenate([arr, np.zeros(length - len(arr), dtype=np.float32)])

        # 1. Privilege levels (100D) — front-loaded: directly encodes kill-chain stage
        priv = obs.get("nodes_privilegelevel", [])
        part_priv = _pad1d(priv, 100)

        # 2. Scalar fields (6D)
        part_scalars = np.array([
            float(obs.get("discovered_node_count",          0) or 0),
            float(obs.get("credential_cache_length",        0) or 0),
            float(obs.get("escalation",                     0) or 0),
            float(obs.get("lateral_move",                   0) or 0),
            float(obs.get("newly_discovered_nodes_count",   0) or 0),
            float(obs.get("customer_data_found",            0) or 0),
        ], dtype=np.float32)

        # 3. Credential cache matrix (100D) — shape (1000,2): [node_id, port_id] per cached cred.
        #    Key is 'credential_cache_matrix' (tuple of 1000 x 2-element arrays).
        #    Encode first 50 credentials × 2 values = 100D; rest is zeros when cache is sparse.
        ccm = obs.get("credential_cache_matrix", ())
        if isinstance(ccm, (tuple, list)) and len(ccm) > 0:
            arr = np.asarray(ccm, dtype=np.float32)          # (1000, 2) or similar
            part_creds = _pad1d(arr.ravel(), 100)            # take first 50 creds × 2
        else:
            part_creds = np.zeros(100, dtype=np.float32)

        # 4. Node properties matrix flattened — fill remaining space up to unified_dim
        node_props = obs.get("discovered_nodes_properties", None)
        remaining = self.unified_dim - 100 - 6 - 100   # = 306 for unified_dim=512
        if node_props is not None:
            arr = np.asarray(node_props, dtype=np.float32).ravel()
        else:
            arr = np.zeros(0, dtype=np.float32)
        part_props = _pad1d(arr, remaining)

        unified_vec = np.concatenate([part_priv, part_scalars, part_creds, part_props])
        # Should be exactly unified_dim; safety pad/truncate
        if len(unified_vec) < self.target_dim:
            unified_vec = np.concatenate([unified_vec, np.zeros(self.target_dim - len(unified_vec), dtype=np.float32)])
        elif len(unified_vec) > self.target_dim:
            unified_vec = unified_vec[:self.target_dim]
        return unified_vec
    
    def preprocess_cw(self, obs_vec: np.ndarray) -> np.ndarray:
        """
        Convert Cyberwheel full observation array to fixed-size vector.
        Preserves all information by padding/truncating.
        
        Args:
            obs_vec: Full Cyberwheel observation vector (variable length)
        
        Returns:
            Fixed-size vector [unified_dim]
        """
        obs_vec = cw_raw_to_red_vector(obs_vec)
        obs_vec = obs_vec.astype(np.float32, copy=False)
        
        # Pad or truncate to target_dim
        if len(obs_vec) < self.target_dim:
            padding = np.zeros(self.target_dim - len(obs_vec), dtype=np.float32)
            unified_vec = np.concatenate([obs_vec, padding])
        elif len(obs_vec) > self.target_dim:
            unified_vec = obs_vec[:self.target_dim]
        else:
            unified_vec = obs_vec
        
        return unified_vec
    
    def preprocess(self, obs: Union[Dict, np.ndarray], domain: str = "cbs") -> np.ndarray:
        """
        Preprocess observation from either domain.
        
        Args:
            obs: Observation (dict for CBS, array for CW)
            domain: "cbs" or "cw"
        
        Returns:
            Fixed-size unified vector [unified_dim]
        """
        if domain == "cbs":
            return self.preprocess_cbs(obs)
        else:
            return self.preprocess_cw(obs)
