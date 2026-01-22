"""
Unified preprocessor that converts full observations from both domains
to a fixed-size vector representation, enabling use of a single encoder.
Preserves more information than 8D while following DAPN master's single encoder concept.
"""

import torch
import numpy as np
from typing import Dict, Union, Optional


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
        Preserves all information by flattening all fields.
        
        Args:
            obs: Full CBS observation dictionary
        
        Returns:
            Fixed-size vector [unified_dim]
        """
        vec_parts = []
        
        # 1. Scalar fields (6 values)
        scalars = [
            float(obs.get("newly_discovered_nodes_count", 0)),
            float(obs.get("lateral_move", 0)),
            float(obs.get("customer_data_found", 0)),
            float(obs.get("probe_result", 0)),
            float(obs.get("escalation", 0)),
            float(obs.get("credential_cache_length", 0))
        ]
        vec_parts.append(np.array(scalars, dtype=np.float32))
        
        # 2. Node properties matrix (max_nodes × property_count)
        node_props = obs.get("discovered_nodes_properties", np.zeros((0, self.cbs_property_count)))
        if isinstance(node_props, np.ndarray):
            if node_props.size == 0:
                node_props = np.zeros((self.cbs_max_nodes, self.cbs_property_count), dtype=np.float32)
            else:
                # Pad or truncate to max_nodes
                if node_props.shape[0] < self.cbs_max_nodes:
                    padding = np.zeros((self.cbs_max_nodes - node_props.shape[0], node_props.shape[1]), dtype=np.float32)
                    node_props = np.vstack([node_props, padding])
                elif node_props.shape[0] > self.cbs_max_nodes:
                    node_props = node_props[:self.cbs_max_nodes]
                
                # Ensure correct property count
                if node_props.shape[1] < self.cbs_property_count:
                    padding = np.zeros((node_props.shape[0], self.cbs_property_count - node_props.shape[1]), dtype=np.float32)
                    node_props = np.hstack([node_props, padding])
                elif node_props.shape[1] > self.cbs_property_count:
                    node_props = node_props[:, :self.cbs_property_count]
        else:
            node_props = np.zeros((self.cbs_max_nodes, self.cbs_property_count), dtype=np.float32)
        
        vec_parts.append(node_props.flatten())
        
        # 3. Privilege levels (max_nodes)
        priv = obs.get("nodes_privilegelevel", np.array([], dtype=np.int32))
        if isinstance(priv, np.ndarray):
            if priv.size == 0:
                priv = np.zeros(self.cbs_max_nodes, dtype=np.float32)
            else:
                priv = priv.astype(np.float32)
                if len(priv) < self.cbs_max_nodes:
                    padding = np.zeros(self.cbs_max_nodes - len(priv), dtype=np.float32)
                    priv = np.concatenate([priv, padding])
                elif len(priv) > self.cbs_max_nodes:
                    priv = priv[:self.cbs_max_nodes]
        else:
            priv = np.zeros(self.cbs_max_nodes, dtype=np.float32)
        
        vec_parts.append(priv)
        
        # 4. Credential cache matrix (max_credentials × 2)
        cred_cache = obs.get("credential_cache_matrix", ())
        if isinstance(cred_cache, tuple) and len(cred_cache) > 0:
            # Convert tuple of arrays to matrix
            cred_list = []
            for cred in cred_cache[:self.cbs_max_credentials]:
                if isinstance(cred, np.ndarray) and cred.size >= 2:
                    cred_list.append(cred[:2])
                else:
                    cred_list.append(np.array([0, 0], dtype=np.float32))
            
            if len(cred_list) < self.cbs_max_credentials:
                padding = [np.array([0, 0], dtype=np.float32) for _ in range(self.cbs_max_credentials - len(cred_list))]
                cred_list.extend(padding)
            
            cred_matrix = np.array(cred_list, dtype=np.float32)
        else:
            cred_matrix = np.zeros((self.cbs_max_credentials, 2), dtype=np.float32)
        
        vec_parts.append(cred_matrix.flatten())
        
        # 5. Graph statistics (num_nodes, num_edges)
        explored = obs.get("_explored_network", None)
        if explored is not None and hasattr(explored, "nodes") and hasattr(explored, "edges"):
            num_nodes = len(explored.nodes()) if hasattr(explored, "nodes") else 0
            num_edges = len(explored.edges()) if hasattr(explored, "edges") else 0
        else:
            num_nodes = obs.get("discovered_node_count", 0)
            num_edges = 0
        
        vec_parts.append(np.array([float(num_nodes), float(num_edges)], dtype=np.float32))
        
        # 6. Additional fields (probe_result, escalation details)
        additional = [
            float(obs.get("discovered_node_count", 0)),
            float(obs.get("probe_result", 0)),
            float(obs.get("escalation", 0))
        ]
        vec_parts.append(np.array(additional, dtype=np.float32))
        
        # Concatenate all parts
        unified_vec = np.concatenate(vec_parts)
        
        # Pad or truncate to target_dim
        if len(unified_vec) < self.target_dim:
            padding = np.zeros(self.target_dim - len(unified_vec), dtype=np.float32)
            unified_vec = np.concatenate([unified_vec, padding])
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
        if not isinstance(obs_vec, np.ndarray):
            obs_vec = np.asarray(obs_vec, dtype=np.float32)
        else:
            obs_vec = obs_vec.astype(np.float32)
        
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
