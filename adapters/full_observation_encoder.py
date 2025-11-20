"""
Full observation encoders for both CyberBattleSim and Cyberwheel.
Uses all observation fields instead of the reduced 8-dim representation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple


class CBSFullObservationEncoder(nn.Module):
    """
    Encoder for full CyberBattleSim observations.
    Handles all fields including graph structure.
    """
    
    def __init__(
        self,
        max_nodes: int = 50,
        max_credentials: int = 100,
        property_count: int = 3,
        feature_size: int = 64,
        use_graph: bool = True
    ):
        super().__init__()
        self.max_nodes = max_nodes
        self.max_credentials = max_credentials
        self.property_count = property_count
        self.feature_size = feature_size
        self.use_graph = use_graph
        
        # Encoder for scalar fields
        scalar_dim = 6  # newly_discovered_nodes_count, lateral_move, customer_data_found, 
                         # probe_result, escalation, credential_cache_length
        self.scalar_encoder = nn.Sequential(
            nn.Linear(scalar_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )
        
        # Encoder for node properties matrix
        # Shape: (max_nodes, property_count)
        self.node_prop_encoder = nn.Sequential(
            nn.Linear(max_nodes * property_count, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        # Encoder for privilege levels
        # Shape: (max_nodes,)
        self.privilege_encoder = nn.Sequential(
            nn.Linear(max_nodes, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        
        # Encoder for credential cache matrix
        # Variable length, use attention or RNN
        self.credential_encoder = nn.Sequential(
            nn.Linear(2, 16),  # Each credential: (node_idx, port_idx)
            nn.ReLU(),
            nn.Linear(16, 16)
        )
        # Pooling for variable-length credentials
        self.credential_pool = nn.AdaptiveAvgPool1d(1)  # or use attention
        
        # Graph encoder (if using graph structure)
        if use_graph:
            try:
                from torch_geometric.nn import GCNConv, global_mean_pool
                self.graph_available = True
                self.gcn1 = GCNConv(1, 32)  # Node features: 1-dim (can be extended)
                self.gcn2 = GCNConv(32, 32)
                self.graph_pool = global_mean_pool
            except ImportError:
                print("Warning: torch_geometric not available, graph encoding disabled")
                self.graph_available = False
                self.use_graph = False
        else:
            self.graph_available = False
        
        # Graph feature encoder (fallback if no GNN)
        if not self.use_graph:
            # Use edge count and node count as graph features
            self.graph_feature_encoder = nn.Sequential(
                nn.Linear(2, 16),  # (num_nodes, num_edges)
                nn.ReLU(),
                nn.Linear(16, 16)
            )
        
        # Fusion layer: combine all features
        total_features = 32 + 64 + 32 + 16 + (32 if use_graph else 16)
        self.fusion = nn.Sequential(
            nn.Linear(total_features, 128),
            nn.ReLU(),
            nn.Linear(128, feature_size),
            nn.LayerNorm(feature_size)
        )
    
    def forward(self, obs: Dict) -> torch.Tensor:
        """
        Encode full CBS observation.
        
        Args:
            obs: Full CBS observation dict
            
        Returns:
            features: Encoded features [feature_size]
        """
        features_list = []
        
        # 1. Scalar fields
        scalars = torch.tensor([
            obs.get("newly_discovered_nodes_count", 0),
            obs.get("lateral_move", 0),
            obs.get("customer_data_found", 0),
            obs.get("probe_result", 0),
            obs.get("escalation", 0),
            obs.get("credential_cache_length", 0)
        ], dtype=torch.float32, device=next(self.parameters()).device).unsqueeze(0)
        scalar_features = self.scalar_encoder(scalars)
        features_list.append(scalar_features)
        
        # 2. Node properties matrix
        device = next(self.parameters()).device
        node_props = obs.get("discovered_nodes_properties", np.zeros((0, self.property_count)))
        if isinstance(node_props, np.ndarray):
            node_props = torch.from_numpy(node_props).float().to(device)
        else:
            node_props = torch.tensor(node_props, dtype=torch.float32, device=device)
        
        # Handle empty or wrong shape
        if node_props.numel() == 0:
            node_props = torch.zeros(self.max_nodes, self.property_count)
        elif len(node_props.shape) == 1:
            # Reshape if needed
            node_props = node_props.view(-1, self.property_count) if node_props.shape[0] % self.property_count == 0 else torch.zeros(self.max_nodes, self.property_count)
        
        # Pad or truncate to max_nodes
        if node_props.shape[0] < self.max_nodes:
            padding = torch.zeros(self.max_nodes - node_props.shape[0], node_props.shape[1] if len(node_props.shape) > 1 else self.property_count, device=device)
            node_props = torch.cat([node_props, padding], dim=0)
        elif node_props.shape[0] > self.max_nodes:
            node_props = node_props[:self.max_nodes]
        
        # Ensure correct property count
        if node_props.shape[1] != self.property_count:
            if node_props.shape[1] < self.property_count:
                padding = torch.zeros(node_props.shape[0], self.property_count - node_props.shape[1], device=device)
                node_props = torch.cat([node_props, padding], dim=1)
            else:
                node_props = node_props[:, :self.property_count]
        
        node_props_flat = node_props.flatten().unsqueeze(0)
        node_prop_features = self.node_prop_encoder(node_props_flat)
        features_list.append(node_prop_features)
        
        # 3. Privilege levels
        privileges = obs.get("nodes_privilegelevel", np.zeros(self.max_nodes))
        if isinstance(privileges, np.ndarray):
            privileges = torch.from_numpy(privileges).float().to(device)
        else:
            privileges = torch.tensor(privileges, dtype=torch.float32, device=device)
        
        # Pad or truncate
        if len(privileges) < self.max_nodes:
            padding = torch.zeros(self.max_nodes - len(privileges), device=device)
            privileges = torch.cat([privileges, padding])
        elif len(privileges) > self.max_nodes:
            privileges = privileges[:self.max_nodes]
        
        privilege_features = self.privilege_encoder(privileges.unsqueeze(0))
        features_list.append(privilege_features)
        
        # 4. Credential cache
        cred_cache = obs.get("credential_cache_matrix", [])
        if len(cred_cache) > 0:
            # Convert to tensor
            cred_tensors = []
            for cred in cred_cache[:self.max_credentials]:
                if isinstance(cred, (list, tuple, np.ndarray)):
                    cred_tensors.append(torch.tensor(cred[:2], dtype=torch.float32, device=device))
                else:
                    cred_tensors.append(torch.zeros(2, device=device))
            
            if cred_tensors:
                cred_batch = torch.stack(cred_tensors).unsqueeze(0)  # [1, num_creds, 2]
                cred_features = self.credential_encoder(cred_batch)  # [1, num_creds, 16]
                # Pool to fixed size
                cred_features = cred_features.transpose(1, 2)  # [1, 16, num_creds]
                cred_features = self.credential_pool(cred_features)  # [1, 16, 1]
                cred_features = cred_features.squeeze(-1)  # [1, 16]
            else:
                cred_features = torch.zeros(1, 16, device=device)
        else:
            cred_features = torch.zeros(1, 16, device=device)
        features_list.append(cred_features)
        
        # 5. Graph structure
        if self.use_graph and self.graph_available:
            graph = obs.get("_explored_network", None)
            if graph is not None and hasattr(graph, "nodes") and hasattr(graph, "edges"):
                # Convert to PyTorch Geometric format
                try:
                    node_features = torch.ones(len(graph.nodes()), 1, device=device)  # Simple node features
                    edge_index = torch.tensor(list(graph.edges()), device=device).t().contiguous() if len(graph.edges()) > 0 else torch.empty((2, 0), dtype=torch.long, device=device)
                    
                    if edge_index.shape[1] > 0:
                        x = self.gcn1(node_features, edge_index)
                        x = F.relu(x)
                        x = self.gcn2(x, edge_index)
                        graph_features = self.graph_pool(x, torch.zeros(len(graph.nodes()), dtype=torch.long, device=device))  # [1, 32]
                    else:
                        graph_features = torch.zeros(1, 32, device=device)
                except Exception as e:
                    print(f"Warning: Graph encoding failed: {e}")
                    graph_features = torch.zeros(1, 32, device=device)
            else:
                graph_features = torch.zeros(1, 32, device=device)
        else:
            # Fallback: use graph statistics
            graph = obs.get("_explored_network", None)
            if graph is not None:
                num_nodes = len(graph.nodes()) if hasattr(graph, "nodes") else 0
                num_edges = len(graph.edges()) if hasattr(graph, "edges") else 0
            else:
                num_nodes, num_edges = 0, 0
            
            graph_stats = torch.tensor([[num_nodes, num_edges]], dtype=torch.float32, device=device)
            graph_features = self.graph_feature_encoder(graph_stats)
        
        features_list.append(graph_features)
        
        # 6. Fuse all features
        combined = torch.cat(features_list, dim=-1)  # [1, total_features]
        output = self.fusion(combined)  # [1, feature_size]
        
        return output.squeeze(0)  # [feature_size]


class CWFullObservationEncoder(nn.Module):
    """
    Encoder for full Cyberwheel observations.
    Handles variable-length host vectors.
    """
    
    def __init__(
        self,
        max_obs_size: int = 701,
        feature_size: int = 64,
        host_attr_size: int = 7
    ):
        super().__init__()
        self.max_obs_size = max_obs_size
        self.feature_size = feature_size
        self.host_attr_size = host_attr_size
        
        # Encoder for full observation vector
        self.encoder = nn.Sequential(
            nn.Linear(max_obs_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, feature_size),
            nn.LayerNorm(feature_size)
        )
        
        # Alternative: Host-aware encoder (processes hosts separately)
        self.host_encoder = nn.Sequential(
            nn.Linear(host_attr_size, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )
        self.host_pool = nn.AdaptiveAvgPool1d(1)
    
    def forward(self, obs_vec: np.ndarray, use_host_aware: bool = False) -> torch.Tensor:
        """
        Encode full Cyberwheel observation.
        
        Args:
            obs_vec: Full observation vector (variable length)
            use_host_aware: If True, process hosts separately
            
        Returns:
            features: Encoded features [feature_size]
        """
        if isinstance(obs_vec, np.ndarray):
            obs_vec = torch.from_numpy(obs_vec).float()
        else:
            obs_vec = torch.tensor(obs_vec, dtype=torch.float32)
        
        if use_host_aware:
            # Process hosts separately
            n = obs_vec.shape[0]
            standalone_len = n % self.host_attr_size
            num_hosts = (n - standalone_len) // self.host_attr_size
            
            host_features = []
            for i in range(num_hosts):
                base = i * self.host_attr_size
                host_attrs = obs_vec[base:base + self.host_attr_size]
                host_feat = self.host_encoder(host_attrs.unsqueeze(0))
                host_features.append(host_feat)
            
            if host_features:
                host_batch = torch.cat(host_features, dim=0)  # [num_hosts, 16]
                host_batch = host_batch.transpose(0, 1).unsqueeze(0)  # [1, 16, num_hosts]
                pooled = self.host_pool(host_batch)  # [1, 16, 1]
                pooled = pooled.squeeze(-1).squeeze(0)  # [16]
                
                # Combine with standalone features
                if standalone_len > 0:
                    standalone = obs_vec[-standalone_len:]
                    # Pad standalone to fixed size
                    if standalone_len < 4:
                        standalone = torch.cat([standalone, torch.zeros(4 - standalone_len)])
                    standalone_feat = nn.Linear(4, 16)(standalone.unsqueeze(0)).squeeze(0)
                    combined = torch.cat([pooled, standalone_feat])
                else:
                    combined = pooled
                
                # Project to feature size
                if combined.shape[0] < self.feature_size:
                    combined = F.pad(combined, (0, self.feature_size - combined.shape[0]))
                elif combined.shape[0] > self.feature_size:
                    combined = combined[:self.feature_size]
                
                return combined
            else:
                # Fallback to simple encoding
                obs_vec_padded = self._pad_or_truncate(obs_vec, self.max_obs_size)
                return self.encoder(obs_vec_padded.unsqueeze(0)).squeeze(0)
        else:
            # Simple: pad/truncate and encode
            obs_vec_padded = self._pad_or_truncate(obs_vec, self.max_obs_size)
            return self.encoder(obs_vec_padded.unsqueeze(0)).squeeze(0)
    
    def _pad_or_truncate(self, vec: torch.Tensor, target_size: int) -> torch.Tensor:
        """Pad or truncate vector to target size"""
        if vec.shape[0] < target_size:
            padding = torch.zeros(target_size - vec.shape[0])
            return torch.cat([vec, padding])
        elif vec.shape[0] > target_size:
            return vec[:target_size]
        return vec


class UnifiedFullObservationEncoder(nn.Module):
    """
    Unified encoder that can handle both CBS and Cyberwheel full observations.
    Projects both to a shared feature space.
    """
    
    def __init__(
        self,
        cbs_encoder: Optional[CBSFullObservationEncoder] = None,
        cw_encoder: Optional[CWFullObservationEncoder] = None,
        feature_size: int = 64
    ):
        super().__init__()
        self.feature_size = feature_size
        
        if cbs_encoder is None:
            self.cbs_encoder = CBSFullObservationEncoder(feature_size=feature_size)
        else:
            self.cbs_encoder = cbs_encoder
        
        if cw_encoder is None:
            self.cw_encoder = CWFullObservationEncoder(feature_size=feature_size)
        else:
            self.cw_encoder = cw_encoder
    
    def encode_cbs(self, obs: Dict) -> torch.Tensor:
        """Encode full CBS observation"""
        return self.cbs_encoder(obs)
    
    def encode_cw(self, obs_vec: np.ndarray) -> torch.Tensor:
        """Encode full Cyberwheel observation"""
        return self.cw_encoder(obs_vec)
    
    def forward(self, obs, obs_type: str = "cbs") -> torch.Tensor:
        """
        Encode observation based on type.
        
        Args:
            obs: Observation (dict for CBS, array for CW)
            obs_type: "cbs" or "cw"
        """
        if obs_type == "cbs":
            return self.encode_cbs(obs)
        else:
            return self.encode_cw(obs)

