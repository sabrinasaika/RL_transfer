"""
Episodic training utilities for DAPN.
Implements CategoriesSampler, observation clustering, and episodic dataset handling.
"""

import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.cluster import KMeans
from typing import List, Tuple, Optional, Dict, Any


class CategoriesSampler:
    """
    Episodic sampler that randomly selects N classes and K samples per class.
    Based on DAPN-master/data_loader.py CategoriesSampler.
    """
    
    def __init__(self, labels: np.ndarray, n_batch: int, n_cls: int, n_per: int):
        """
        Args:
            labels: Array of class labels for each sample (shape: [n_samples])
            n_batch: Number of batches/episodes to generate
            n_cls: Number of classes to sample per episode (Nsc or Ndc)
            n_per: Number of samples per class (k + query)
        """
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_per
        
        labels = np.array(labels)
        max_label = int(labels.max())
        self.m_ind = []
        for i in range(max_label + 1):
            ind = np.argwhere(labels == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)
    
    def __len__(self):
        return self.n_batch
    
    def __iter__(self):
        # Filter out empty classes
        valid_classes = [i for i, l in enumerate(self.m_ind) if len(l) >= self.n_per]
        
        # If not enough classes with sufficient samples, use classes with at least 1 sample
        if len(valid_classes) < self.n_cls:
            valid_classes = [i for i, l in enumerate(self.m_ind) if len(l) > 0]
        
        # Use at most as many classes as we have (no repetition)
        n_cls_eff = min(self.n_cls, len(valid_classes))
        if n_cls_eff < self.n_cls and len(valid_classes) > 0:
            print(f"Using {n_cls_eff} classes per episode (only {len(valid_classes)} available with enough samples).")
        
        for i_batch in range(self.n_batch):
            batch = []
            # Randomly select n_cls_eff classes from valid classes (no replacement)
            classes = torch.randperm(len(valid_classes))[:n_cls_eff]
            
            for c_idx in classes:
                c = valid_classes[c_idx]
                l = self.m_ind[c]
                if len(l) < self.n_per:
                    # If not enough samples, repeat with replacement
                    if len(l) == 0:
                        continue  # Skip empty classes
                    pos = torch.randint(0, len(l), (self.n_per,))
                else:
                    # Randomly select n_per samples from this class
                    pos = torch.randperm(len(l))[:self.n_per]
                batch.append(l[pos])
            
            if len(batch) == 0:
                continue  # Skip if no valid samples
            
            batch = torch.stack(batch).t().reshape(-1)
            yield batch


# =============================================================================
# Deterministic situation + action labels (domain-invariant)
# =============================================================================
ACTION_FAMILIES = ("SCAN", "EXPLOIT", "CRED_ACCESS", "PRIV_ESC", "MOVE", "NOOP")
ACTION_FAMILY_TO_ID = {name: idx for idx, name in enumerate(ACTION_FAMILIES)}

DEFAULT_LABEL_BINS = {
    # Upper bounds for np.digitize(..., right=True)
    "discovered": [0, 1, 2, 4, 8, 16],
    "owned": [0, 1, 2, 4, 8],
    "admin": [0, 1, 2, 4],
    "creds": [0, 1, 2, 4, 8],
}


def _bin_idx(value: int, edges: List[int]) -> int:
    """Return deterministic bin index using fixed edges."""
    try:
        v = int(value)
    except Exception:
        v = 0
    return int(np.digitize(v, edges, right=True))


def _cw_progress_metrics(obs_vec: np.ndarray) -> Dict[str, int]:
    """Extract coarse progress metrics from Cyberwheel raw obs vector."""
    HOST_ATTRS = 7  # type, sweeped, scanned, discovered, on_host, escalated, impacted
    if not isinstance(obs_vec, np.ndarray):
        obs_vec = np.asarray(obs_vec)
    n = int(obs_vec.size)
    standalone_len = n % HOST_ATTRS
    max_hosts = (n - standalone_len) // HOST_ATTRS if n >= HOST_ATTRS else 0

    discovered_hosts = 0
    owned_hosts = 0
    admin_hosts = 0
    creds = 0

    for i in range(max_hosts):
        base = i * HOST_ATTRS
        chunk = obs_vec[base : base + HOST_ATTRS]
        if np.all(chunk == -1):
            continue
        discovered = int(chunk[3] == 1)
        on_host = int(chunk[4] == 1)
        escalated = int(chunk[5] == 1)
        impacted = int(chunk[6] == 1)
        discovered_hosts += discovered
        owned_hosts += int((on_host + escalated + impacted) > 0)
        admin_hosts += escalated
        creds += escalated  # proxy: escalations imply credential progress

    return {
        "discovered": discovered_hosts,
        "owned": owned_hosts,
        "admin": admin_hosts,
        "creds": creds,
    }


def _cbs_progress_metrics(obs: Dict[str, Any]) -> Dict[str, int]:
    """Extract coarse progress metrics from CBS raw obs dict."""
    try:
        discovered = int(obs.get("discovered_node_count", 0) or 0)
    except Exception:
        discovered = 0
    try:
        priv = obs.get("nodes_privilegelevel", np.array([], dtype=np.int32))
        if not isinstance(priv, np.ndarray):
            priv = np.array(priv, dtype=np.int32) if priv is not None else np.array([], dtype=np.int32)
        owned = int((priv >= 1).sum())
        admin = int((priv >= 2).sum())
    except Exception:
        owned = 0
        admin = 0
    try:
        creds = int(obs.get("credential_cache_length", 0) or 0)
    except Exception:
        creds = 0
    return {"discovered": discovered, "owned": owned, "admin": admin, "creds": creds}


def _action_family_from_action(action) -> str:
    """Map action to shared semantic family."""
    try:
        # Unified action index path (recommended)
        if isinstance(action, (int, np.integer)):
            from adapters.action_translator import ActionTranslator
            name = ActionTranslator().unified_actions[int(action)]
            if name in ("ping_sweep", "port_scan", "discovery"):
                return "SCAN"
            if name == "lateral_move":
                return "MOVE"
            if name == "privilege_escalation":
                return "PRIV_ESC"
            if name == "impact":
                return "EXPLOIT"
            if name == "noop":
                return "NOOP"
    except Exception:
        pass

    # Fallback for CBS raw action dicts (best-effort)
    if isinstance(action, dict):
        if "connect" in action:
            try:
                conn = action.get("connect")
                if isinstance(conn, (list, tuple, np.ndarray)) and len(conn) >= 4:
                    cred_idx = int(np.asarray(conn)[3])
                    if cred_idx >= 0:
                        return "CRED_ACCESS"
            except Exception:
                pass
            return "MOVE"
        if "remote_vulnerability" in action:
            return "SCAN"
        if "local_vulnerability" in action:
            return "PRIV_ESC"
        if "credential" in action or "credentials" in action:
            return "CRED_ACCESS"
    # Fallback for tuple/array connect-like actions
    if isinstance(action, (list, tuple, np.ndarray)) and len(action) >= 4:
        try:
            cred_idx = int(np.asarray(action)[3])
            if cred_idx >= 0:
                return "CRED_ACCESS"
        except Exception:
            pass
    return "NOOP"


def build_situation_action_label(
    obs,
    action,
    domain: str,
    bins: Optional[Dict[str, List[int]]] = None
) -> int:
    """Create deterministic label from state fingerprint + action family."""
    bins = bins or DEFAULT_LABEL_BINS
    if domain == "cbs":
        metrics = _cbs_progress_metrics(obs if isinstance(obs, dict) else {})
    else:
        metrics = _cw_progress_metrics(obs if isinstance(obs, np.ndarray) else np.asarray(obs))

    d_bin = _bin_idx(metrics["discovered"], bins["discovered"])
    o_bin = _bin_idx(metrics["owned"], bins["owned"])
    a_bin = _bin_idx(metrics["admin"], bins["admin"])
    c_bin = _bin_idx(metrics["creds"], bins["creds"])
    action_family = _action_family_from_action(action)
    f_bin = ACTION_FAMILY_TO_ID.get(action_family, ACTION_FAMILY_TO_ID["NOOP"])

    d_size = len(bins["discovered"]) + 1
    o_size = len(bins["owned"]) + 1
    a_size = len(bins["admin"]) + 1
    c_size = len(bins["creds"]) + 1
    f_size = len(ACTION_FAMILIES)

    label = (((d_bin * o_size + o_bin) * a_size + a_bin) * c_size + c_bin) * f_size + f_bin
    return int(label)


def build_situation_action_labels(
    observations: List,
    actions: Optional[List],
    domain: str,
    bins: Optional[Dict[str, List[int]]] = None
) -> np.ndarray:
    """Vectorized label builder for a domain."""
    labels = []
    for i, obs in enumerate(observations):
        action = actions[i] if actions is not None and i < len(actions) else 0
        labels.append(build_situation_action_label(obs, action, domain=domain, bins=bins))
    return np.array(labels, dtype=np.int64)


def compress_labels(labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compress arbitrary labels to contiguous IDs [0..K-1]."""
    labels = np.asarray(labels)
    unique_labels, inverse = np.unique(labels, return_inverse=True)
    return inverse.astype(np.int64), unique_labels


def cluster_by_actions(
    actions: np.ndarray,
    n_clusters: int
) -> np.ndarray:
    """
    Cluster observations by actions. Each unique action becomes a class.
    If there are fewer unique actions than n_clusters, we split large action groups.
    If there are more unique actions than n_clusters, we group similar actions.
    
    Args:
        actions: Action array (n_samples,)
        n_clusters: Desired number of clusters/classes
    
    Returns:
        labels: Cluster labels for each observation (n_samples,)
    """
    actions = np.array(actions)
    unique_actions = np.unique(actions)
    n_unique = len(unique_actions)
    
    print(f"  Unique actions: {n_unique}")
    
    if n_unique <= n_clusters:
        # Use each unique action as a class
        # Map actions to class labels (0 to n_unique-1)
        action_to_class = {action: idx for idx, action in enumerate(unique_actions)}
        labels = np.array([action_to_class[action] for action in actions])
        
        # If we need more classes, split large action groups iteratively
        current_n_classes = n_unique
        while current_n_classes < n_clusters:
            # Find the largest class
            unique_labels, counts = np.unique(labels, return_counts=True)
            largest_class_idx = np.argmax(counts)
            largest_class = unique_labels[largest_class_idx]
            largest_indices = np.where(labels == largest_class)[0]
            
            # Need at least 2 samples to split
            if len(largest_indices) < 2:
                break
            
            # Split the largest class in half
            split_point = len(largest_indices) // 2
            labels[largest_indices[:split_point]] = current_n_classes
            current_n_classes += 1
            
            if current_n_classes >= n_clusters:
                break
        
        # Ensure we have exactly n_clusters (or as close as possible)
        unique_labels = np.unique(labels)
        if len(unique_labels) < n_clusters:
            # Still need more - split more aggressively
            while len(unique_labels) < n_clusters:
                unique_labels, counts = np.unique(labels, return_counts=True)
                if counts.max() < 2:
                    break  # Can't split further
                largest_class_idx = np.argmax(counts)
                largest_class = unique_labels[largest_class_idx]
                largest_indices = np.where(labels == largest_class)[0]
                if len(largest_indices) < 2:
                    break
                split_point = len(largest_indices) // 2
                labels[largest_indices[:split_point]] = len(unique_labels)
                unique_labels = np.unique(labels)
    else:
        # More unique actions than desired clusters - group similar actions
        # Use K-means on action values to group them
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        # Reshape actions to 2D for KMeans
        actions_2d = actions.reshape(-1, 1)
        labels = kmeans.fit_predict(actions_2d)
    
    unique, counts = np.unique(labels, return_counts=True)
    print(f"  Classes created: {len(unique)}")
    print(f"  Samples per class: min={counts.min()}, max={counts.max()}, mean={counts.mean():.1f}")
    
    return labels


def cluster_observations(
    observations: np.ndarray,
    n_clusters: int,
    random_state: int = 42
) -> np.ndarray:
    """
    Cluster observations into classes using K-means.
    
    Args:
        observations: Observation array (n_samples, obs_dim)
        n_clusters: Number of clusters/classes to create
        random_state: Random seed for reproducibility
    
    Returns:
        labels: Cluster labels for each observation (n_samples,)
    """
    if len(observations) < n_clusters:
        raise ValueError(
            f"Not enough samples ({len(observations)}) for {n_clusters} clusters. "
            f"Need at least {n_clusters} samples."
        )
    
    # Check for duplicate/very similar observations
    # If all observations are too similar, use random assignment instead
    obs_std = np.std(observations, axis=0)
    if np.all(obs_std < 1e-6):
        print(f"Warning: Observations are too similar (std < 1e-6). Using random assignment.")
        # Randomly assign to clusters
        labels = np.random.randint(0, n_clusters, size=len(observations))
    else:
        try:
            kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
            labels = kmeans.fit_predict(observations)
        except Exception as e:
            print(f"Warning: K-means failed ({e}). Using random assignment.")
            labels = np.random.randint(0, n_clusters, size=len(observations))
    
    # Verify each cluster has at least some samples
    unique, counts = np.unique(labels, return_counts=True)
    actual_n_clusters = len(unique)
    print(f"Clustering results: {actual_n_clusters} classes (requested {n_clusters})")
    print(f"  Samples per class: min={counts.min()}, max={counts.max()}, mean={counts.mean():.1f}")
    
    # Ensure we have at least n_clusters classes by splitting large clusters if needed
    if actual_n_clusters < n_clusters:
        print(f"  Splitting large clusters to reach {n_clusters} classes...")
        # Find the largest cluster and split it
        largest_cluster = unique[np.argmax(counts)]
        largest_indices = np.where(labels == largest_cluster)[0]
        
        # Split into multiple clusters
        n_needed = n_clusters - actual_n_clusters + 1
        split_labels = np.random.randint(actual_n_clusters, actual_n_clusters + n_needed - 1, 
                                        size=len(largest_indices))
        labels[largest_indices] = split_labels[:len(largest_indices)]
        
        # Reassign to ensure we have exactly n_clusters
        unique_new = np.unique(labels)
        if len(unique_new) > n_clusters:
            # Map extra clusters back
            for i, extra_cluster in enumerate(unique_new[n_clusters:]):
                labels[labels == extra_cluster] = unique_new[i % n_clusters]
        
        unique, counts = np.unique(labels, return_counts=True)
        print(f"  After splitting: {len(unique)} classes")
        print(f"  Samples per class: min={counts.min()}, max={counts.max()}, mean={counts.mean():.1f}")
    
    return labels


class EpisodicDataset(Dataset):
    """
    Dataset for episodic training with support/query sets.
    """
    
    def __init__(
        self,
        observations: List,
        labels: np.ndarray,
        shot: int = 5,
        query: int = 15,
        preprocessor=None
    ):
        """
        Args:
            observations: List of observations (can be dicts for CBS, arrays for Cyberwheel)
            labels: Class labels for each observation
            shot: Number of support samples per class (k)
            query: Number of query samples per class
            preprocessor: UnifiedFullObsPreprocessor to convert raw obs to fixed-size vectors
        """
        self.observations = observations
        self.labels = np.array(labels)
        self.shot = shot
        self.query = query
        self.preprocessor = preprocessor
    
    def __len__(self):
        return len(self.observations)
    
    def _process_obs(self, obs):
        """Convert raw observation to fixed-size tensor using preprocessor."""
        if isinstance(obs, np.ndarray) and obs.dtype == object:
            try:
                if obs.size == 1:
                    obs = obs.item()
            except Exception:
                pass
        if self.preprocessor is None:
            # Fallback: try to convert directly
            if isinstance(obs, dict):
                raise ValueError("Cannot process dict observations without preprocessor")
            elif isinstance(obs, np.ndarray):
                return torch.FloatTensor(obs)
            elif isinstance(obs, torch.Tensor):
                return obs
            else:
                return torch.FloatTensor(np.array(obs))
        
        # Use preprocessor to convert raw obs to fixed-size vector
        if isinstance(obs, dict):
            unified_vec = self.preprocessor.preprocess_cbs(obs)
        elif isinstance(obs, np.ndarray):
            unified_vec = self.preprocessor.preprocess_cw(obs)
        elif isinstance(obs, torch.Tensor):
            return obs
        else:
            unified_vec = np.array(obs, dtype=np.float32)
            # Pad/truncate to unified_dim
            if len(unified_vec) < self.preprocessor.unified_dim:
                unified_vec = np.pad(unified_vec, (0, self.preprocessor.unified_dim - len(unified_vec)), 'constant')
            elif len(unified_vec) > self.preprocessor.unified_dim:
                unified_vec = unified_vec[:self.preprocessor.unified_dim]
        
        # Normalize to [0, 1] range
        max_vals = np.ones(self.preprocessor.unified_dim, dtype=np.float32) * 100.0
        normalized = np.clip(unified_vec / max_vals, 0.0, 1.0)
        
        return torch.FloatTensor(normalized)
    
    def __getitem__(self, idx):
        obs = self.observations[idx]
        label = self.labels[idx]
        
        obs_tensor = self._process_obs(obs)
        return obs_tensor, label


def create_episodic_dataloaders(
    source_obs: List,
    target_obs: List,
    source_actions: List[int] = None,
    target_actions: List[int] = None,
    n_sc: int = 20,  # Number of classes for Ds
    n_dc: int = 5,   # Number of classes for Dd
    k: int = 5,      # Shots per class
    query: int = 15, # Query samples per class
    n_batch_ds: int = 100,  # Number of episodes from Ds
    n_batch_dd: int = 100,  # Number of episodes from Dd
    device: str = "cpu",
    preprocessor=None,
    label_mode: str = "cluster",
    label_bins: Optional[Dict[str, List[int]]] = None
):
    """
    Create episodic dataloaders for Ds and Dd following DAPN paper.
    
    Args:
        source_obs: Source domain observations (Ds)
        target_obs: Target domain observations (Dd)
        n_sc: Number of classes for source domain episodes (Nsc)
        n_dc: Number of classes for target domain episodes (Ndc)
        k: Number of support samples per class
        query: Number of query samples per class
        n_batch_ds: Number of episodes to generate from Ds
        n_batch_dd: Number of episodes to generate from Dd
        device: Device to use
    
    Returns:
        source_sampler: CategoriesSampler for Ds
        target_sampler: CategoriesSampler for Dd
        source_dataset: EpisodicDataset for Ds
        target_dataset: EpisodicDataset for Dd
        source_labels: Cluster labels for source observations
        target_labels: Cluster labels for target observations
    """
    def _unwrap_obs(raw_obs):
        """Handle numpy object arrays that wrap dicts or other mappings."""
        if isinstance(raw_obs, np.ndarray) and raw_obs.dtype == object:
            try:
                if raw_obs.size == 1:
                    return raw_obs.item()
            except Exception:
                pass
        return raw_obs
    
    # Convert raw observations to fixed-size vectors for clustering
    # If preprocessor is provided, use it; otherwise try direct conversion
    if preprocessor is not None:
        source_obs_array = np.array([
            preprocessor.preprocess_cbs(_unwrap_obs(obs)) if isinstance(_unwrap_obs(obs), dict)
            else preprocessor.preprocess_cw(obs) if isinstance(obs, np.ndarray)
            else np.array(obs)
            for obs in source_obs
        ])
        target_obs_array = np.array([
            preprocessor.preprocess_cbs(_unwrap_obs(obs)) if isinstance(_unwrap_obs(obs), dict)
            else preprocessor.preprocess_cw(obs) if isinstance(obs, np.ndarray)
            else np.array(obs)
            for obs in target_obs
        ])
    else:
        # Fallback: try direct conversion (may fail for dicts)
        try:
            source_obs_array = np.array([np.array(obs) for obs in source_obs])
            target_obs_array = np.array([np.array(obs) for obs in target_obs])
        except (ValueError, TypeError) as e:
            raise ValueError(f"Cannot convert observations to arrays without preprocessor: {e}. "
                           "Provide UnifiedFullObsPreprocessor for raw observations.")
    
    print(f"\n{'='*60}")
    print("CREATING EPISODIC DATALOADERS")
    print(f"{'='*60}")
    print(f"Source domain (Ds): {len(source_obs)} samples")
    print(f"Target domain (Dd): {len(target_obs)} samples")
    
    # Step 1: Create class labels
    label_mode = (label_mode or "cluster").lower()
    if label_mode == "situation_action":
        print("\nCreating deterministic situation+action labels...")
        has_src_actions = source_actions is not None and len(source_actions) > 0
        has_tgt_actions = target_actions is not None and len(target_actions) > 0
        if not has_src_actions or not has_tgt_actions:
            print("Warning: situation_action requested but actions are missing or empty.")
            print("  Falling back to cluster labels. With cluster, more samples often don't improve results")
            print("  because K-means classes are not semantically meaningful.")
            print("  To use situation_action: run without --load-data (collect fresh with actions), or")
            print("  load a .npz that contains 'source_actions' and 'target_actions' (e.g. from --save-data).")
            label_mode = "cluster"
        else:
            source_labels = build_situation_action_labels(
                observations=source_obs,
                actions=source_actions,
                domain="cw",
                bins=label_bins
            )
            target_labels = build_situation_action_labels(
                observations=target_obs,
                actions=target_actions,
                domain="cbs",
                bins=label_bins
            )

    if label_mode == "action":
        if source_actions is not None and len(source_actions) > 0:
            print(f"\nClustering source domain into {n_sc} classes by actions...")
            source_labels = cluster_by_actions(np.array(source_actions), n_clusters=n_sc)
        else:
            print(f"\nClustering source domain into {n_sc} classes (no actions, using observation similarity)...")
            source_labels = cluster_observations(source_obs_array, n_clusters=n_sc)
        if target_actions is not None and len(target_actions) > 0:
            print(f"\nClustering target domain into {n_dc} classes by actions...")
            target_labels = cluster_by_actions(np.array(target_actions), n_clusters=n_dc)
        else:
            print(f"\nClustering target domain into {n_dc} classes (no actions, using observation similarity)...")
            target_labels = cluster_observations(target_obs_array, n_clusters=n_dc)

    if label_mode == "cluster":
        print(f"\nClustering source domain into {n_sc} classes (observation similarity)...")
        if (source_actions is not None and len(source_actions) > 0) or (target_actions is not None and len(target_actions) > 0):
            print("  (Actions are available; for semantically meaningful classes use --label-mode situation_action and ensure actions are saved when collecting/loading.)")
        source_labels = cluster_observations(source_obs_array, n_clusters=n_sc)
        print(f"\nClustering target domain into {n_dc} classes (observation similarity)...")
        target_labels = cluster_observations(target_obs_array, n_clusters=n_dc)

    # Compress labels to contiguous IDs for CategoriesSampler
    source_labels, source_label_values = compress_labels(source_labels)
    target_labels, target_label_values = compress_labels(target_labels)
    print(f"Source labels compressed: {len(source_label_values)} unique classes")
    print(f"Target labels compressed: {len(target_label_values)} unique classes")
    
    # Step 2: Create datasets (pass preprocessor to handle raw observations)
    source_dataset = EpisodicDataset(source_obs, source_labels, shot=k, query=query, preprocessor=preprocessor)
    target_dataset = EpisodicDataset(target_obs, target_labels, shot=k, query=query, preprocessor=preprocessor)
    
    # Step 3: Create episodic samplers
    n_per_source = k + query  # Total samples per class needed
    n_per_target = k + query
    
    source_sampler = CategoriesSampler(
        labels=source_labels,
        n_batch=n_batch_ds,
        n_cls=n_sc,
        n_per=n_per_source
    )
    
    target_sampler = CategoriesSampler(
        labels=target_labels,
        n_batch=n_batch_dd,
        n_cls=n_dc,
        n_per=n_per_target
    )
    
    print(f"\n{'='*60}")
    print("EPISODIC SAMPLER SUMMARY")
    print(f"{'='*60}")
    print(f"Source domain:")
    print(f"  - Classes per episode: {n_sc}")
    print(f"  - Support samples: {n_sc} × {k} = {n_sc * k}")
    print(f"  - Query samples: {n_sc} × {query} = {n_sc * query}")
    print(f"  - Total per episode: {n_sc * (k + query)}")
    print(f"  - Number of episodes: {n_batch_ds}")
    print(f"\nTarget domain:")
    print(f"  - Classes per episode: {n_dc}")
    print(f"  - Support samples: {n_dc} × {k} = {n_dc * k}")
    print(f"  - Query samples: {n_dc} × {query} = {n_dc * query}")
    print(f"  - Total per episode: {n_dc * (k + query)}")
    print(f"  - Number of episodes: {n_batch_dd}")
    
    return (
        source_sampler,
        target_sampler,
        source_dataset,
        target_dataset,
        source_labels,
        target_labels
    )


def euclidean_metric(query: torch.Tensor, proto: torch.Tensor) -> torch.Tensor:
    """
    Compute euclidean distance between query samples and prototypes.
    
    Args:
        query: Query features (n_query, feature_dim)
        proto: Prototype features (n_classes, feature_dim)
    
    Returns:
        logits: Distance matrix (n_query, n_classes) - negative distances for similarity
    """
    n_query = query.size(0)
    n_proto = proto.size(0)
    
    # Expand dimensions for broadcasting
    query = query.unsqueeze(1).expand(n_query, n_proto, -1)  # (n_query, n_proto, feature_dim)
    proto = proto.unsqueeze(0).expand(n_query, n_proto, -1)  # (n_query, n_proto, feature_dim)
    
    # Compute negative euclidean distance (negative because smaller distance = higher similarity)
    logits = -torch.pow(query - proto, 2).sum(dim=2)  # (n_query, n_proto)
    
    return logits


def compute_prototypes(
    support_features: torch.Tensor,
    shot: int,
    n_way: int
) -> torch.Tensor:
    """
    Compute prototypes by averaging support set features.
    
    Args:
        support_features: Support set features (shot * n_way, feature_dim)
        shot: Number of support samples per class
        n_way: Number of classes
    
    Returns:
        prototypes: Prototype for each class (n_way, feature_dim)
    """
    # Reshape: (shot, n_way, feature_dim)
    support_features = support_features.reshape(shot, n_way, -1)
    # Average over shot dimension: (n_way, feature_dim)
    prototypes = support_features.mean(dim=0)
    return prototypes


def count_acc(logits: torch.Tensor, label: torch.Tensor) -> float:
    """
    Compute accuracy from logits and labels.
    
    Args:
        logits: Prediction logits (n_samples, n_classes)
        label: True labels (n_samples,)
    
    Returns:
        accuracy: Classification accuracy
    """
    pred = torch.argmax(logits, dim=1)
    return float((pred == label).float().mean().item())
