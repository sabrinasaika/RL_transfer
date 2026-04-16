
import numpy as np
import torch
from typing import Optional

OBS_DIM = 8

# ---------- CBS helpers ----------
def _get(o, *candidates, default=0):
    for name in candidates:
        if isinstance(o, dict) and name in o:
            return o[name]
        if hasattr(o, name):
            return getattr(o, name)
    return default

def _len_safe(x):
    try:
        return len(x)
    except Exception:
        return 0

def _sum_flag(seq, *flag_names):
    total = 0
    for item in (seq or []):
        val = False
        for fn in flag_names:
            if isinstance(item, dict):
                val = val or bool(item.get(fn, False))
            elif hasattr(item, fn):
                val = val or bool(getattr(item, fn))
        total += int(val)
    return total

class ObservationTranslator:
    def __init__(self, use_transfer: bool = False, encoder_path: Optional[str] = None, 
                 feature_size: int = 64, device: Optional[torch.device] = None):
        """
        Initialize observation translator with optional transfer learning encoder.
        
        Args:
            use_transfer: Whether to use transfer learning encoder
            encoder_path: Path to saved encoder checkpoint
            feature_size: Size of feature space (if loading encoder)
            device: Device to run encoder on
        """
        self.default_scales = np.array([50, 50, 50, 200, 50, 1000, 20.0, 100], dtype=np.float32)
        self.use_transfer = use_transfer
        self.encoder = None
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if use_transfer and encoder_path:
            try:
                # Prefer DAPN encoder (main pipeline); fall back to legacy transfer_encoder
                try:
                    from adapters.dapn_observation_encoder import DAPNObservationTranslator
                    _t = DAPNObservationTranslator(use_dapn=True, encoder_path=encoder_path,
                                                   feature_size=feature_size, device=self.device)
                    self.encoder = _t.shared_encoder or _t.cbs_encoder
                except Exception:
                    from adapters.transfer_encoder import ObservationEncoder, load_transfer_models
                    self.encoder, _ = load_transfer_models(encoder_path, self.device)
                self.encoder.eval()
                print(f"Loaded transfer encoder from {encoder_path}")
            except Exception as e:
                print(f"Warning: Could not load encoder from {encoder_path}: {e}")
                self.use_transfer = False

    # -------- CyberBattleSim mapping (matches the schema you posted) --------
    def from_cbs(self, obs) -> np.ndarray:
        import numpy as _np

        discovered_node_count = int(obs.get("discovered_node_count", 0))

        priv = obs.get("nodes_privilegelevel", _np.array([], dtype=_np.int32))
        if not isinstance(priv, _np.ndarray):
            priv = _np.array(priv, dtype=_np.int32) if priv is not None else _np.array([], dtype=_np.int32)
        compromised_hosts = int((priv >= 1).sum())

        discovered_hosts = discovered_node_count

        known_vulns = 0
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

        creds = int(obs.get("credential_cache_length", 0))

        steps_elapsed = 0
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

        probe_result = int(obs.get("probe_result", 0) or 0)
        escalation_val = int(obs.get("escalation", 0) or 0)
        alerts = int((probe_result == 1)) + int(escalation_val > 0)

        vec = _np.array([
            discovered_node_count,
            compromised_hosts,
            discovered_hosts,
            known_vulns,
            creds,
            steps_elapsed,
            float(dist_goal),
            alerts
        ], dtype=_np.float32)

        vec_normalized = self._normalize(vec)
        
        # If using transfer, encode to feature space
        if self.use_transfer and self.encoder is not None:
            return self._encode_to_features(vec_normalized)
        
        return vec_normalized

    # -------- Cyberwheel RedObservation vector mapping --------
    def from_cw(self, obs_vec: np.ndarray) -> np.ndarray:
        from adapters.kill_chain import cw_raw_to_red_vector

        HOST_ATTRS = 7  # type, sweeped, scanned, discovered, on_host, escalated, impacted
        obs_vec = cw_raw_to_red_vector(obs_vec)
        n = int(obs_vec.size)
        standalone_len = n % HOST_ATTRS
        max_hosts = (n - standalone_len) // HOST_ATTRS if n >= HOST_ATTRS else 0

        total_hosts_present = 0
        compromised_hosts = 0
        discovered_hosts = 0
        scanned_hosts = 0
        sweeped_hosts = 0
        escalated_count = 0
        impacted_count = 0

        for i in range(max_hosts):
            base = i * HOST_ATTRS
            chunk = obs_vec[base : base + HOST_ATTRS]
            if np.all(chunk == -1):
                continue
            total_hosts_present += 1
            discovered = int(chunk[3] == 1)
            on_host   = int(chunk[4] == 1)
            escalated = int(chunk[5] == 1)
            impacted  = int(chunk[6] == 1)
            scanned   = int(chunk[2] == 1)
            sweeped   = int(chunk[1] == 1)
            discovered_hosts += discovered
            compromised_hosts += int((on_host + escalated + impacted) > 0)
            scanned_hosts += scanned
            sweeped_hosts += sweeped
            escalated_count += escalated
            impacted_count += impacted

        # Strict semantic mapping + proxies where needed
        # known_vulns: proxy by number of scanned hosts
        known_vulns = scanned_hosts
        # credentials_found: proxy by number of escalations
        credentials_found = escalated_count
        # steps_elapsed: use standalone quadrant (1-4) as fraction of episode progress
        steps_fraction = 0.0
        if standalone_len > 0:
            try:
                quadrant = int(obs_vec[-standalone_len])
                quadrant = min(max(quadrant, 1), 4)
                steps_fraction = (quadrant - 0.5) / 4.0  # midpoint of quadrant
            except Exception:
                steps_fraction = 0.0
        steps_elapsed = steps_fraction * float(self.default_scales[5])
        # dist_to_goal: proxy by remaining fraction of hosts not impacted
        if total_hosts_present > 0:
            goal_fraction_remaining = 1.0 - (impacted_count / float(total_hosts_present))
        else:
            goal_fraction_remaining = 1.0
        dist_to_goal = goal_fraction_remaining * float(self.default_scales[6])
        # alerts: proxy by sum of escalations and impacts
        alerts = escalated_count + impacted_count

        vec = np.array([
            discovered_hosts,
            compromised_hosts,
            total_hosts_present,
            known_vulns,
            credentials_found,
            steps_elapsed,
            dist_to_goal,
            alerts
        ], dtype=np.float32)

        scales = self.default_scales.copy()
        host_scale = max(1, max_hosts)
        scales[0] = host_scale
        scales[1] = host_scale
        scales[2] = host_scale

        vec_normalized = np.clip(vec / scales, 0.0, 1.0)
        
        # If using transfer, encode to feature space
        if self.use_transfer and self.encoder is not None:
            return self._encode_to_features(vec_normalized)
        
        return vec_normalized

    def _normalize(self, vec: np.ndarray) -> np.ndarray:
        return np.clip(vec / self.default_scales, 0.0, 1.0)
    
    def _encode_to_features(self, obs: np.ndarray) -> np.ndarray:
        """
        Encode observation to feature space using transfer encoder.
        
        Args:
            obs: Normalized observation vector [OBS_DIM]
        
        Returns:
            features: Encoded features [feature_size]
        """
        if self.encoder is None:
            return obs
        
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            features = self.encoder(obs_tensor)
            return features.squeeze(0).cpu().numpy()
