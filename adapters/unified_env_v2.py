# adapters/unified_env_v2.py
"""
UnifiedSecEnvV2  —  v2 kill-chain wrapper that uses REAL CBS state instead of a
simulated CW phase counter to decide which CBS action to take.

Design differences from v1 (UnifiedSecEnv):
  - _slot_cw_phases is zeroed out at reset and NEVER written during an episode.
  - _build_obs  (CBS path) derives phase values from live CBS privilege/cred state
    instead of injecting the fake simulated CW phases.
  - _advance_cbs resolves the action from actual CBS ownership + credential cache:
      Case 1: node already owned  → local_vulnerability(node_id, ...)   (pivot)
      Case 2: creds in CBS cache  → connect(src, node_id, port, cred)   (exploit)
      Case 3: otherwise           → local_vulnerability(frontier, ...)  (probe)
    where frontier = max(owned_set), i.e. the last-owned node.

Everything else (CW backend, reward, slot-map, safety-net, etc.) is inherited
unchanged from UnifiedSecEnv.
"""

import numpy as np
from adapters.unified_env import UnifiedSecEnv


class UnifiedSecEnvV2(UnifiedSecEnv):
    """Kill-chain wrapper that resolves CBS actions from real CBS state."""

    # ── Reset ──────────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        # v2 never uses the simulated phase counter; clear it to make that explicit.
        self._slot_cw_phases = {}
        return obs, info

    # ── CBS credential-cache helper ────────────────────────────────────────────

    def _get_cred_nodes(self) -> set:
        """Return the set of node IDs that have credentials in the CBS cache."""
        if not isinstance(self._raw_obs, dict):
            return set()
        cred_cache = self._raw_obs.get("credential_cache_matrix")
        cred_len   = int((self._raw_obs.get("credential_cache_length") or 0))
        if cred_cache is None or cred_len == 0:
            return set()
        try:
            arr = np.asarray(cred_cache, dtype=np.int32)  # shape (N, 2): [node_id, port_id]
            return set(int(arr[i, 0]) for i in range(min(cred_len, arr.shape[0])))
        except Exception:
            return set()

    # ── Observation builder (override CBS path only) ───────────────────────────

    def _build_obs(self) -> np.ndarray:
        """
        For CBS backend: derive phase float for each node from real CBS state:
          - priv >= 1  → 1.00  (owned)
          - creds in CBS cache → 0.75  (credential stolen, ready to exploit)
          - otherwise  → 0.00  (not yet probed)

        For CW backend: fall through to v1 (preprocess_cw unchanged).
        """
        from adapters.unified_env import PREPROCESSED_DIM

        if self._raw_obs is None:
            return np.zeros(PREPROCESSED_DIM, dtype=np.float32)

        try:
            if self.backend == "cw":
                return self.preprocessor.preprocess_cw(self._raw_obs)

            # CBS path — build phase vector from real state
            if not isinstance(self._raw_obs, dict):
                return self.preprocessor.preprocess_cbs(self._raw_obs)

            priv_raw = self._raw_obs.get("nodes_privilegelevel")
            if priv_raw is None:
                return self.preprocessor.preprocess_cbs(self._raw_obs)

            priv      = np.asarray(priv_raw, dtype=np.float32)
            cred_nodes = self._get_cred_nodes()

            phase_vec = np.zeros(len(priv), dtype=np.float32)
            for nid in range(len(priv)):
                if priv[nid] >= 1:
                    phase_vec[nid] = 1.00
                elif nid in cred_nodes:
                    phase_vec[nid] = 0.75
                else:
                    phase_vec[nid] = 0.00

            obs_copy = dict(self._raw_obs)
            obs_copy["nodes_privilegelevel"] = phase_vec
            return self.preprocessor.preprocess_cbs(obs_copy)

        except Exception:
            return np.zeros(PREPROCESSED_DIM, dtype=np.float32)

    # ── Action resolver (override CBS path only) ───────────────────────────────

    def _advance_cbs(self, slot: int) -> dict:
        """
        Resolve the CBS action for the chosen slot using real CBS state.

        Case 1: node_id is owned  → local_vulnerability(node_id, ...) to steal
                                     credentials for the next hop (pivot)
        Case 2: creds for node_id are in CBS cache  → connect(src, node_id, port, cred)
        Case 3: otherwise  → local_vulnerability(frontier, ...) where
                             frontier = max(owned_set) (latest owned node)

        Final fallback: _any_valid_cbs_action(am).
        """
        if slot >= len(self._slot_map):
            return self._noop_action()

        node_id = int(self._slot_map[slot])

        am = getattr(self, "_cached_am", None)
        if am is None:
            try:
                am = self.env.compute_action_mask() if hasattr(self.env, "compute_action_mask") else None
            except Exception:
                pass
        if am is None:
            return {"local_vulnerability": (0, 0)}

        # Privilege levels
        priv = None
        if isinstance(self._raw_obs, dict):
            priv_raw = self._raw_obs.get("nodes_privilegelevel")
            if priv_raw is not None:
                priv = np.asarray(priv_raw, dtype=np.int32)

        # Owned set and local_vulnerability entries available from owned nodes
        owned_set = set()
        if priv is not None:
            owned_set = set(int(i) for i in np.where(priv >= 1)[0])

        lv = am.get("local_vulnerability")
        owned_lv = []
        if lv is not None and priv is not None:
            lv_arr    = np.asarray(lv)
            lv_indices = np.argwhere(lv_arr)
            owned_lv  = [e for e in lv_indices.tolist() if int(priv[int(e[0])]) >= 1]

        # Credential cache
        cred_cache = self._raw_obs.get("credential_cache_matrix") if isinstance(self._raw_obs, dict) else None
        cred_len   = int((self._raw_obs.get("credential_cache_length") or 0)
                         if isinstance(self._raw_obs, dict) else 0)
        conn       = am.get("connect")

        # ── Case 1: node_id is already owned → pivot (local_vulnerability on node_id) ──
        if node_id in owned_set:
            if owned_lv:
                # Prefer vulns originating from node_id itself
                node_lv = [e for e in owned_lv if int(e[0]) == node_id]
                if node_lv:
                    pick = self._step_ctr % len(node_lv)
                    e    = node_lv[pick]
                    return {"local_vulnerability": (int(e[0]), int(e[1]))}
                # Fallback: any owned-node vuln
                pick = self._step_ctr % len(owned_lv)
                e    = owned_lv[pick]
                return {"local_vulnerability": (int(e[0]), int(e[1]))}

        # ── Case 2: creds for node_id exist → connect ─────────────────────────
        if conn is not None and cred_cache is not None and cred_len > 0:
            cred_arr = np.asarray(cred_cache)
            conn_arr = np.asarray(conn)
            for cred_idx in range(min(cred_len, cred_arr.shape[0])):
                cred_target = int(cred_arr[cred_idx, 0])
                cred_port   = int(cred_arr[cred_idx, 1])
                if cred_target != node_id:
                    continue
                # Find a valid owned source node
                for src in owned_set:
                    if (src < conn_arr.shape[0]
                            and cred_target < conn_arr.shape[1]
                            and cred_port   < conn_arr.shape[2]
                            and cred_idx    < conn_arr.shape[3]
                            and conn_arr[src, cred_target, cred_port, cred_idx] > 0
                            and src != cred_target):
                        return {"connect": (src, cred_target, cred_port, cred_idx)}

        # ── Case 3: no creds → probe frontier (latest owned node) ─────────────
        if owned_set and owned_lv:
            frontier = max(owned_set)
            frontier_lv = [e for e in owned_lv if int(e[0]) == frontier]
            if frontier_lv:
                pick = self._step_ctr % len(frontier_lv)
                e    = frontier_lv[pick]
                return {"local_vulnerability": (int(e[0]), int(e[1]))}
            # Fallback: any owned-node local_vulnerability
            pick = self._step_ctr % len(owned_lv)
            e    = owned_lv[pick]
            return {"local_vulnerability": (int(e[0]), int(e[1]))}

        # Final fallback
        return self._any_valid_cbs_action(am)
