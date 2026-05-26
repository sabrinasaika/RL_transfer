# adapters/unified_env.py
"""
UnifiedSecEnv  —  kill-chain intent interface for CyberWheel and CyberBattleSim.

Policy interface
────────────────
  Observation : (MAX_SLOTS × HOST_FEATURES) float32 matrix, flattened to 1-D.
                Each row describes one tracked host/node.

  Action      : Discrete(MAX_SLOTS + 1)
                  action i   → advance the kill-chain on host slot i
                  action MAX_SLOTS  → no-op

The env wrapper resolves "advance slot i" into the concrete environment action
needed at the current kill-chain phase of that slot.  The policy never sees
CW or CBS action names.

Host features (per slot)
────────────────────────
  [0] phase_norm     kill-chain phase normalised to [0, 1]
                     CW phases 0-6 / 6.0,  CBS phases 0-4 / 6.0 (same scale)
  [1] reachable      1 if the agent can target this slot right now
  [2] on_host        1 if agent is currently on this host (CW only; 0 for CBS)
  [3] is_entry       1 if this is the attacker's entry / start node
  [4] active         1 if this slot is used; 0 if padding

CW per-host phase encoding (0-6):
  0  not yet swept
  1  swept, not scanned
  2  scanned, not service-discovered
  3  discovered, agent not yet on host
  4  agent on host, not escalated
  5  escalated, not impacted
  6  impacted

CBS per-node phase encoding (mapped to same 0-6 scale):
  0  not owned (priv = 0)
  4  owned (priv >= 1)   ← maps to CW "on_host" level
  5  deeper access (priv >= 2)
  (phases 1-3 skipped — CBS has no recon sub-steps)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from adapters.kill_chain import stage_from_cbs, stage_from_cw
from adapters.reward_normalizer import RewardNormalizer
from adapters.unified_full_obs_preprocessor import UnifiedFullObsPreprocessor

# ── Constants ──────────────────────────────────────────────────────────────────
MAX_SLOTS    = 12   # maximum tracked hosts / nodes
HOST_FEATURES = 5   # phase_norm, reachable, on_host, is_entry, active
MAX_PHASE    = 6.0  # normalisation denominator (CW has phases 0-6)
PREPROCESSED_DIM = 512  # UnifiedFullObsPreprocessor output size

# CW kill-chain phase index → kill-chain action index inside CW's action space
# CW action space: num_actions = 6, kc indices 0-5
_CW_PHASE_TO_KC = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}


# ── Per-host phase helpers ─────────────────────────────────────────────────────

def _cw_host_phase(chunk: np.ndarray) -> int:
    """Return CW kill-chain phase (0-6) from a 7-element host observation chunk.
    Layout: [type, sweeped, scanned, discovered, on_host, escalated, impacted]
    """
    if len(chunk) < 7 or np.all(chunk == -1):
        return -1  # padding slot
    if int(chunk[6]): return 6  # impacted
    if int(chunk[5]): return 5  # escalated
    if int(chunk[4]): return 4  # on_host
    if int(chunk[3]): return 3  # discovered
    if int(chunk[2]): return 2  # scanned
    if int(chunk[1]): return 1  # swept
    return 0


def _cbs_node_phase(priv: int) -> int:
    """Return CBS kill-chain phase mapped to 0-6 scale from privilege level."""
    if priv >= 2: return 5   # deeper / escalated access
    if priv >= 1: return 4   # owned (foothold)
    return 0                  # not yet owned (no recon sub-steps in CBS)


# ══════════════════════════════════════════════════════════════════════════════
class UnifiedSecEnv(gym.Env):
    """Kill-chain intent Gym wrapper for CyberWheel and CyberBattleSim."""

    metadata = {"render_modes": []}

    def __init__(self, backend: str, cbs_factory=None, cw_factory=None):
        super().__init__()
        assert backend in ["cbs", "cw"], f"Unknown backend: {backend}"
        self.backend = backend
        self.rnorm   = RewardNormalizer()

        # Observation: full preprocessed 512-D (all raw features from both domains)
        # DAPN encoder (in DAPNEnvWrapper) will align these to 256-D.
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(PREPROCESSED_DIM,), dtype=np.float32
        )
        # Action: which slot to advance (+ no-op)
        self.action_space = spaces.Discrete(MAX_SLOTS + 1)

        # Preprocessor: converts raw CW/CBS obs → fixed 512-D vector
        self.preprocessor = UnifiedFullObsPreprocessor(unified_dim=PREPROCESSED_DIM)

        # Internal state
        self._raw_obs   = None
        self._slot_map  = []   # slot_idx → host_name (CW) or node_idx (CBS)
        self._step_ctr  = 0
        self._prev_stage = 0
        self._prev_kc_stage = 0
        self._slot_cw_phases = {}  # slot_idx → simulated CW phase (0-6)

        if backend == "cw":
            self._install_cyberwheel_import_workarounds()

        if backend == "cbs":
            assert cbs_factory is not None, "Provide cbs_factory"
            self.env = cbs_factory()
        else:
            assert cw_factory is not None, "Provide cw_factory"
            self.env = cw_factory()

    # ── Gym interface ──────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        raw_obs, info = self.env.reset(seed=seed, options=options)

        # Snapshot mutable CW list immediately to prevent aliasing
        if self.backend == "cw" and isinstance(raw_obs, dict):
            raw_obs = {k: (np.array(v, dtype=np.int64) if isinstance(v, list) else v)
                       for k, v in raw_obs.items()}

        self._raw_obs    = raw_obs
        self._slot_map   = []
        self._step_ctr   = 0
        self._prev_stage = 0
        self._prev_kc_stage = 0
        self._slot_cw_phases = {}  # reset per-slot simulated CW phases each episode
        self._update_slot_map()

        obs = self._build_obs()
        return obs, (info or {})

    def step(self, action: int):
        action = int(action)
        no_op_action = MAX_SLOTS

        # Cache action mask once per step (CBS only) — avoids calling the
        # expensive compute_action_mask() twice (once in _advance_cbs and
        # once in _ensure_valid_backend_action).
        self._cached_am = None
        if self.backend == "cbs" and hasattr(self.env, "compute_action_mask"):
            try:
                self._cached_am = self.env.compute_action_mask()
            except Exception:
                self._cached_am = None

        if action == no_op_action or action >= len(self._slot_map):
            backend_action = self._noop_action()
        else:
            backend_action = self._advance(action)

        backend_action = self._ensure_valid_backend_action(backend_action)

        raw_obs, raw_r, terminated, truncated, info = self.env.step(backend_action)
        # Store the actual backend action so wrappers / trace scripts can read it
        if info is None:
            info = {}
        info["cbs_action"] = backend_action

        # Snapshot CW mutable list
        if self.backend == "cw" and isinstance(raw_obs, dict):
            raw_obs = {k: (np.array(v, dtype=np.int64) if isinstance(v, list) else v)
                       for k, v in raw_obs.items()}

        self._raw_obs = raw_obs
        self._step_ctr += 1
        self._update_slot_map()

        reward = self._compute_reward(raw_r, raw_obs, info)
        obs    = self._build_obs()

        return obs, float(reward), bool(terminated), bool(truncated), (info or {})

    # ── Slot map management ────────────────────────────────────────────────────

    def _update_slot_map(self):
        """Discover newly active hosts/nodes and add them to slot_map."""
        if self.backend == "cw":
            self._update_slot_map_cw()
        else:
            self._update_slot_map_cbs()

    def _update_slot_map_cw(self):
        """Add any new active CW host names to slot_map (entry host at slot 0)."""
        red_agent = getattr(self.env, "red_agent", None)
        if red_agent is None:
            return
        action_space = getattr(red_agent, "action_space", None)
        if action_space is None:
            return
        hosts = getattr(action_space, "hosts", [])
        for h in hosts:
            if h == "available":
                continue
            if h not in self._slot_map and len(self._slot_map) < MAX_SLOTS:
                # Entry host goes to slot 0 (it's first in the list)
                self._slot_map.append(h)

    def _update_slot_map_cbs(self):
        """Add CBS node indices to slot_map (entry node at slot 0).

        At reset (slot_map empty) we eagerly pre-populate with nodes 0..MAX_SLOTS-1
        so the policy can immediately target all chain nodes via remote_vulnerability,
        mirroring how CW training always exposes multiple host slots from the start.
        After the first step we continue updating as nodes are discovered/owned.
        """
        if not isinstance(self._raw_obs, dict):
            return
        priv = self._raw_obs.get("nodes_privilegelevel")
        if priv is None:
            return
        priv = np.asarray(priv, dtype=np.int32)
        n_nodes = priv.size

        # Node 0 is always entry — ensure it's at slot 0
        if 0 not in self._slot_map and len(self._slot_map) < MAX_SLOTS:
            self._slot_map.insert(0, 0)

        # On first call (slot_map was just seeded with node 0): eagerly add all
        # potential target nodes so the policy sees a full slot map from step 0.
        # This closes the CW↔CBS distribution gap: CW training exposes discovered
        # hosts in the slot map immediately; CBS should do the same.
        if len(self._slot_map) <= 1:
            for nid in range(1, min(n_nodes, MAX_SLOTS)):
                if nid not in self._slot_map:
                    self._slot_map.append(nid)
            return

        # Subsequent calls: add any newly discovered/owned nodes not yet tracked
        disc = int(self._raw_obs.get("discovered_node_count", 1) or 1)
        for nid in range(1, n_nodes):
            if nid in self._slot_map:
                continue
            if len(self._slot_map) >= MAX_SLOTS:
                break
            if int(priv[nid]) >= 1 or nid < disc:
                self._slot_map.append(nid)

    # ── Observation builder ────────────────────────────────────────────────────

    def _build_obs(self) -> np.ndarray:
        """
        Convert raw env observation to a fixed 512-D vector via
        UnifiedFullObsPreprocessor.  Both CW and CBS raw observations are
        projected into this common space so the DAPN encoder can align them.

        CW  raw obs: dict {"red": 70001-D int vec, "blue": ...}
                     → preprocessor takes the red vector, pads/truncates to 512-D
        CBS raw obs: dict {nodes_privilegelevel, credential_cache_matrix, ...}
                     → preprocessor encodes scalars + node features into 512-D

        For CBS, simulated CW phases are injected into nodes_privilegelevel so
        the policy sees intermediate phases (0.25, 0.50, 0.75) instead of only
        0 or 1.  This matches the phase progression the policy saw during CW
        training and lets DAPN align the two domains more accurately.
          phase 0 → 0.00  (ping_sweep — not started)
          phase 1 → 0.25  (port_scan  — recon begun)
          phase 2 → 0.50  (service_disc — recon mid)
          phase 3 → 0.75  (exploit attempt — credential stolen)
          phase 4 → 1.00  (owned)
        """
        if self._raw_obs is None:
            return np.zeros(PREPROCESSED_DIM, dtype=np.float32)
        try:
            if self.backend == "cw":
                return self.preprocessor.preprocess_cw(self._raw_obs)
            else:
                # Inject simulated CW phases for unowned nodes
                if self._slot_cw_phases and isinstance(self._raw_obs, dict):
                    priv_raw = self._raw_obs.get("nodes_privilegelevel")
                    if priv_raw is not None:
                        priv = np.array(priv_raw, dtype=np.float32)
                        for slot, phase in self._slot_cw_phases.items():
                            if slot < len(self._slot_map):
                                node_id = int(self._slot_map[slot])
                                if node_id < len(priv) and priv[node_id] < 1:
                                    # Map phase 0-3 → 0.0, 0.25, 0.50, 0.75
                                    priv[node_id] = min(phase, 3) * 0.25
                        obs_copy = dict(self._raw_obs)
                        obs_copy["nodes_privilegelevel"] = priv
                        return self.preprocessor.preprocess_cbs(obs_copy)
                return self.preprocessor.preprocess_cbs(self._raw_obs)
        except Exception:
            return np.zeros(PREPROCESSED_DIM, dtype=np.float32)

    # ── Action resolution ──────────────────────────────────────────────────────

    def _advance(self, slot: int) -> dict:
        """Return the backend action that advances the kill-chain for slot."""
        if self.backend == "cw":
            return self._advance_cw(slot)
        return self._advance_cbs(slot)

    def _advance_cw(self, slot: int) -> dict:
        """
        Advance the CW host at `slot` by one kill-chain phase.

        Phase → CW kill-chain action index (kc):
          0 → ping_sweep   (kc 0)
          1 → port_scan    (kc 1)
          2 → service_disc (kc 2)
          3 → move         (kc 3)
          4 → escalate     (kc 4)
          5 → impact       (kc 5)
          6 → nothing (already impacted)
        """
        if slot >= len(self._slot_map):
            return self._noop_action()

        host_name   = self._slot_map[slot]
        red_agent   = getattr(self.env, "red_agent", None)
        if red_agent is None:
            return self._noop_action()

        action_space = getattr(red_agent, "action_space", None)
        if action_space is None:
            return self._noop_action()

        hosts        = getattr(action_space, "hosts", [])
        num_actions  = getattr(action_space, "num_actions", 6)
        max_size     = getattr(action_space, "max_size", None)

        # Current CW host phase
        phase = self._get_cw_host_phase(host_name)

        if phase >= 6:  # already impacted — no-op
            return self._noop_action()

        kc = _CW_PHASE_TO_KC.get(phase, 0)

        # Special case: the entry host starts at phase=4 (on_host=1) but
        # sweeped=0.  We must ping_sweep (kc=0) first to discover the network
        # before we can escalate.  Without this, the policy never sees other
        # hosts in the slot map and gets stuck taking no useful actions.
        if phase >= 4:
            raw = self._raw_obs
            if isinstance(raw, dict):
                raw = raw.get("red")
            if raw is not None:
                vec = np.asarray(raw, dtype=np.float32).ravel()
                try:
                    cw_idx_check = hosts.index(host_name)
                    chunk_check = vec[cw_idx_check * 7: cw_idx_check * 7 + 7]
                    if chunk_check.size >= 7 and int(chunk_check[1]) == 0:
                        kc = 0  # ping_sweep before escalating
                except (ValueError, IndexError):
                    pass

        try:
            host_idx = hosts.index(host_name)
        except (ValueError, AttributeError):
            host_idx = 0

        action_int = host_idx * num_actions + kc
        if isinstance(max_size, int) and max_size > 0:
            action_int = max(0, min(action_int, max_size - 2))  # -2 to avoid no-op slot

        # blue action: default to 0
        blue_max = 0
        try:
            blue_space = getattr(getattr(self.env, "blue_agent", None), "action_space", None)
            blue_max = getattr(blue_space, "max_size", 1) or 1
        except Exception:
            blue_max = 1
        blue_action = 0

        return {"red": int(action_int), "blue": int(blue_action)}

    def _get_cw_host_phase(self, host_name: str) -> int:
        """Get current kill-chain phase (0-6) for a named CW host."""
        raw = self._raw_obs
        if raw is None:
            return 0
        if isinstance(raw, dict):
            raw = raw.get("red")
        if raw is None:
            return 0
        vec = np.asarray(raw, dtype=np.float32).ravel()
        HOST_ATTRS = 7

        red_agent    = getattr(self.env, "red_agent", None)
        action_space = getattr(red_agent, "action_space", None) if red_agent else None
        hosts        = getattr(action_space, "hosts", []) if action_space else []

        try:
            cw_idx = hosts.index(host_name)
        except ValueError:
            return 0

        chunk = vec[cw_idx * HOST_ATTRS: cw_idx * HOST_ATTRS + HOST_ATTRS]
        if chunk.size < HOST_ATTRS:
            return 0
        return max(0, _cw_host_phase(chunk))

    def _advance_cbs(self, slot: int) -> dict:
        """
        Advance the CBS node at `slot` using a simulated CW kill-chain phase.

        Each slot tracks its own simulated CW phase (_slot_cw_phases[slot]):
          Phase 0  ping_sweep    → local_vulnerability(prev_owned, idx=0 mod n)
          Phase 1  port_scan     → local_vulnerability(prev_owned, idx=1 mod n)
          Phase 2  service_disc  → local_vulnerability(prev_owned, idx=2 mod n)
          Phase 3  exploit       → connect(prev_owned → node_id) if cred available
                                   else local_vulnerability (retry, stay at phase 3)
          Phase 4+ owned / pivot → local_vulnerability(node_id) or
                                   connect(node_id → next_unowned)

        This mirrors CW's phase progression so the policy sees the same
        phase sequence it was trained on, fixing the semantic mismatch where
        CBS was silently attacking node N+1 while the policy thought it was
        still working on node N.
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

        # Check actual CBS ownership of this node
        actually_owned = (priv is not None and node_id < priv.size and priv[node_id] >= 1)

        # Simulated CW phase for this slot
        sim_phase = self._slot_cw_phases.get(slot, 0)

        # Sync: if CBS already owns the node but sim_phase < 4, jump forward
        if actually_owned and sim_phase < 4:
            sim_phase = 4
            self._slot_cw_phases[slot] = 4

        # Local vulnerabilities available on owned nodes
        lv = am.get("local_vulnerability")
        owned_lv = []
        if lv is not None and priv is not None:
            lv_arr    = np.asarray(lv)
            lv_indices = np.argwhere(lv_arr)
            owned_lv  = [e for e in lv_indices.tolist() if int(priv[int(e[0])]) >= 1]

        # Credential cache helpers
        cred_cache = self._raw_obs.get("credential_cache_matrix") if isinstance(self._raw_obs, dict) else None
        cred_len   = int((self._raw_obs.get("credential_cache_length") or 0)
                         if isinstance(self._raw_obs, dict) else 0)
        conn       = am.get("connect")

        if sim_phase < 4:
            # ── Not yet owned ─────────────────────────────────────────────────
            # Phases 0-2: recon (ping_sweep / port_scan / service_disc)
            #   → local_vulnerability with index cycling by phase
            #   → advances phase on every call (mirrors CW one-step-per-phase)
            if sim_phase < 3:
                if owned_lv:
                    pick = sim_phase % len(owned_lv)
                    e    = owned_lv[pick]
                    self._slot_cw_phases[slot] = sim_phase + 1
                    return {"local_vulnerability": (int(e[0]), int(e[1]))}
                self._slot_cw_phases[slot] = sim_phase + 1
                return self._noop_action()

            # Phase 3: exploit → try connect; retry local_vuln if no cred yet
            if conn is not None and cred_cache is not None and cred_len > 0:
                cred_arr = np.asarray(cred_cache)
                conn_arr = np.asarray(conn)
                for cred_idx in range(min(cred_len, cred_arr.shape[0])):
                    cred_target = int(cred_arr[cred_idx, 0])
                    cred_port   = int(cred_arr[cred_idx, 1])
                    if cred_target != node_id:
                        continue
                    n_src = conn_arr.shape[0]
                    for src in range(n_src):
                        if (cred_port < conn_arr.shape[2]
                                and cred_idx < conn_arr.shape[3]
                                and conn_arr[src, cred_target, cred_port, cred_idx] > 0
                                and src != cred_target):
                            # Credential found → connect and advance to owned
                            self._slot_cw_phases[slot] = 4
                            return {"connect": (src, cred_target, cred_port, cred_idx)}

            # No credential yet — do another local_vuln (stay at phase 3)
            if owned_lv:
                pick = self._step_ctr % len(owned_lv)
                e    = owned_lv[pick]
                return {"local_vulnerability": (int(e[0]), int(e[1]))}

        else:
            # ── Owned — pivot to attack next unowned node ──────────────────────
            # Mirrors CW "on_host → pivot/escalate/impact" phases.
            # Policy should now switch to slot N+1; if it keeps calling this slot,
            # we use node_id as a launch pad rather than silently jumping ahead.
            owned_set = set(int(i) for i in np.where(priv >= 1)[0]) if priv is not None else set()

            # 1. Connect FROM node_id to an unowned neighbour (cred already stolen)
            if conn is not None and cred_cache is not None and cred_len > 0:
                cred_arr = np.asarray(cred_cache)
                conn_arr = np.asarray(conn)
                for cred_idx in range(min(cred_len, cred_arr.shape[0])):
                    cred_target = int(cred_arr[cred_idx, 0])
                    cred_port   = int(cred_arr[cred_idx, 1])
                    if cred_target in owned_set:
                        continue
                    if (node_id < conn_arr.shape[0]
                            and cred_target < conn_arr.shape[1]
                            and cred_port   < conn_arr.shape[2]
                            and cred_idx    < conn_arr.shape[3]
                            and conn_arr[node_id, cred_target, cred_port, cred_idx] > 0):
                        self._slot_cw_phases[slot] = min(sim_phase + 1, 6)
                        return {"connect": (node_id, cred_target, cred_port, cred_idx)}

            # 2. local_vulnerability on node_id — steals cred for next chain node
            if owned_lv:
                node_lv = [e for e in owned_lv if int(e[0]) == node_id]
                if node_lv:
                    pick = self._step_ctr % len(node_lv)
                    e    = node_lv[pick]
                    self._slot_cw_phases[slot] = min(sim_phase + 1, 6)
                    return {"local_vulnerability": (int(e[0]), int(e[1]))}
                # Fallback: any owned node
                pick = self._step_ctr % len(owned_lv)
                e    = owned_lv[pick]
                self._slot_cw_phases[slot] = min(sim_phase + 1, 6)
                return {"local_vulnerability": (int(e[0]), int(e[1]))}

        # Final fallback
        return self._any_valid_cbs_action(am)

    def _any_valid_cbs_action(self, am: dict) -> dict:
        """Return any valid CBS action from the mask."""
        for key, length in [("connect", 4), ("remote_vulnerability", 3), ("local_vulnerability", 2)]:
            arr = am.get(key)
            if arr is None:
                continue
            indices = np.argwhere(np.asarray(arr))
            if indices.size == 0:
                continue
            for entry in indices.tolist():
                if key in ("connect", "remote_vulnerability") and int(entry[0]) != int(entry[1]):
                    return {key: tuple(int(x) for x in entry[:length])}
            e = indices[0].tolist()
            return {key: tuple(int(x) for x in e[:length])}
        return {"local_vulnerability": (0, 0)}

    def _noop_action(self) -> dict:
        if self.backend == "cbs":
            # CBS has no true no-op; self-probe entry node as stand-in
            return {"remote_vulnerability": (0, 0, 0)}
        # CW: use max_size - 1 (the dedicated nothing slot)
        red_agent    = getattr(self.env, "red_agent", None)
        action_space = getattr(red_agent, "action_space", None) if red_agent else None
        max_size     = getattr(action_space, "max_size", None) if action_space else None
        if isinstance(max_size, int) and max_size > 0:
            return {"red": max_size - 1, "blue": 0}
        return {"red": 0, "blue": 0}

    # ── Reward ────────────────────────────────────────────────────────────────

    def _compute_reward(self, raw_r: float, raw_obs, info: dict) -> float:
        shaped = 0.0
        # Kill-chain stage-transition bonus (dense reward for progression)
        try:
            if self.backend == "cw":
                new_stage = stage_from_cw(raw_obs)
            else:
                new_stage = stage_from_cbs(raw_obs if isinstance(raw_obs, dict) else {})
            delta = max(0, new_stage - self._prev_kc_stage)
            if delta > 0:
                bonus = {1: 5.0, 2: 20.0, 3: 50.0, 4: 100.0}
                shaped += bonus.get(new_stage, float(delta) * 10.0)
            self._prev_kc_stage = new_stage
        except Exception:
            pass
        return float(raw_r) + shaped

    # ── Safety net ─────────────────────────────────────────────────────────────

    def _ensure_valid_backend_action(self, backend_action: dict) -> dict:
        """Validate and repair backend actions before submitting to env."""
        if self.backend != "cbs":
            # CW: ensure both red and blue keys exist
            if not isinstance(backend_action, dict):
                backend_action = {}
            red_agent    = getattr(self.env, "red_agent", None)
            action_space = getattr(red_agent, "action_space", None) if red_agent else None
            max_size     = getattr(action_space, "max_size", None) if action_space else None
            red_val = int(backend_action.get("red", 0))
            if isinstance(max_size, int) and max_size > 0:
                red_val = max(0, min(red_val, max_size - 1))
            backend_action = {"red": red_val, "blue": int(backend_action.get("blue", 0))}
            return backend_action

        # CBS: must be a single-key dict with correct tuple length
        valid = True
        if not isinstance(backend_action, dict) or len(backend_action) != 1:
            valid = False
        else:
            (k, v), = backend_action.items()
            expected = {"local_vulnerability": 2, "remote_vulnerability": 3, "connect": 4}
            if k not in expected or not isinstance(v, tuple) or len(v) != expected[k]:
                valid = False

        if not valid:
            backend_action = {"local_vulnerability": (0, 0)}

        # CBS mask-based repair — reuse cached mask, no second compute_action_mask() call
        try:
            am = getattr(self, "_cached_am", None)
            if am is None:
                am = self.env.compute_action_mask() if hasattr(self.env, "compute_action_mask") else None
            if am is not None and hasattr(self.env, "is_action_valid"):
                if not self.env.is_action_valid(backend_action, am):
                    backend_action = self._any_valid_cbs_action(am)
        except Exception:
            pass

        return backend_action

    # ── CW import workarounds ──────────────────────────────────────────────────

    def _install_cyberwheel_import_workarounds(self):
        import sys, types, importlib, re
        from pathlib import Path
        if "cyberwheel.utils" in sys.modules:
            return
        try:
            utils_dir = Path(__file__).resolve().parents[1] / "cyberwheel" / "cyberwheel" / "utils"
        except Exception:
            return
        if not utils_dir.exists():
            return

        def camel_to_snake(name):
            s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
            return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1).lower()

        m = types.ModuleType("cyberwheel.utils")
        m.__file__ = str(utils_dir / "__init__.py")
        m.__path__ = [str(utils_dir)]

        def __getattr__(name):
            if name == "get_service_map":
                mod = importlib.import_module("cyberwheel.utils.get_service_map")
                return getattr(mod, "get_service_map")
            try:
                mod = importlib.import_module(f"cyberwheel.utils.{name}")
                return getattr(mod, name) if hasattr(mod, name) else mod
            except Exception:
                pass
            snake = camel_to_snake(name)
            try:
                mod = importlib.import_module(f"cyberwheel.utils.{snake}")
                if hasattr(mod, name):
                    return getattr(mod, name)
                if hasattr(mod, snake):
                    return getattr(mod, snake)
            except Exception:
                pass
            for candidate in ("collections", "helpers", "types", "data_structures"):
                try:
                    mod = importlib.import_module(f"cyberwheel.utils.{candidate}")
                    if hasattr(mod, name):
                        return getattr(mod, name)
                except Exception:
                    continue
            raise AttributeError(name)

        m.__getattr__ = __getattr__
        sys.modules["cyberwheel.utils"] = m
