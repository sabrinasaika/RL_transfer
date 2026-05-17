import numpy as np
import random


class ActionTranslator:
    """
    6-action unified interface — one action per CyberWheel kill-chain step.

    Actions are defined by their KILL-CHAIN EFFECT (what they achieve),
    giving a direct 1-to-1 mapping on the CW side and the best available
    mapping on the CBS side.

      idx  name         CW action               CBS action
      ---  -----------  ----------------------  ---------------------------
       0   ping_sweep   ARTPingSweep  (kc=0)    remote_vulnerability
       1   port_scan    ARTPortScan   (kc=1)    remote_vulnerability
       2   service_disc ARTDiscovery  (kc=2)    local_vulnerability  (stage 0-1)
                                                remote_vulnerability (stage 2+)
       3   move         ARTLateral    (kc=3)    connect
       4   escalate     ARTPrivEsc    (kc=4)    local_vulnerability
       5   impact       ARTImpact     (kc=5)    local_vulnerability  (proxy;
                                                CBS game ends via own_pct goal)

    CBS collisions are unavoidable: CBS has only 3 action types
    (remote_vulnerability, connect, local_vulnerability) vs CW's 6.
    Actions 0 & 1 both → remote_vulnerability; actions 2, 4, 5 all →
    local_vulnerability (but at different kill-chain stages, so the right
    CBS action fires at the right time).

    Why is ARTDiscovery (2) stage-aware in CBS?
      In CyberBattleChain, remote_vulnerability leaks NO credentials.
      Node discovery requires local_vulnerability on the start node.
      At stage 0-1, "service_disc" maps to local_vulnerability so the
      agent actually finds new nodes and leaks the first credential.
      At stage 2+, "escalate" (4) already handles local_vulnerability;
      "service_disc" falls back to remote_vulnerability as a probe.

    NOTE: changing from 3 → 6 actions requires retraining the CW policy
    (action head output size changes). DAPN encoder and observation
    pipeline are unaffected.
    """

    def __init__(self):
        self.unified_actions = [
            "ping_sweep",    # 0  CW kc=0          →  CBS remote_vulnerability
            "port_scan",     # 1  CW kc=1          →  CBS remote_vulnerability
            "service_disc",  # 2  CW kc=2          →  CBS local_vuln(s0-1) / remote_vuln(s2+)
            "move",          # 3  CW kc=3          →  CBS connect
            "escalate",      # 4  CW kc=4          →  CBS local_vulnerability
            "impact",        # 5  CW kc=5          →  CBS local_vulnerability (proxy)
            "nothing",       # 6  CW max_size-1    →  CBS self-probe (no-op)
                             #    Always valid in CW. Gives policy a safe fallback.
        ]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _tupleN(self, arr_like, n, pad_val=-1):
        a = np.asarray(arr_like, dtype=np.int64).reshape(-1)
        if a.size < n:
            a = np.concatenate([a, np.full(n - a.size, pad_val, dtype=np.int64)], axis=0)
        return tuple(int(a[i]) for i in range(n))

    def _sample_from_mask(self, action_mask, key, length, prefer_diff=True):
        if not isinstance(action_mask, dict):
            return None
        arr = action_mask.get(key)
        if arr is None:
            return None
        try:
            idxs = np.argwhere(arr)
            if idxs.size == 0:
                return None
            if prefer_diff and key in ("connect", "remote_vulnerability"):
                np.random.shuffle(idxs)
                for candidate in idxs.tolist():
                    if len(candidate) >= 2 and int(candidate[0]) != int(candidate[1]):
                        return self._tupleN(candidate, length)
            idxs = idxs.tolist()
            candidate = random.choice(idxs)
            return self._tupleN(candidate, length)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # CBS translation
    # ------------------------------------------------------------------
    def to_cbs(self, action_idx, last_raw_obs=None, action_space=None, action_mask=None):
        name = self.unified_actions[int(action_idx) % len(self.unified_actions)]

        # Defaults from action_space sample
        base_connect = (0, 0, 0, -1)
        base_remote = (0, 0, 0)
        base_local = (0, 0)
        try:
            sampled = action_space.sample() if action_space is not None else {}
            if "connect" in sampled:
                base_connect = self._tupleN(sampled["connect"], 4)
            if "remote_vulnerability" in sampled:
                base_remote = self._tupleN(sampled["remote_vulnerability"], 3)
            if "local_vulnerability" in sampled:
                base_local = self._tupleN(sampled["local_vulnerability"], 2)
        except Exception:
            pass

        # Heuristic from observation
        try:
            disc = 1
            owned_indices = [0]
            total_nodes = disc
            if isinstance(last_raw_obs, dict):
                disc = int(last_raw_obs.get("discovered_node_count", 1) or 1)
                priv = last_raw_obs.get("nodes_privilegelevel", np.array([], dtype=np.int32))
                if not isinstance(priv, np.ndarray):
                    priv = np.array(priv, dtype=np.int32) if priv is not None else np.array([], dtype=np.int32)
                total_nodes = max(priv.size, disc)
                owned_indices = [int(i) for i, v in enumerate(priv.tolist()) if int(v) >= 1] or [0]
            src_owned = owned_indices[-1] if owned_indices else 0
            dest_candidate = (src_owned + 1) % max(total_nodes, 1)
            if total_nodes <= 1:
                dest_candidate = src_owned
            default_port = int(base_connect[2]) if base_connect[2] is not None else 0
            base_connect = (src_owned, dest_candidate, default_port, 0)
            base_remote = (src_owned, dest_candidate, int(base_remote[2]) if base_remote[2] is not None else 0)
            base_local = (src_owned, int(base_local[1]) if base_local[1] is not None else 0)
        except Exception:
            pass

        def _remote():
            s = self._sample_from_mask(action_mask, "remote_vulnerability", 3)
            return {"remote_vulnerability": self._tupleN(s if s else base_remote, 3)}

        def _local():
            s = self._sample_from_mask(action_mask, "local_vulnerability", 2, prefer_diff=False)
            return {"local_vulnerability": self._tupleN(s if s else base_local, 2)}

        def _connect():
            s = self._sample_from_mask(action_mask, "connect", 4)
            return {"connect": self._tupleN(s if s else base_connect, 4)}

        # ── 0: ping_sweep — broad host scan ──────────────────────────────────
        if name == "ping_sweep":
            return _remote()

        # ── 1: port_scan — service enumeration ───────────────────────────────
        if name == "port_scan":
            return _remote()

        # ── 2: service_disc — service/node discovery ─────────────────────────
        # Stage-aware: in CyberBattleChain, node discovery happens via
        # local_vulnerability (which also leaks credentials). At stage 0-1 this
        # is the only action that actually makes progress. At stage 2+, "escalate"
        # handles local_vulnerability; this falls back to remote probing.
        if name == "service_disc":
            stage = 0
            try:
                if isinstance(last_raw_obs, dict):
                    from adapters.kill_chain import stage_from_cbs
                    stage = stage_from_cbs(last_raw_obs)
            except Exception:
                stage = 0
            if stage <= 1:
                # local_vulnerability: discovers neighbours + leaks first credential
                s = self._sample_from_mask(action_mask, "local_vulnerability", 2,
                                           prefer_diff=False)
                if s:
                    return {"local_vulnerability": self._tupleN(s, 2)}
                return _remote()   # fallback if no local available
            return _remote()       # stage 2+: just probe

        # ── 3: move — lateral movement to a new node ─────────────────────────
        if name == "move":
            return _connect()

        # ── 4: escalate — privilege escalation / credential mining ───────────
        if name == "escalate":
            return _local()

        # ── 5: impact — final objective (CBS: proxy via local_vuln) ──────────
        # CBS game ends automatically when own_atleast_percent is reached.
        # Using local_vulnerability keeps pressure on owned nodes.
        if name == "impact":
            return _local()

        # ── 6: nothing — no-op (CW: always valid; CBS: self-probe of start node) ──
        # CBS has no true no-op. Probe the start node (index 0) as a safe stand-in;
        # it has minimal effect and keeps the episode running.
        if name == "nothing":
            return {"remote_vulnerability": (0, 0, 0)}

        # Fallback — should never reach here
        return _local()

    # ------------------------------------------------------------------
    # CW translation
    # ------------------------------------------------------------------
    def to_cw(self, action_idx, state=None, red_agent=None):
        """
        Map unified action to CW's (host_idx * num_actions + kc_index) encoding.

        recon    → ping_sweep(0) / port_scan(1) / discovery(2)  chosen by host state
        move     → lateral_move(3)
        escalate → privilege_escalation(4) / impact(5)          chosen by host state
        """
        if red_agent is None or not hasattr(red_agent, "action_space"):
            return {"red": int(action_idx)}

        action_space = red_agent.action_space
        num_actions = getattr(action_space, "num_actions", None)
        hosts = getattr(action_space, "hosts", None)
        max_size = getattr(action_space, "max_size", None)

        if not isinstance(num_actions, int) or num_actions <= 0 or not isinstance(hosts, list):
            return {"red": int(action_idx)}

        name = self.unified_actions[int(action_idx) % len(self.unified_actions)]

        obs_dict = getattr(getattr(red_agent, "observation", None), "obs", {}) or {}

        def _host_attrs(h):
            return obs_dict.get(h, {})

        def _bool(attrs, key):
            try:
                return bool(int(attrs.get(key, 0)))
            except Exception:
                return False

        def _pick(predicate, fallback_any=True):
            candidates = [h for h in hosts if h != "available" and predicate(_host_attrs(h))]
            if candidates:
                return random.choice(candidates)
            if fallback_any:
                avail = [h for h in hosts if h != "available"]
                if avail:
                    return random.choice(avail)
            return hosts[0] if hosts else None

        # Each unified action maps directly to one CW kill-chain index (kc).
        # Host selection targets the most appropriate host for each step.

        if name == "ping_sweep":        # kc=0 — sweep hosts not yet pinged
            host = _pick(lambda a: not _bool(a, "sweeped")) or _pick(lambda _: True)
            kc = 0

        elif name == "port_scan":       # kc=1 — scan hosts already swept
            host = (_pick(lambda a: _bool(a, "sweeped") and not _bool(a, "scanned"))
                    or _pick(lambda _: True))
            kc = 1

        elif name == "service_disc":    # kc=2 — discover services on scanned hosts
            host = (_pick(lambda a: _bool(a, "scanned") and not _bool(a, "discovered"))
                    or _pick(lambda _: True))
            kc = 2

        elif name == "move":            # kc=3 — move to a discovered but unvisited host
            host = (_pick(lambda a: _bool(a, "discovered") and not _bool(a, "on_host"))
                    or _pick(lambda _: True))
            kc = 3

        elif name == "escalate":        # kc=4 — escalate on the current host
            host = (_pick(lambda a: _bool(a, "on_host") and not _bool(a, "escalated"))
                    or _pick(lambda _: True))
            kc = 4

        elif name == "impact":          # kc=5 — impact on the current, already-escalated host
            # CW mask ONLY allows kc=4/5 on the current host (on_host==1).
            # Must prioritise: on_host=1 AND escalated=1 AND not yet impacted.
            host = (_pick(lambda a: _bool(a, "on_host") and _bool(a, "escalated") and not _bool(a, "impacted"))
                    or _pick(lambda a: _bool(a, "on_host"))   # current host even if not yet escalated
                    or _pick(lambda _: True))
            kc = 5

        elif name == "nothing":         # global no-op — max_size - 1 (always valid in CW)
            if isinstance(max_size, int) and max_size > 0:
                return {"red": max_size - 1}
            return {"red": int(action_idx)}

        else:                           # unknown action — should never be reached
            host = _pick(lambda _: True)
            kc = 0

        try:
            host_idx = hosts.index(host)
        except Exception:
            host_idx = 0

        action_int = host_idx * num_actions + kc
        if isinstance(max_size, int) and max_size > 0:
            action_int = max(0, min(action_int, max_size - 1))
        return {"red": int(action_int)}
