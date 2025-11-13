import numpy as np
import random

class ActionTranslator:
    def __init__(self):
        self.unified_actions = [
            "noop",
            "ping_sweep",
            "port_scan",
            "discovery",
            "lateral_move",
            "privilege_escalation",
            "impact",
        ]

    # -- helpers: coerce to exact tuple lengths of python ints --
    def _tupleN(self, arr_like, n, pad_val=-1):
        a = np.asarray(arr_like, dtype=np.int64).reshape(-1)
        if a.size < n:
            a = np.concatenate([a, np.full(n - a.size, pad_val, dtype=np.int64)], axis=0)
        return tuple(int(a[i]) for i in range(n))

    def to_cbs(self, action_idx, last_raw_obs=None, action_space=None, action_mask=None):
        name = self.unified_actions[action_idx]

        # Defaults
        base_connect = (0, 0, 0, -1)   # src, dst, port, cred_idx
        base_remote  = (0, 0, 0)       # src, dst, vuln_idx
        base_local   = (0, 0)          # node_idx, vuln_idx

        # Try to sample a valid-shaped action from CBS space
        try:
            sampled = action_space.sample() if action_space is not None else {}
            if "connect" in sampled:
                base_connect = self._tupleN(sampled["connect"], 4)
            if "remote_vulnerability" in sampled:
                # Coerce to 3 even if sample is shorter
                base_remote = self._tupleN(sampled["remote_vulnerability"], 3)
            if "local_vulnerability" in sampled:
                base_local = self._tupleN(sampled["local_vulnerability"], 2)
        except Exception:
            pass

        def _sample_from_mask(key, length, prefer_diff=True):
            import numpy as _np
            if not isinstance(action_mask, dict):
                return None
            arr = action_mask.get(key)
            if arr is None:
                return None
            try:
                idxs = _np.argwhere(arr)
                if idxs.size == 0:
                    return None
                # Prefer dst != src when available for connect/remote
                if prefer_diff and key in ("connect", "remote_vulnerability"):
                    _np.random.shuffle(idxs)
                    for candidate in idxs.tolist():
                        if len(candidate) >= 2 and int(candidate[0]) != int(candidate[1]):
                            return self._tupleN(candidate, length)
                idxs = idxs.tolist()
                if not idxs:
                    return None
                candidate = random.choice(idxs)
                return self._tupleN(candidate, length)
                choice = idxs[_np.random.randint(0, idxs.shape[0])].tolist()
                return self._tupleN(choice, length)
            except Exception:
                return None

        # Heuristic fallback using observation (discovered/owned nodes)
        heuristic_connect = base_connect
        heuristic_remote = base_remote
        heuristic_local = base_local
        try:
            disc = 1
            owned_indices = [0]
            total_nodes = disc
            if isinstance(last_raw_obs, dict):
                disc = int(last_raw_obs.get("discovered_node_count", 1) or 1)
                import numpy as _np
                priv = last_raw_obs.get("nodes_privilegelevel", _np.array([], dtype=_np.int32))
                if not isinstance(priv, _np.ndarray):
                    priv = _np.array(priv, dtype=_np.int32) if priv is not None else _np.array([], dtype=_np.int32)
                total_nodes = max(priv.size, disc)
                owned_indices = [int(i) for i, v in enumerate(priv.tolist()) if int(v) >= 1] or [0]
            src_owned = owned_indices[-1] if owned_indices else 0
            dest_candidate = (src_owned + 1) % max(total_nodes, 1)
            if total_nodes <= 1:
                dest_candidate = src_owned
            default_port = int(base_connect[2]) if base_connect[2] is not None else 0
            heuristic_connect = (src_owned, dest_candidate, default_port, 0)
            heuristic_remote = (src_owned, dest_candidate, int(base_remote[2]) if base_remote[2] is not None else 0)
            heuristic_local = (src_owned, int(base_local[1]) if base_local[1] is not None else 0)
        except Exception:
            pass

        sampled_connect = _sample_from_mask("connect", 4)
        base_connect = sampled_connect if sampled_connect else heuristic_connect
        sampled_remote = _sample_from_mask("remote_vulnerability", 3)
        base_remote = sampled_remote if sampled_remote else heuristic_remote
        sampled_local = _sample_from_mask("local_vulnerability", 2, prefer_diff=False)
        base_local = sampled_local if sampled_local else heuristic_local

        base_connect = self._tupleN(base_connect, 4)
        base_remote = self._tupleN(base_remote, 3)
        base_local = self._tupleN(base_local, 2)

        if name == "noop":
            return {}

        if name == "ping_sweep":
            # Use connect to explore outward where possible
            return {"connect": self._tupleN(base_connect, 4)}

        if name == "discovery":
            # Remote action to enumerate services
            return {"remote_vulnerability": self._tupleN(base_remote, 3)}

        if name == "port_scan":
            # Remote scan to enumerate services
            return {"remote_vulnerability": self._tupleN(base_remote, 3)}

        if name == "lateral_move":
            # Prefer credentialed connect to move laterally
            return {"connect": self._tupleN(base_connect, 4)}

        if name == "privilege_escalation":
            # MUST be 2-tuple
            return {"local_vulnerability": self._tupleN(base_local, 2)}

        if name == "impact":
            # benign local op; still 2-tuple
            return {"local_vulnerability": self._tupleN(base_local, 2)}


        return {}

    def to_cw(self, action_idx, state=None, red_agent=None):
        """Return action dict for CW RL red-only interface: {"red": int}.

        CW's RL wrapper expects a single discrete integer encoding both
        killchain action and target host (indexed in action_space.hosts).
        We map the 7 unified actions onto Cyberwheel's killchain ordering
        and pick a target host based on the current observation.
        """
        # Fallback: if we don't have access to the red agent, behave as before
        if red_agent is None or not hasattr(red_agent, "action_space"):
            try:
                action_int = int(action_idx)
            except Exception:
                action_int = 0
            return {"red": action_int}

        action_space = red_agent.action_space
        num_actions = getattr(action_space, "num_actions", None)
        hosts = getattr(action_space, "hosts", None)
        max_size = getattr(action_space, "max_size", None)
        if not isinstance(num_actions, int) or num_actions <= 0 or not isinstance(hosts, list):
            try:
                action_int = int(action_idx)
            except Exception:
                action_int = 0
            return {"red": action_int}

        unified_name = self.unified_actions[int(action_idx) % len(self.unified_actions)]

        # Nothing/no-op -> explicitly pick the trailing Nothing action entry
        if unified_name == "noop":
            if isinstance(max_size, int) and max_size > 0:
                return {"red": max_size - 1}
            return {"red": hosts.index(red_agent.current_host.name) * num_actions}

        killchain_map = {
            "ping_sweep": 0,
            "port_scan": 1,
            "discovery": 2,
            "lateral_move": 3,
            "privilege_escalation": 4,
            "impact": 5,
        }
        kc_index = killchain_map.get(unified_name, 0)

        # Helper: iterate over known hosts with their attributes
        obs_dict = getattr(red_agent.observation, "obs", {}) or {}
        def _host_attrs(name):
            return obs_dict.get(name, {})

        def _pick_host(predicate, fallback_current=True, fallback_any=True):
            candidates = [h for h in hosts if h != "available" and predicate(_host_attrs(h), h)]
            if candidates:
                return random.choice(candidates)
            if fallback_current and hasattr(red_agent, "current_host") and red_agent.current_host:
                try:
                    return red_agent.current_host.name
                except Exception:
                    pass
            if fallback_any:
                avail = [h for h in hosts if h != "available"]
                if avail:
                    return random.choice(avail)
            return hosts[0] if hosts else 0

        def _bool(attrs, key):
            try:
                return bool(int(attrs.get(key, 0)))
            except Exception:
                return False

        if unified_name == "ping_sweep":
            host = _pick_host(lambda attrs, _: not _bool(attrs, "sweeped"))
        elif unified_name == "port_scan":
            host = _pick_host(lambda attrs, _: _bool(attrs, "sweeped") and not _bool(attrs, "scanned"))
        elif unified_name == "discovery":
            host = _pick_host(lambda attrs, _: _bool(attrs, "scanned") and not _bool(attrs, "discovered"))
        elif unified_name == "lateral_move":
            host = _pick_host(lambda attrs, h: _bool(attrs, "discovered") and not _bool(attrs, "on_host") and _bool(attrs, "sweeped") and _bool(attrs, "scanned"))
        elif unified_name == "privilege_escalation":
            host = _pick_host(lambda attrs, _: _bool(attrs, "on_host") and not _bool(attrs, "escalated"))
        elif unified_name == "impact":
            host = _pick_host(lambda attrs, _: _bool(attrs, "on_host") and _bool(attrs, "escalated") and not _bool(attrs, "impacted"))
        else:
            host = _pick_host(lambda attrs, _: True)

        try:
            host_idx = hosts.index(host)
        except Exception:
            host_idx = 0

        action_int = host_idx * num_actions + kc_index
        # Clamp to valid range
        if isinstance(max_size, int) and max_size > 0:
            action_int = max(0, min(action_int, max_size - 1))
        return {"red": int(action_int)}
