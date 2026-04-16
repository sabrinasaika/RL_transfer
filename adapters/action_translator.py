import numpy as np
import random


class ActionTranslator:
    """
    3-action unified interface valid in both CyberWheel and CyberBattleSim.

    Action semantics:
      0  recon    — gather information (sweep / scan / enumerate)
      1  move     — reach a new node   (lateral movement / connect)
      2  escalate — gain more power or exfiltrate (priv-esc / impact / local exploit)

    CBS native mapping:
      recon    → remote_vulnerability  (enumerate a discovered node)
                 fallback: connect     (reach out when no remote vuln available)
      move     → connect               (move to a discovered node)
      escalate → local_vulnerability   (exploit on an owned node)

    CW native mapping:
      recon    → ping_sweep (kc=0) / port_scan (kc=1) / discovery (kc=2)
                 chosen by what the next unfinished recon step is
      move     → lateral_move (kc=3)
      escalate → privilege_escalation (kc=4) / impact (kc=5)
                 chosen by whether any host still needs escalation
    """

    def __init__(self):
        self.unified_actions = ["recon", "move", "escalate"]

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

        if name == "recon":
            # recon = remote probe/scan — always remote_vulnerability, never connect
            sampled = self._sample_from_mask(action_mask, "remote_vulnerability", 3)
            if sampled:
                return {"remote_vulnerability": self._tupleN(sampled, 3)}
            return {"remote_vulnerability": self._tupleN(base_remote, 3)}

        if name == "move":
            sampled = self._sample_from_mask(action_mask, "connect", 4)
            if sampled:
                return {"connect": self._tupleN(sampled, 4)}
            return {"connect": self._tupleN(base_connect, 4)}

        if name == "escalate":
            sampled = self._sample_from_mask(action_mask, "local_vulnerability", 2, prefer_diff=False)
            if sampled:
                return {"local_vulnerability": self._tupleN(sampled, 2)}
            return {"local_vulnerability": self._tupleN(base_local, 2)}

        # Fallback — should never reach here
        return {"local_vulnerability": self._tupleN(base_local, 2)}

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

        if name == "recon":
            # Pick the earliest unfinished recon step across any host
            host = _pick(lambda a: not _bool(a, "sweeped"))
            if host is None:
                host = _pick(lambda a: _bool(a, "sweeped") and not _bool(a, "scanned"))
            if host is None:
                host = _pick(lambda a: _bool(a, "scanned") and not _bool(a, "discovered"))
            if host is None:
                host = _pick(lambda _: True)
            # Choose kc step
            attrs = _host_attrs(host) if host else {}
            if not _bool(attrs, "sweeped"):
                kc = 0
            elif not _bool(attrs, "scanned"):
                kc = 1
            else:
                kc = 2

        elif name == "move":
            host = _pick(lambda a: _bool(a, "discovered") and not _bool(a, "on_host"))
            if host is None:
                host = _pick(lambda _: True)
            kc = 3

        else:  # escalate
            host = _pick(lambda a: _bool(a, "on_host") and not _bool(a, "escalated"))
            if host is None:
                host = _pick(lambda a: _bool(a, "escalated") and not _bool(a, "impacted"))
            if host is None:
                host = _pick(lambda _: True)
            attrs = _host_attrs(host) if host else {}
            kc = 4 if not _bool(attrs, "escalated") else 5

        try:
            host_idx = hosts.index(host)
        except Exception:
            host_idx = 0

        action_int = host_idx * num_actions + kc
        if isinstance(max_size, int) and max_size > 0:
            action_int = max(0, min(action_int, max_size - 1))
        return {"red": int(action_int)}
