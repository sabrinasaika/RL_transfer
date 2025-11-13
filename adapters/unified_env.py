# adapters/unified_env.py
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from adapters.observation_translator import ObservationTranslator, OBS_DIM
from adapters.action_translator import ActionTranslator
from adapters.reward_normalizer import RewardNormalizer


class UnifiedSecEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, backend, cbs_factory=None, cw_factory=None):
        super().__init__()
        assert backend in ["cbs", "cw"]
        self.backend = backend

        # Translators and reward normalizer
        self.obs_t = ObservationTranslator()
        self.act_t = ActionTranslator()
        self.rnorm = RewardNormalizer()
        self._reset_progress_trackers()
        self._last_unified_action = None
        self._last_backend_action = None

        # Observation space
        use_multi = False
        if backend == "cbs":
            try:
                import os as _os
                use_multi = _os.environ.get("CBS_MULTI_INPUT", "1") == "1"
            except Exception:
                use_multi = True
            if use_multi:
                # Multi-input: normalized obs + unified action mask (7)
                self.observation_space = spaces.Dict({
                    "obs": spaces.Box(low=0.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32),
                    "mask": spaces.Box(low=0.0, high=1.0, shape=(len(ActionTranslator().unified_actions),), dtype=np.float32)
                })
            else:
                self.observation_space = spaces.Box(
                    low=0.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32
                )
        else:
            # Cyberwheel path remains 1D vector
            self.observation_space = spaces.Box(
                low=0.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32
            )
        self.action_space = spaces.Discrete(len(self.act_t.unified_actions))

        self._last_raw_obs = None
        self._dbg_step = 0

        
        if backend == "cw":
            self._install_cyberwheel_import_workarounds()

        if backend == "cbs":
            assert cbs_factory is not None, "Provide cbs_factory in config/env_builders.py"
            self.env = cbs_factory()
        else:
            assert cw_factory is not None, "Provide cw_factory in config/env_builders.py"
            self.env = cw_factory()

    
    def reset(self, seed=None, options=None):
        # Seed numpy for any local sampling you do here
        if seed is not None:
            np.random.seed(seed)

        raw_obs, info = self.env.reset(seed=seed, options=options)
        self._last_raw_obs = raw_obs
        self._reset_progress_trackers()
        if isinstance(raw_obs, dict) and self.backend == "cbs":
            self._sync_progress_counters(raw_obs)
        obs = self._xlate_obs(raw_obs)

        # Determine multi-input mode consistently with __init__
        _use_multi = False
        if self.backend == "cbs":
            try:
                import os as _os
                _use_multi = _os.environ.get("CBS_MULTI_INPUT", "1") == "1"
            except Exception:
                _use_multi = True

        if self.backend == "cbs" and _use_multi:
            mask = self._compute_unified_mask()
            out = {"obs": np.asarray(obs, dtype=np.float32), "mask": mask}
        else:
            out = np.asarray(obs, dtype=np.float32)

        return out, (info or {})

    def step(self, action_idx):
        # Optional unified-action masking at inference (does not change weights)
        try:
            import os as _os
            do_mask = _os.environ.get("EVAL_MASK_POLICY", "0") == "1"
        except Exception:
            do_mask = False
        if self.backend == "cbs" and do_mask:
            mask = self._compute_unified_mask()
            # If chosen action is invalid, pick a random valid unified action
            try:
                if int(mask[int(action_idx)]) == 0 and mask.sum() > 0:
                    valid_idxs = np.where(mask > 0.0)[0]
                    # Simple preference: remote > connect > local > noop when available
                    names = self.act_t.unified_actions
                    pref_order = [
                        "remote_vulnerability",  # covers port_scan/lateral_move mapped types
                        "connect",
                        "local_vulnerability",
                        "noop",
                    ]
                    # Map names to indices present in valid_idxs
                    name_to_idx = {n: i for i, n in enumerate(names)}
                    pick = None
                    for name in pref_order:
                        # find any matching unified action label among valid_idxs
                        for vi in valid_idxs:
                            if names[int(vi)] == name:
                                pick = int(vi)
                                break
                        if pick is not None:
                            break
                    if pick is None:
                        pick = int(np.random.choice(valid_idxs))
                    action_idx = pick
            except Exception:
                pass

        self._last_unified_action = int(action_idx)
        backend_action = self._xlate_action(action_idx)
        # Optional explore booster: for the first BOOST_STEPS steps, sample a random valid CBS action
        try:
            import os as _os, numpy as _np
            boost_n = int(_os.environ.get("BOOST_STEPS", "0") or 0)
            if self.backend == "cbs" and boost_n > 0:
                if self._dbg_step < boost_n:
                    am = self.env.compute_action_mask() if hasattr(self.env, "compute_action_mask") else None
                    if am is not None:
                        # Prefer connect, then remote, then local; sample a random valid index
                        def sample_valid(key):
                            try:
                                idxs = _np.argwhere(am[key])
                                if idxs.size == 0:
                                    return None
                                choice = idxs[_np.random.randint(0, idxs.shape[0])].tolist()
                                if key == "connect":
                                    li, ri, pi, ci = choice
                                    return {key: (int(li), int(ri), int(pi), int(ci))}
                                if key == "remote_vulnerability":
                                    li, ri, vi = choice
                                    return {key: (int(li), int(ri), int(vi))}
                                if key == "local_vulnerability":
                                    li, vi = choice
                                    return {key: (int(li), int(vi))}
                            except Exception:
                                return None
                            return None
                        boosted = sample_valid("connect") or sample_valid("remote_vulnerability") or sample_valid("local_vulnerability")
                        if boosted is not None:
                            backend_action = boosted
        except Exception:
            pass
        backend_action = self._ensure_valid_backend_action(backend_action)
        self._last_backend_action = backend_action

        # Optional debug: print per-step details for CBS/CW
        try:
            import os as _os
            dbg_on = _os.environ.get("STEP_DEBUG", "0") == "1"
            dbg_every = int(_os.environ.get("STEP_LOG_EVERY", "50") or 50)
            if dbg_every < 1:
                dbg_every = 1
        except Exception:
            dbg_on = False
            dbg_every = 50
        if dbg_on:
            self._dbg_step += 1
            if self._dbg_step % dbg_every == 0:
                try:
                    if self.backend == "cbs":
                        # Extract simple telemetry from last raw obs
                        disc_total, new_disc, owned_total, new_owned = self._progress_snapshot(self._last_raw_obs)
                        # Validate via CBS if available
                        valid = None
                        try:
                            if hasattr(self.env, "is_action_valid"):
                                am = self.env.compute_action_mask() if hasattr(self.env, "compute_action_mask") else None
                                valid = self.env.is_action_valid(backend_action, am)
                                if am is not None:
                                    import numpy as _np
                                    connect_count = int(_np.sum(am.get("connect"))) if isinstance(am.get("connect"), _np.ndarray) else 0
                                    remote_count = int(_np.sum(am.get("remote_vulnerability"))) if isinstance(am.get("remote_vulnerability"), _np.ndarray) else 0
                                    local_count = int(_np.sum(am.get("local_vulnerability"))) if isinstance(am.get("local_vulnerability"), _np.ndarray) else 0
                                else:
                                    connect_count = remote_count = local_count = -1
                        except Exception:
                            valid = None
                            connect_count = remote_count = local_count = -1
                        print(
                            f"[CBS dbg pre] step={self._dbg_step} ua={self._last_unified_action} backend={backend_action} "
                            f"valid={valid} disc_total={disc_total} new_disc={new_disc} owned_total={owned_total} new_owned={new_owned} "
                            f"mask(connect)={connect_count} mask(remote)={remote_count} mask(local)={local_count}",
                            flush=True,
                        )
                    else:
                        print(f"[CW dbg] step={self._dbg_step} action={backend_action}", flush=True)
                except Exception:
                    pass

        raw_obs, raw_r, terminated, truncated, info = self.env.step(backend_action)
        self._last_raw_obs = raw_obs
        self._step_counter += 1

        if dbg_on and self.backend == "cbs":
            disc_total, new_disc, owned_total, new_owned = self._progress_snapshot(raw_obs)
            print(
                f"[CBS dbg post] step={self._dbg_step} ua={self._last_unified_action} backend={backend_action} "
                f"disc_total={disc_total} new_disc={new_disc} owned_total={owned_total} new_owned={new_owned} reward={raw_r}",
                flush=True,
            )

        obs = self._xlate_obs(raw_obs)
        r = self._xlate_reward(raw_r, info)

        # Types must match Gym
        # Same multi-input gating as in reset
        _use_multi = False
        if self.backend == "cbs":
            try:
                import os as _os
                _use_multi = _os.environ.get("CBS_MULTI_INPUT", "1") == "1"
            except Exception:
                _use_multi = True

        if self.backend == "cbs" and _use_multi:
            mask = self._compute_unified_mask()
            out_obs = {"obs": np.asarray(obs, dtype=np.float32), "mask": mask}
        else:
            out_obs = np.asarray(obs, dtype=np.float32)

        return (out_obs, float(r), bool(terminated), bool(truncated), (info or {}))

    
    # Translations
    def _xlate_obs(self, raw):
        if self.backend == "cbs":
            return self.obs_t.from_cbs(raw)
        else:
            return self.obs_t.from_cw(raw)

    def _xlate_action(self, a):
        if self.backend == "cbs":
            # Provide CBS action_space so translator can produce valid-shaped tuples
            try:
                action_space = getattr(self.env, "action_space", None)
            except Exception:
                action_space = None
            try:
                action_mask = self.env.compute_action_mask() if hasattr(self.env, "compute_action_mask") else None
            except Exception:
                action_mask = None
            return self.act_t.to_cbs(a, last_raw_obs=self._last_raw_obs, action_space=action_space, action_mask=action_mask)
        # CyberWheel branch: expose underlying red_agent so translator can pick host indices
        red_agent = getattr(self.env, "red_agent", None)
        return self.act_t.to_cw(a, state=self._last_raw_obs, red_agent=red_agent)

    def _xlate_reward(self, r, info):
          # Optional shaping for CBS to provide fast non-zero signals during evaluation
          shaped_bonus = 0.0
          try:
              import os as _os
              use_shaping = _os.environ.get("EVAL_SHAPED_REWARD", "0") == "1"
          except Exception:
              use_shaping = False
          if use_shaping and self.backend == "cbs":
              raw = self._last_raw_obs if isinstance(self._last_raw_obs, dict) else {}
              try:
                  disc = int(raw.get("newly_discovered_nodes_count", 0) or 0)
              except Exception:
                  disc = 0
              try:
                  disc_total = int(raw.get("discovered_node_count", 0) or 0)
              except Exception:
                  disc_total = 0
              try:
                  lat = int(raw.get("lateral_move", 0) or 0)
              except Exception:
                  lat = 0
              try:
                  esc = int(raw.get("escalation", 0) or 0)
              except Exception:
                  esc = 0
              try:
                  probe = int(raw.get("probe_result", 0) or 0)  # 1=failed, 2=success
              except Exception:
                  probe = 0
              # Coefficients configurable via env
              try:
                  import math as _math
                  c_disc = float(_os.environ.get("SHAPE_DISCOVERY", "1.0") or 1.0)
                  c_disc_total = float(_os.environ.get("SHAPE_DISCOVERY_TOTAL", "0.2") or 0.2)
                  c_lat = float(_os.environ.get("SHAPE_LATERAL", "2.0") or 2.0)
                  c_esc = float(_os.environ.get("SHAPE_ESC", "5.0") or 5.0)
                  c_probe = float(_os.environ.get("SHAPE_PROBE", "0.2") or 0.2)
              except Exception:
                  c_disc, c_disc_total, c_lat, c_esc, c_probe = 1.0, 0.2, 2.0, 5.0, 0.2
              probe_bonus = c_probe * (1.0 if probe > 0 else 0.0) + (c_probe * 4.0 if probe == 2 else 0.0)
              # Reward per new discovery + a small weight on total discovered
              prev_disc_total = getattr(self, "_prev_disc_total", 0)
              new_disc = max(0, disc_total - prev_disc_total)
              self._prev_disc_total = disc_total

              owned_total = 0
              try:
                  import numpy as _np
                  priv = raw.get("nodes_privilegelevel", _np.array([], dtype=_np.int32))
                  if not isinstance(priv, _np.ndarray):
                      priv = _np.array(priv, dtype=_np.int32) if priv is not None else _np.array([], dtype=_np.int32)
                  owned_total = int((priv >= 1).sum())
              except Exception:
                  owned_total = 0
              prev_owned_total = getattr(self, "_prev_owned_total", 0)
              new_owned = max(0, owned_total - prev_owned_total)
              self._prev_owned_total = owned_total

              shaped_bonus = (
                  (c_disc * float(new_disc))
                  + (c_disc_total * float(new_disc))
                  + (c_lat * float(new_owned))
                  + (c_lat * (1.0 if lat == 1 else 0.0))
                  + (c_esc * (1.0 if esc > 0 else 0.0))
                  + probe_bonus
              )
              try:
                  act_idx = int(self._last_unified_action) if self._last_unified_action is not None else None
              except Exception:
                  act_idx = None
              if act_idx is not None:
                  action_shape = {
                      0: float(_os.environ.get("SHAPE_ACTION_NOOP", "0.0") or 0.0),
                      1: float(_os.environ.get("SHAPE_ACTION_PING", "0.0") or 0.0),
                      2: float(_os.environ.get("SHAPE_ACTION_PORT", "0.0") or 0.0),
                      3: float(_os.environ.get("SHAPE_ACTION_DISC", "0.0") or 0.0),
                      4: float(_os.environ.get("SHAPE_ACTION_LATERAL", "0.0") or 0.0),
                      5: float(_os.environ.get("SHAPE_ACTION_ESC", "0.0") or 0.0),
                      6: float(_os.environ.get("SHAPE_ACTION_IMPACT", "0.0") or 0.0),
                  }
                  shaped_bonus += action_shape.get(act_idx, 0.0)
              step_cost = float(_os.environ.get("SHAPE_STEP_COST", "0.0") or 0.0)
              if step_cost != 0.0:
                  shaped_bonus -= step_cost
              last_probe_step = getattr(self, "_last_probe_success_step", -1)
              if probe == 2 and self._step_counter != last_probe_step:
                  shaped_bonus += c_probe * 2.0
                  self._last_probe_success_step = self._step_counter
              else:
                  self._last_probe_success_step = last_probe_step
          return float(r) + float(shaped_bonus)

    
    # Safety nets
   
    def _ensure_valid_backend_action(self, backend_action):
        """
        CyberBattleSim: requires exactly one key in the action dict.
        Cyberwheel: return a single definitive action; fallback to 'noop' if needed.
        """
        cbs_fallback = {"local_vulnerability": (0, 0)}
        cw_fallback = {"noop": 0}  # adjust if your CW expects a different minimal action

        if self.backend == "cbs":
            valid_shape = True
            if not isinstance(backend_action, dict) or len(backend_action) != 1:
                valid_shape = False
            else:
                (k, v), = backend_action.items()
                try:
                    if k == "local_vulnerability" and not (isinstance(v, tuple) and len(v) == 2):
                        valid_shape = False
                    if k == "remote_vulnerability" and not (isinstance(v, tuple) and len(v) == 3):
                        valid_shape = False
                    if k == "escalate_privilege" and not (isinstance(v, tuple) and len(v) == 2):
                        valid_shape = False
                except Exception:
                    valid_shape = False

            if not valid_shape:
                backend_action = cbs_fallback

            # Try CBS mask-based repair; prefer connect-first to expand frontier
            try:
                import os as _os
                do_repair = _os.environ.get("CBS_REPAIR", "1") == "1"
                action_mask = self.env.compute_action_mask() if hasattr(self.env, "compute_action_mask") else None
                if do_repair and action_mask is not None and hasattr(self.env, "is_action_valid"):
                    # Always try to turn self-connect into dst!=src when possible
                    try:
                        (ak, av), = backend_action.items()
                        if ak == "connect" and isinstance(av, tuple) and len(av) == 4:
                            li, ri, pi, ci = [int(x) for x in av]
                            # If dst==src, try to pick any valid dst!=src from the mask
                            if int(ri) == int(li):
                                import numpy as _np
                                valid_idxs = _np.argwhere(action_mask["connect"])  # shape (N,4)
                                for vi in valid_idxs.tolist():
                                    li2, ri2, pi2, ci2 = [int(x) for x in vi]
                                    if li2 != ri2:
                                        backend_action = {"connect": (li2, ri2, pi2, ci2)}
                                        break
                    except Exception:
                        pass

                    if not self.env.is_action_valid(backend_action, action_mask):
                        import numpy as _np
                        def first_valid(key):
                            try:
                                if key == "connect":
                                    local_ids, remote_ids, ports, creds = _np.argwhere(action_mask[key]).T
                                    # prefer dst != src when available
                                    for li, ri, pi, ci in zip(local_ids.tolist(), remote_ids.tolist(), ports.tolist(), creds.tolist()):
                                        if int(ri) != int(li):
                                            return {key: (int(li), int(ri), int(pi), int(ci))}
                                    return {key: (int(local_ids[0]), int(remote_ids[0]), int(ports[0]), int(creds[0]))}
                                if key == "remote_vulnerability":
                                    local_ids, remote_ids, vuln_ids = _np.argwhere(action_mask[key]).T
                                    for li, ri, vi in zip(local_ids.tolist(), remote_ids.tolist(), vuln_ids.tolist()):
                                        if int(ri) != int(li):
                                            return {key: (int(li), int(ri), int(vi))}
                                    return {key: (int(local_ids[0]), int(remote_ids[0]), int(vuln_ids[0]))}
                                if key == "local_vulnerability":
                                    local_ids, vuln_ids = _np.argwhere(action_mask[key]).T
                                    return {key: (int(local_ids[0]), int(vuln_ids[0]))}
                            except Exception:
                                return None
                        # If only 1 discovered, strongly bias connect
                        disc = 0
                        try:
                            disc = int((self._last_raw_obs or {}).get("discovered_node_count", 0) or 0)
                        except Exception:
                            disc = 0
                        order = ("connect", "remote_vulnerability", "local_vulnerability") if disc <= 1 else ("connect", "remote_vulnerability", "local_vulnerability")
                        repaired = None
                        for ktry in order:
                            repaired = repaired or first_valid(ktry)
                        if repaired is not None:
                            backend_action = repaired
            except Exception:
                pass

            return backend_action

        # CW branch: ensure 'red' action stays within a single-host window
        if not isinstance(backend_action, dict):
            backend_action = {}

        red_value = backend_action.get("red", None)
        try:
            if hasattr(self.env, "red_agent"):
                action_space = getattr(self.env.red_agent, "action_space", None)
                max_size = getattr(action_space, "max_size", None)
                if red_value is None:
                    red_value = 0
                if isinstance(max_size, (int, np.integer)) and max_size > 0:
                    red_value = int(red_value) % int(max_size)
                else:
                    red_value = int(red_value)
        except Exception:
            red_value = 0

        # Ensure blue action exists; default to the first action (0)
        blue_value = backend_action.get("blue", None)
        try:
            if blue_value is None and hasattr(self.env, "blue_agent"):
                blue_action_space = getattr(self.env.blue_agent, "action_space", None)
                if blue_action_space is not None and hasattr(blue_action_space, "max_size"):
                    blue_max = getattr(blue_action_space, "max_size")
                    if isinstance(blue_max, (int, np.integer)) and blue_max > 0:
                        blue_value = 0
                if blue_value is None:
                    blue_value = 0
        except Exception:
            blue_value = 0

        backend_action = {"red": red_value, "blue": blue_value}
        return backend_action

    def _compute_unified_mask(self) -> np.ndarray:
        """Compute a 7-D unified action mask from CBS action mask.
        noop always valid; other entries depend on whether CBS exposes at least
        one valid low-level option for the mapped action type.
        Order: [noop, ping_sweep, port_scan, discovery, lateral_move, privilege_escalation, impact]
        """
        mask = np.ones((len(self.act_t.unified_actions),), dtype=np.float32)
        try:
            am = self.env.compute_action_mask() if hasattr(self.env, "compute_action_mask") else None
            if am is None:
                return mask
            def any_true(key):
                try:
                    arr = am.get(key)
                    return bool(np.any(arr))
                except Exception:
                    return False
            # map
            has_connect = any_true("connect")
            has_remote = any_true("remote_vulnerability")
            has_local = any_true("local_vulnerability")
            name_to_idx = {n:i for i,n in enumerate(self.act_t.unified_actions)}
            # noop
            mask[name_to_idx["noop"]] = 1.0
            # ping_sweep, discovery via connect
            mask[name_to_idx["ping_sweep"]] = 1.0 if has_connect else 0.0
            mask[name_to_idx["discovery"]] = 1.0 if has_connect else 0.0
            # port_scan, lateral_move via remote
            mask[name_to_idx["port_scan"]] = 1.0 if has_remote else 0.0
            mask[name_to_idx["lateral_move"]] = 1.0 if has_remote else 0.0
            # privilege_escalation, impact via local
            mask[name_to_idx["privilege_escalation"]] = 1.0 if has_local else 0.0
            mask[name_to_idx["impact"]] = 1.0 if has_local else 0.0
        except Exception:
            pass
        return mask

    def _reset_progress_trackers(self):
        self._prev_disc_total = 0
        self._prev_owned_total = 0
        self._last_probe_success_step = -1
        self._step_counter = 0

    def _sync_progress_counters(self, raw):
        try:
            disc_total = int(raw.get("discovered_node_count", 0) or 0)
        except Exception:
            disc_total = 0
        self._prev_disc_total = disc_total
        try:
            import numpy as _np
            priv = raw.get("nodes_privilegelevel", _np.array([], dtype=_np.int32))
            if not isinstance(priv, _np.ndarray):
                priv = _np.array(priv, dtype=_np.int32) if priv is not None else _np.array([], dtype=_np.int32)
            owned_total = int((priv >= 1).sum())
        except Exception:
            owned_total = 0
        self._prev_owned_total = owned_total
        self._last_probe_success_step = -1

    def _progress_snapshot(self, raw):
        disc_total = 0
        try:
            disc_total = int((raw or {}).get("discovered_node_count", 0) or 0)
        except Exception:
            disc_total = 0
        new_disc = max(0, disc_total - getattr(self, "_prev_disc_total", 0))
        owned_total = 0
        new_owned = 0
        try:
            import numpy as _np
            priv = (raw or {}).get("nodes_privilegelevel", _np.array([], dtype=_np.int32))
            if not isinstance(priv, _np.ndarray):
                priv = _np.array(priv, dtype=_np.int32) if priv is not None else _np.array([], dtype=_np.int32)
            owned_total = int((priv >= 1).sum())
            new_owned = max(0, owned_total - getattr(self, "_prev_owned_total", 0))
        except Exception:
            owned_total = 0
            new_owned = 0
        return disc_total, new_disc, owned_total, new_owned

   
    # Import workarounds (no vendor edits)
    def _install_cyberwheel_import_workarounds(self):
        import sys
        import types
        import importlib
        import re
        from pathlib import Path

        # If already loaded (or previously stubbed), do nothing.
        if 'cyberwheel.utils' in sys.modules:
            return

        # Locate the vendor utils/ directory: repo_root/cyberwheel/cyberwheel/utils
        try:
            utils_dir = (Path(__file__).resolve().parents[1] / 'cyberwheel' / 'cyberwheel' / 'utils')
        except Exception:
            return
        if not utils_dir.exists():
            return

        # Helper: CamelCase -> snake_case  (HybridSetList -> hybrid_set_list)
        def camel_to_snake(name: str) -> str:
            s1 = re.sub(r'(.)([A-Z][a-z]+)', r'\1_\2', name)
            return re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

        # Create a minimal package-like module with a __path__ so submodules import normally.
        m = types.ModuleType('cyberwheel.utils')
        m.__file__ = str(utils_dir / '__init__.py')
        m.__path__ = [str(utils_dir)]  # behave like a package

        # Lazy attribute resolver to mimic "from cyberwheel.utils import X"
        def __getattr__(name):
            # 1) Special-case: some code does "from cyberwheel.utils import get_service_map"
            if name == 'get_service_map':
                mod = importlib.import_module('cyberwheel.utils.get_service_map')
                return getattr(mod, 'get_service_map')

            # 2) Try a submodule with the SAME name
            try:
                mod = importlib.import_module(f'cyberwheel.utils.{name}')
                if hasattr(mod, name):
                    return getattr(mod, name)
                # return module itself so "from cyberwheel.utils import host_types" still works
                return mod
            except Exception:
                pass

            # 3) Try snake_case submodule for CamelCase symbols (HybridSetList -> hybrid_set_list)
            snake = camel_to_snake(name)
            try:
                mod = importlib.import_module(f'cyberwheel.utils.{snake}')
                if hasattr(mod, name):
                    return getattr(mod, name)
                if hasattr(mod, snake):
                    return getattr(mod, snake)
            except Exception:
                pass

            # 4) Try a few common container module names
            for candidate in ('collections', 'helpers', 'types', 'data_structures'):
                try:
                    mod = importlib.import_module(f'cyberwheel.utils.{candidate}')
                    if hasattr(mod, name):
                        return getattr(mod, name)
                except Exception:
                    continue

            # Not found
            raise AttributeError(name)

        m.__getattr__ = __getattr__

        # Register the stub BEFORE ANY vendor import happens.
        sys.modules['cyberwheel.utils'] = m
