# config/env_builders.py

import gymnasium as gym
import os

# --------------------- CYBERBATTLESIM ---------------------
def make_cbs_env():
    # Lazy import to avoid requiring CyberBattleSim when only using CyberWheel
    import CyberBattleSim.cyberbattle._env.cyberbattle_env as cbs_env
    # Allow choosing different CBS envs; default to Chain
    env_id = os.environ.get("CBS_ENV", "CyberBattleChain-v0")
    # Chain requires an EVEN size
    size = int(os.environ.get("CBS_SIZE", "6"))
    kwargs = {"attacker_goal": cbs_env.AttackerGoal(own_atleast_percent=0.2, reward=50)}
    # Optionally zero terminal rewards so shaped rewards are visible (non-constant)
    if os.environ.get("CBS_ZERO_WIN_LOSE_REWARD", "0") == "1":
        kwargs["winning_reward"] = 0.0
        kwargs["losing_reward"] = 0.0
    if env_id == "CyberBattleChain-v0":
        # Relax goal for fast non-zero native rewards
        kwargs["attacker_goal"] = cbs_env.AttackerGoal(own_atleast_percent=0.1, reward=0)
    # Custom builder: mirror CyberWheel 10-host network topology
    if env_id == "CyberBattleCW10-v0":
        from adapters.cbs_topologies import build_cbs_env_from_cw_yaml
        # Prefer absolute path if available
        yaml_path = os.environ.get(
            "CW_NET_YAML",
            "/home/ssaika/rl-transfer-sec-clean/cyberwheel/cyberwheel/data/configs/network/10-host-network.yaml",
        )
        return build_cbs_env_from_cw_yaml(yaml_path)
    if env_id == "CyberBattleChain-v0":
        kwargs["size"] = size
    return gym.make(env_id, **kwargs)


# ----------------------- CYBERWHEEL -----------------------
CW_ENV_PKG = "cyberwheel.data.configs.environment"
# CW_DEFAULT_ENV_YAML = "train_rl_red_agent_vs_rl_blue.yaml"  # primary
# Use a packaged, present YAML that exists in this repo
CW_DEFAULT_ENV_YAML = "cyberwheel.yaml"
CW_FALLBACK_ENV_YAML = "cyberwheel.yaml"

def make_cw_env():
    import sys, types, yaml
    from types import SimpleNamespace
    from importlib.resources import files

    # ---- helpers ----
    def _to_ns(obj):
        if isinstance(obj, dict):
            return SimpleNamespace(**{k: _to_ns(v) for k, v in obj.items()})
        if isinstance(obj, list):
            return [_to_ns(v) for v in obj]
        return obj

    def _normalize_args(cfg: dict) -> SimpleNamespace:
        """Ensure args has .agent_config['red'/'blue'] and the expected config paths."""
        c = dict(cfg)

        # Agent config: guarantee keys exist
        agent_cfg = c.get("agent_config", {})
        red_cfg = (agent_cfg.get("red") if isinstance(agent_cfg, dict) else None) or \
                  c.get("red_agent_config") or c.get("red_agent") or c.get("red") or {}
        blue_cfg = (agent_cfg.get("blue") if isinstance(agent_cfg, dict) else None) or \
                   c.get("blue_agent_config") or c.get("blue_agent") or c.get("blue") or {}
        # Ensure plain dicts for agent_config (downstream expects dict indexing)
        def _to_dict(obj):
            if isinstance(obj, SimpleNamespace):
                return {k: _to_dict(v) for k, v in vars(obj).items()}
            if isinstance(obj, dict):
                return {k: _to_dict(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_to_dict(v) for v in obj]
            return obj

        agent_config_dict = {"red": _to_dict(red_cfg), "blue": _to_dict(blue_cfg)}
        # Avoid converting agent_config into SimpleNamespace; attach after namespacing
        c.pop("agent_config", None)

        # Keep any existing config keys; do NOT overwrite with {} if present
        for k in ["host_config", "network_config", "services_config", "detector_config", "campaign_config"]:
            c.setdefault(k, c.get(k))

        ns = _to_ns(c)
        # Provide sensible defaults for RL if missing
        if not hasattr(ns, "valid_targets"):
            setattr(ns, "valid_targets", "all")
        if not hasattr(ns, "red_reward_function"):
            setattr(ns, "red_reward_function", "reward_decoy_hits")
        if not hasattr(ns, "blue_reward_function"):
            setattr(ns, "blue_reward_function", "reward_red_delay")
        setattr(ns, "agent_config", agent_config_dict)
        return ns

    def _validate_required_args(args_ns: SimpleNamespace, yaml_name: str):
        """Be permissive for packaged CW configs: require only host/network, and accept 'agents' as source for agent YAMLs."""
        missing = []
        # Only strictly require host and network configs; packaged files may derive services/detectors internally
        for k in ("host_config", "network_config"):
            v = getattr(args_ns, k, None)
            if isinstance(v, (list, tuple)):
                if len(v) == 0:
                    missing.append(k)
            elif not v or not isinstance(v, (str, bytes)):
                missing.append(k)
        # If agent_config not present yet, allow presence of 'agents' to fulfill requirement
        ac = getattr(args_ns, "agent_config", None)
        has_agents_block = hasattr(args_ns, "agents")
        if not ac and not has_agents_block:
            missing.append("agent_config or agents")
        if missing:
            raise ValueError(
                f"[{yaml_name}] Missing or invalid keys required by Cyberwheel: {', '.join(missing)}.\n"
                "Pick a simpler config (e.g., 'cyberwheel.yaml') or update the YAML so these are present."
            )

    # no-op: rely on vendor cyberwheel.utils.get_service_map
    def _install_lazy_service_map():
        return

    def _build_base(args_ns, cfg_dict, yaml_name):
        """Build a Cyberwheel base with a realized Network and safe args."""
        from importlib.resources import files as _files
        from cyberwheel.network.network_base import Network
        from cyberwheel.cyberwheel_envs.cyberwheel import Cyberwheel
        from cyberwheel.utils.get_service_map import get_service_map

        # Resolve network config path
        try:
            net_cfg_name = getattr(args_ns, "network_config")
            if isinstance(net_cfg_name, (list, tuple)):
                if len(net_cfg_name) == 0:
                    raise ValueError("empty network_config list")
                net_cfg_name = net_cfg_name[0]
            net_cfg_path = _files("cyberwheel.data.configs.network") / net_cfg_name
        except Exception as e:
            raise ValueError(f"[{yaml_name}] Unable to resolve network_config path: {e}")

        # Build the network from YAML (host_config name is resolved internally)
        network = Network.create_network_from_yaml(str(net_cfg_path), getattr(args_ns, "host_config", "host_defs_services.yaml"))

        # Ensure service_mapping exists and includes this network
        # Use the real service map helper from Cyberwheel
        service_map = get_service_map(network)
        setattr(args_ns, "service_mapping", {network.name: service_map})

        # Merge agent YAMLs into agent_config so required fields (e.g., entry_host/strategy) are present
        try:
            # Load red agent yaml
            red_yaml_name = None
            agents_obj = getattr(args_ns, "agents", None)
            if isinstance(agents_obj, SimpleNamespace):
                agents_obj = vars(agents_obj)
            if isinstance(agents_obj, dict):
                red_yaml_name = agents_obj.get("red")
                blue_yaml_name = agents_obj.get("blue")
            # Also support packaged schema: top-level red_agent / blue_agent
            if not red_yaml_name:
                red_yaml_name = getattr(args_ns, "red_agent", None)
            if not blue_yaml_name:
                blue_yaml_name = getattr(args_ns, "blue_agent", None)
            # Ensure a mutable dict exists
            if not isinstance(getattr(args_ns, "agent_config", None), dict):
                setattr(args_ns, "agent_config", {})
            agent_cfg_dict = args_ns.agent_config
            if red_yaml_name:
                red_yaml_path = _files("cyberwheel.data.configs.red_agent").joinpath(red_yaml_name)
                with red_yaml_path.open("r") as f:
                    red_yaml = yaml.safe_load(f) or {}
                agent_cfg_dict["red"] = {**red_yaml, **agent_cfg_dict.get("red", {})}
            if blue_yaml_name:
                blue_yaml_path = _files("cyberwheel.data.configs.blue_agent").joinpath(blue_yaml_name)
                with blue_yaml_path.open("r") as f:
                    blue_yaml = yaml.safe_load(f) or {}
                agent_cfg_dict["blue"] = {**blue_yaml, **agent_cfg_dict.get("blue", {})}

            # Ensure required field entry_host
            if "entry_host" not in agent_cfg_dict.get("red", {}):
                first_host = sorted(list(network.hosts.keys()))[0] if getattr(network, "hosts", None) else None
                if first_host:
                    agent_cfg_dict.setdefault("red", {})["entry_host"] = first_host
            # Provide a default red strategy if missing (required by ARTAgent)
            agent_cfg_dict.setdefault("red", {})
            agent_cfg_dict["red"].setdefault("strategy", "ServerDowntime")
        except Exception:
            pass

        return Cyberwheel(args_ns, network=network)

    def _wrap_rl(base, cfg_dict):
        from cyberwheel.cyberwheel_envs.cyberwheel_rl import CyberwheelRL

        # Always build RL env with args namespace and explicit network
        args_obj = getattr(base, "args", None)
        net_obj = getattr(base, "network", None)
        return CyberwheelRL(args_obj, network=net_obj)

    def _wrap_gym(env):
        class _GymLike:
            def __init__(self, e):
                self._env = e
            def reset(self, seed=None, options=None):
                if seed is not None and hasattr(self._env, "seed"):
                    self._env.seed(seed)
                out = self._env.reset()
                return out if (isinstance(out, tuple) and len(out) == 2) else (out, {})
            def step(self, action_obj):
                out = self._env.step(action_obj)
                if isinstance(out, tuple):
                    if len(out) == 4:
                        obs, reward, done, info = out
                        return obs, float(reward), bool(done), False, (info or {})
                    if len(out) == 5:
                        obs, reward, terminated, truncated, info = out
                        return obs, float(reward), bool(terminated), bool(truncated), (info or {})
                return out, 0.0, False, False, {}
            def __getattr__(self, name):
                return getattr(self._env, name)
        return _GymLike(env)

    # ---- build with primary YAML; fallback if needed ----
    _install_lazy_service_map()

    # Determine which YAML to load (allow override via env var, accept absolute paths)
    override_yaml = os.environ.get("CW_ENV_YAML")
    def _load_yaml(name: str):
        if not name:
            raise ValueError("Empty CW env yaml name")
        if os.path.isabs(name):
            with open(name, "r") as fh:
                return yaml.safe_load(fh)
        path = files(CW_ENV_PKG) / name
        with path.open("r") as fh:
            return yaml.safe_load(fh)

    primary_yaml_name = override_yaml or CW_DEFAULT_ENV_YAML
    try:
        primary_cfg = _load_yaml(primary_yaml_name)
    except Exception as e:
        raise ValueError(f"Failed to load Cyberwheel env YAML '{primary_yaml_name}': {e}")
    primary_args = _normalize_args(primary_cfg)
    # Ensure dict agent_config, not SimpleNamespace
    if isinstance(getattr(primary_args, "agent_config", None), SimpleNamespace):
        primary_args.agent_config = vars(primary_args.agent_config)
    _validate_required_args(primary_args, primary_yaml_name)
    base = _build_base(primary_args, primary_cfg, primary_yaml_name)

    # Fallback if network didn't build
    if getattr(base, "network", None) is None:
        fallback_path = files(CW_ENV_PKG) / CW_FALLBACK_ENV_YAML
        with fallback_path.open("r") as f:
            fallback_cfg = yaml.safe_load(f)
        fallback_args = _normalize_args(fallback_cfg)
        if isinstance(getattr(fallback_args, "agent_config", None), SimpleNamespace):
            fallback_args.agent_config = vars(fallback_args.agent_config)
        _validate_required_args(fallback_args, CW_FALLBACK_ENV_YAML)
        base = _build_base(fallback_args, fallback_cfg, CW_FALLBACK_ENV_YAML)
        rl_env = _wrap_rl(base, fallback_cfg)
        return _wrap_gym(rl_env)

    rl_env = _wrap_rl(base, primary_cfg)
    return _wrap_gym(rl_env)
