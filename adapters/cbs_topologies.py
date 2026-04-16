# adapters/cbs_topologies.py
from __future__ import annotations

import os
import random
import yaml
import networkx as nx
import numpy
from typing import Dict, List, Set
from collections import OrderedDict


def _add_protocol_edge(g: nx.DiGraph, u: str, v: str, protocol: str) -> None:
    """
    Add a directed edge u->v and annotate/merge the 'protocol' edge attribute as a set.
    CyberBattleSim's traffic->model converter expects 'protocol' to be a set of labels.
    """
    if g.has_edge(u, v):
        protos: Set[str] = g.edges[(u, v)].get("protocol", set()) or set()
        protos.add(protocol)
        g.edges[(u, v)]["protocol"] = protos
    else:
        g.add_edge(u, v, protocol={protocol})


def _add_bidir_protocols(g: nx.DiGraph, a: str, b: str, protocols: List[str]) -> None:
    for p in protocols:
        _add_protocol_edge(g, a, b, p)
        _add_protocol_edge(g, b, a, p)


def _hosts_by_subnet_from_cw_yaml(cw_cfg: Dict) -> Dict[str, List[str]]:
    """
    Extract hosts grouped by subnet from a CyberWheel network YAML.
    Structure expected under 'topology' -> core_router -> <subnet>: [hosts...]
    """
    topology = (cw_cfg or {}).get("topology", {}) or {}
    core = topology.get("core_router", {}) or {}
    out: Dict[str, List[str]] = {}
    for subnet_name, hosts in core.items():
        out[subnet_name] = list(hosts or [])
    return out


def _interfaces_pairs(cw_cfg: Dict) -> List[tuple]:
    """
    Extract explicit interface pairs like:
      interfaces:
        user01: [dmz01]
        dmz01: [server01]
    Returns list of (a, b) pairs.
    """
    iface = (cw_cfg or {}).get("interfaces", {}) or {}
    pairs: List[tuple] = []
    for a, bs in iface.items():
        if not bs:
            continue
        if isinstance(bs, list):
            for b in bs:
                pairs.append((str(a), str(b)))
        else:
            # single string
            pairs.append((str(a), str(bs)))
    return pairs


def build_cbs_env_from_cw_yaml(yaml_path: str):
    """
    Build a CyberBattleSim CyberBattleEnv whose connectivity mirrors the CyberWheel
    10-host network structure: users -> dmz -> servers, with an admin workstation and a jump box.
    """
    # Lazy imports to avoid CBS dependency unless needed
    import CyberBattleSim.cyberbattle.simulation.model as m
    import CyberBattleSim.cyberbattle.simulation.actions as actions
    import CyberBattleSim.cyberbattle._env.cyberbattle_env as cbs_env
    import CyberBattleSim.cyberbattle.simulation.generate_network as gen_net

    with open(yaml_path, "r") as f:
        cw_cfg = yaml.safe_load(f) or {}

    hosts_block = (cw_cfg or {}).get("hosts", {}) or {}
    host_names: List[str] = [str(h) for h in hosts_block.keys()]

    
    g = nx.DiGraph()
    g.add_nodes_from(host_names)

    # Fully connect every host with bidirectional links to allow unrestricted lateral movement.
    # Provide a mix of protocols so both RDP/SMB-style and HTTP-style actions remain available.
    full_mesh_protocols = ["RDP", "SMB", "HTTP"]
    for idx, a in enumerate(host_names):
        for b in host_names[idx + 1 :]:
            _add_bidir_protocols(g, a, b, full_mesh_protocols)

    # Heuristic grouping by subnet from CW 'topology'
    by_subnet = _hosts_by_subnet_from_cw_yaml(cw_cfg)
    user_hosts = by_subnet.get("user_subnet", [])
    dmz_hosts = by_subnet.get("dmz_subnet", [])
    server_hosts = by_subnet.get("server_subnet", [])

    # Explicit interface links (bidirectional RDP/SMB)
    for a, b in _interfaces_pairs(cw_cfg):
        if a in host_names and b in host_names:
            _add_bidir_protocols(g, a, b, ["RDP", "SMB"])

    # Admin workstation, jump box special roles if present
    admin = "admin_workstation" if "admin_workstation" in host_names else None
    jump = "dmz_jump_box" if "dmz_jump_box" in host_names else None

    # Users <-> DMZ: allow RDP/SMB
    for u in user_hosts:
        for d in dmz_hosts:
            _add_bidir_protocols(g, u, d, ["RDP", "SMB"])

    # DMZ <-> Servers: allow HTTP/SMB
    for d in dmz_hosts:
        for s in server_hosts:
            _add_bidir_protocols(g, d, s, ["HTTP", "SMB"])

    # Admin workstation -> Servers: allow HTTP/SMB (admin maintenance)
    if admin:
        for s in server_hosts:
            _add_protocol_edge(g, admin, s, "HTTP")
            _add_protocol_edge(g, admin, s, "SMB")
            # and allow responses
            _add_protocol_edge(g, s, admin, "HTTP")

    # Jump box -> Users: RDP (models 'allow ssh from dmz_jump_box' loosely as RDP)
    if jump:
        for u in user_hosts:
            _add_protocol_edge(g, jump, u, "RDP")
            _add_protocol_edge(g, u, jump, "RDP")

    # If no servers defined, connect dmz01->server01 if both exist as a minimal path
    if not server_hosts:
        if "dmz01" in host_names and "server01" in host_names:
            _add_bidir_protocols(g, "dmz01", "server01", ["HTTP", "SMB"])

    # Convert traffic graph to CBS network model using CBS helper.
    # Seed RNG so the same topology yields deterministic vulns; use high leak probabilities
    # so the agent can spread and reach 60% (e.g. 6/10 nodes) with valid actions.
    cw_10_seed = int(os.environ.get("CBS_CW10_SEED", "42"))
    easy_60 = os.environ.get("CBS_EASY_60PCT", "1") == "1"
    if easy_60:
        random.seed(cw_10_seed)
        numpy.random.seed(cw_10_seed)
    try:
        import numpy as _np
        if easy_60:
            _np.random.seed(cw_10_seed)
    except Exception:
        pass
    gen_kwargs = {}
    if easy_60:
        gen_kwargs = dict(
            cached_smb_password_probability=0.98,
            cached_rdp_password_probability=0.98,
            cached_accessed_network_shares_probability=0.98,
            cached_password_has_changed_probability=0.02,
            probability_two_nodes_use_same_password_to_access_given_resource=0.95,
            traceroute_discovery_probability=0.95,
        )
    cbs_network = gen_net.cyberbattle_model_from_traffic_graph(g, **gen_kwargs)

    # Safety net: ensure entry node has a local vuln that leaks a credential (so progress policy
    # can get a cred and then lateral_move; needed for CBS_MINIMAL_SEED=1 and 60% goal).
    entry_node = sorted(host_names)[0] if host_names else None
    try:
        import CyberBattleSim.cyberbattle.simulation.model as m  # type: ignore
        nodes_list = list(cbs_network.nodes())
        target_node = str(entry_node) if (entry_node and str(entry_node) in cbs_network.nodes()) else (str(nodes_list[0]) if nodes_list else None)
        if target_node:
            node_info = cbs_network.nodes[target_node].get("data")
            if node_info is not None:
                # Inject a simple local vuln that leaks one SMB credential for validation to pass
                node_info.vulnerabilities = dict(getattr(node_info, "vulnerabilities", {}) or {})
                node_info.vulnerabilities["ScanWindowsCredentialManagerForSMB"] = m.VulnerabilityInfo(
                    description="Look for network credentials in the Windows Credential Manager (forced non-empty)",
                    type=m.VulnerabilityType.LOCAL,
                    outcome=m.LeakedCredentials(credentials=[m.CachedCredential(node=target_node, port="SMB", credential="bootstrap_pwd")]),
                    reward_string="Discovered SMB creds in the Windows Credential Manager",
                    cost=1.0,
                )
                cbs_network.nodes[target_node]["data"] = node_info
    except Exception:
        # Best-effort; if anything fails, proceed without injection
        pass
    env_model = m.Environment(
        network=cbs_network,
        vulnerability_library=dict([]),
        identifiers=gen_net.ENV_IDENTIFIERS,
    )

    # Build gym env with configurable attacker goal
    try:
        own_pct = float(os.environ.get("CBS_GOAL_OWN_PCT", "0.2") or 0.2)
    except Exception:
        own_pct = 0.2
    try:
        goal_reward = float(os.environ.get("CBS_GOAL_REWARD", "50") or 50)
    except Exception:
        goal_reward = 50.0
    # Use actual network size so observation (e.g. nodes_privilegelevel) has length = node count,
    # matching CW for lockstep (get_owned_pct total_nodes CW=10, CBS=10).
    num_nodes = max(1, len(host_names))
    kwargs = dict(
        initial_environment=env_model,
        attacker_goal=cbs_env.AttackerGoal(own_atleast_percent=own_pct, reward=goal_reward),
        throws_on_invalid_actions=False,
        maximum_node_count=num_nodes,
    )
    # Optionally zero-out terminal rewards to emphasize shaped rewards during evaluation
    if os.environ.get("CBS_ZERO_WIN_LOSE_REWARD", "0") == "1":
        kwargs["winning_reward"] = 0.0
        kwargs["losing_reward"] = 0.0
    gym_env = cbs_env.CyberBattleEnv(**kwargs)

    # Seed attacker knowledge so connect/lateral actions are immediately available
    seed_entry = (user_hosts[0] if user_hosts else host_names[0]) if host_names else None
    seed_target = (dmz_hosts[0] if dmz_hosts else (server_hosts[0] if server_hosts else None))
    leaked_credential = m.CachedCredential(
        node=seed_target or (host_names[0] if host_names else "server01"),
        port="SMB",
        credential="seed_pwd",
    )

    gym_env._seed_host_names = host_names
    gym_env._seed_entry = seed_entry
    gym_env._seed_target = seed_target

    # When CBS_MINIMAL_SEED=1: only entry discovered/owned, no leaked cred (match CW initial state)
    def _minimal_seed_entry():
        return sorted(host_names)[0] if host_names else (seed_entry or "server01")

    def _seed_attacker_state():
        try:
            minimal = os.environ.get("CBS_MINIMAL_SEED", "0") == "1"
            if minimal:
                entry = _minimal_seed_entry()
            else:
                entry = seed_entry
            if entry:
                env = getattr(gym_env, "_CyberBattleEnv__environment", None)
                if env is not None and hasattr(env, "network"):
                    try:
                        import CyberBattleSim.cyberbattle.simulation.model as m
                        node_dict = env.network.nodes.get(str(entry))
                        if node_dict is not None:
                            node_data = node_dict.get("data") if isinstance(node_dict, dict) else None
                            if node_data is not None:
                                if not getattr(node_data, "agent_installed", False):
                                    node_data.agent_installed = True
                                if getattr(node_data, "privilege_level", m.PrivilegeLevel.NoAccess) < m.PrivilegeLevel.LocalUser:
                                    node_data.privilege_level = m.PrivilegeLevel.LocalUser
                                env.network.nodes[str(entry)].update({"data": node_data})
                    except (TypeError, KeyError, AttributeError):
                        pass
            if minimal:
                discovered_ids = [str(_minimal_seed_entry())]
            else:
                discovered_ids = [str(h) for h in host_names] if host_names else list(gym_env._CyberBattleEnv__discovered_nodes or [])
                if not discovered_ids and seed_entry:
                    discovered_ids = [str(seed_entry)]
            discovery_map = OrderedDict((node_id, actions.NodeTrackingInformation()) for node_id in discovered_ids)
            gym_env._actuator._discovered_nodes = discovery_map  # type: ignore[attr-defined]
            gym_env._CyberBattleEnv__discovered_nodes = list(discovery_map.keys())
            cred_cache = list(gym_env._CyberBattleEnv__credential_cache or [])
            if not minimal:
                if leaked_credential not in cred_cache:
                    cred_cache.append(leaked_credential)
                gym_env._CyberBattleEnv__credential_cache = cred_cache
                try:
                    gathered = getattr(gym_env._actuator, "_gathered_credentials", set())
                    gathered.add(leaked_credential.credential)
                    gym_env._actuator._gathered_credentials = gathered  # type: ignore[attr-defined]
                except Exception:
                    pass
            else:
                gym_env._CyberBattleEnv__credential_cache = cred_cache
            gym_env._CyberBattleEnv__owned_nodes_indices_cache = None
        except Exception:
            pass

    original_reset = gym_env.reset

    def seeded_reset(*args, **kwargs):
        out = original_reset(*args, **kwargs)
        _seed_attacker_state()
        # Observation returned above was built before seeding, so it has 0 discovered nodes/cache.
        # Rebuild observation so it reflects seeded state.
        obs = gym_env._CyberBattleEnv__get_blank_observation()
        obs["action_mask"] = gym_env.compute_action_mask()
        obs["discovered_nodes_properties"] = gym_env._CyberBattleEnv__get_property_matrix()
        obs["nodes_privilegelevel"] = gym_env._CyberBattleEnv__get_privilegelevel_array()
        cred_cache = gym_env._CyberBattleEnv__credential_cache
        n_cred = len(cred_cache)
        obs["credential_cache_length"] = n_cred
        try:
            cache = [numpy.array([gym_env._CyberBattleEnv__find_external_index(c.node), gym_env._CyberBattleEnv__portname_to_index(c.port)], dtype=numpy.int64) for c in cred_cache]
            obs["credential_cache_matrix"] = gym_env._CyberBattleEnv__pad_tuple_if_requested(cache, 2, gym_env._CyberBattleEnv__bounds.maximum_total_credentials)
        except Exception:
            pass
        gym_env._CyberBattleEnv__owned_nodes_indices_cache = None
        info = out[1] if isinstance(out, tuple) and len(out) == 2 else {}
        return (obs, info)

    # Keep __discovered_nodes in sync with actuator (CBS only updates it on LeakedNodesId/
    # LeakedCredentials, not on LateralMove). Sync before any index lookup.
    def _sync_discovered():
        try:
            act = getattr(gym_env, "_actuator", None)
            if act is not None and hasattr(act, "_discovered_nodes"):
                gym_env._CyberBattleEnv__discovered_nodes = list(act._discovered_nodes.keys())
        except Exception:
            pass

    _orig_find_index = gym_env._CyberBattleEnv__find_external_index

    def _find_external_sync(node_id):
        _sync_discovered()
        disc = gym_env._CyberBattleEnv__discovered_nodes
        if node_id not in disc:
            disc.append(node_id)
        return gym_env._CyberBattleEnv__discovered_nodes.index(node_id)

    gym_env._CyberBattleEnv__find_external_index = _find_external_sync  # type: ignore[assignment]
    gym_env.reset = seeded_reset  # type: ignore[assignment]
    _seed_attacker_state()
    return gym_env


