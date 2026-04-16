"""
Shared kill-chain stage definitions used by:
  - UnifiedSecEnv  (stage-transition reward)
  - train_dapn_encoder (stage-based pairing)

Stage semantics (aligned across both environments):
  0  nothing done yet
  1  recon complete, no foothold yet
  2  foothold — first host/node persistently compromised:
       CW:  escalated on >= 1 host  OR  currently on a host (esc=0)
       CBS: 1 target node owned (priv >= 1)
  3  lateral spread — 2+ hosts/nodes persistently compromised:
       CW:  escalated_count >= 2  (attacker has escalated on 2+ distinct hosts)
       CBS: compromised >= 2  (2+ target nodes with priv >= 1)
       Both now count horizontal multi-host spread, not vertical single-host depth.
  4  impact / exfiltration complete
       CW:  any host impacted
       CBS: unreachable in CyberBattleChain-v0 (game ends at own_atleast_percent=0.5)
"""

import numpy as np

KILL_CHAIN_STAGES = 5


def cw_raw_to_red_vector(obs) -> np.ndarray:
    """
    CyberWheelRL exposes Dict observations {'blue': vec|None, 'red': vec|None}.
    Kill-chain staging and the observation translator expect the flat red per-host vector.
    """
    if isinstance(obs, dict):
        red = obs.get("red")
        if red is None:
            return np.array([], dtype=np.float32)
        return np.asarray(red, dtype=np.float32).reshape(-1)
    return np.asarray(obs, dtype=np.float32).reshape(-1)


def stage_from_cbs(obs) -> int:
    """
    Kill-chain stage from a raw CBS observation dict.

    Aligned with CW depth:
      Stage 1 requires discovered > 0 AND no node owned yet.
      This avoids overlap with stage 2 (CBS remote_vulnerability can
      compress CW's 3 recon steps into 1, so 'discovered' alone is
      not sufficient — we also need to confirm no foothold exists).
    """
    if not isinstance(obs, dict):
        return 0

    # Stage 4: exfiltration complete
    if int(obs.get("customer_data_found", 0) or 0) > 0:
        return 4

    # Privilege levels across all nodes.
    # Index 0 is the attacker's starting node ("start") which always has priv=1
    # by CBS design — skip it so it doesn't inflate the compromised count.
    priv = obs.get("nodes_privilegelevel", np.array([], dtype=np.int32))
    if not isinstance(priv, np.ndarray):
        priv = np.array(priv, dtype=np.int32) if priv is not None else np.array([], dtype=np.int32)

    # Exclude node 0 (attacker start node) from compromised/escalated counts
    priv_targets = priv[1:] if priv.size > 1 else np.array([], dtype=np.int32)
    compromised = int((priv_targets >= 1).sum())

    # Stage 3: lateral movement complete — 2+ target nodes owned.
    # CBS CyberBattleChain does not use priv>=2 escalation; owning multiple
    # nodes is the equivalent depth signal (requires remote_vuln + connect chain).
    # Also check escalation flag for other CBS topologies.
    escalation_flag = int(obs.get("escalation", 0) or 0)
    if (escalation_flag >= 2
            or (priv_targets.size > 0 and int((priv_targets >= 2).sum()) > 0)
            or compromised >= 2):
        return 3

    # Stage 2: foothold — at least one target node owned (priv >= 1)
    if compromised > 0:
        return 2

    # Stage 1: recon done — target nodes discovered but none owned yet
    # discovered_node_count > 1 because count=1 means only the start node
    if int(obs.get("discovered_node_count", 0) or 0) > 1:
        return 1

    return 0


def stage_from_cw(obs_vec) -> int:
    """
    Kill-chain stage from a raw CW observation vector.

    CW obs layout per host (HOST_ATTRS=7):
      [type, sweeped, scanned, discovered, on_host, escalated, impacted]
    """
    HOST_ATTRS = 7
    obs_vec = cw_raw_to_red_vector(obs_vec)

    n = int(obs_vec.size)
    standalone_len = n % HOST_ATTRS
    max_hosts = (n - standalone_len) // HOST_ATTRS if n >= HOST_ATTRS else 0

    any_impacted = any_on_host = any_discovered = False
    escalated_count = 0

    for i in range(max_hosts):
        base = i * HOST_ATTRS
        chunk = obs_vec[base: base + HOST_ATTRS]
        if np.all(chunk == -1):
            continue
        if chunk[6] == 1:
            any_impacted = True
        if chunk[5] == 1:
            escalated_count += 1
        if chunk[4] == 1:
            any_on_host = True
        if chunk[3] == 1:
            any_discovered = True

    # Stage 4: any host impacted
    if any_impacted:
        return 4
    # Stage 3: lateral spread — escalated on 2+ distinct hosts (mirrors CBS compromised>=2)
    if escalated_count >= 2:
        return 3
    # Stage 2: foothold — escalated on first host OR currently on a host pre-escalation
    if escalated_count >= 1 or any_on_host:
        return 2
    # Stage 1: recon done — hosts discovered but no foothold
    if any_discovered:
        return 1
    return 0
