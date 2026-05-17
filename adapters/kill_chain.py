"""
Shared kill-chain stage definitions used by:
  - UnifiedSecEnv  (stage-transition reward, per-host phase)
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
       CBS: attacker achieved the game objective — episode terminated (terminated=True,
            not truncated).  In CyberBattleChain-v0 this means own_atleast_percent
            (default 50 %) of nodes was reached.  Because the game ends the moment the
            threshold is crossed, stage_from_cbs() alone cannot detect stage 4 from obs;
            the caller must pass terminated=True (see eval/eval_cw_dapn_on_cbs.py).
            As a best-effort fallback, stage_from_cbs also returns 4 when
            compromised >= CBS_WIN_NODES (default 3, configurable via env var).
"""

import numpy as np

KILL_CHAIN_STAGES = 5

# ── Per-host phase helpers (used by UnifiedSecEnv._build_obs) ──────────────────

def cw_host_phase(chunk) -> int:
    """
    Return the CW kill-chain phase (0-6) for a single 7-element host chunk.
    Layout: [type, sweeped, scanned, discovered, on_host, escalated, impacted]
    Returns -1 for padding slots (all -1).
    """
    chunk = np.asarray(chunk, dtype=np.float32)
    if chunk.size < 7 or np.all(chunk == -1):
        return -1
    if int(chunk[6]): return 6   # impacted
    if int(chunk[5]): return 5   # escalated
    if int(chunk[4]): return 4   # on_host
    if int(chunk[3]): return 3   # discovered (recon complete)
    if int(chunk[2]): return 2   # scanned
    if int(chunk[1]): return 1   # swept
    return 0


def cbs_node_phase(priv_level: int) -> int:
    """
    Return CBS kill-chain phase on the shared 0-6 scale from a node's
    privilege level.  CBS has no recon sub-steps so phases 1-3 are skipped.

      priv = 0  →  phase 0  (not owned)
      priv = 1  →  phase 4  (foothold / owned — matches CW "on_host")
      priv >= 2 →  phase 5  (deeper access — matches CW "escalated")
    """
    if priv_level >= 2: return 5
    if priv_level >= 1: return 4
    return 0


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

    # Privilege levels across all nodes.
    # Index 0 is the attacker's starting node ("start") which always has priv=1
    # by CBS design — skip it so it doesn't inflate the compromised count.
    priv = obs.get("nodes_privilegelevel", np.array([], dtype=np.int32))
    if not isinstance(priv, np.ndarray):
        priv = np.array(priv, dtype=np.int32) if priv is not None else np.array([], dtype=np.int32)

    # Exclude node 0 (attacker start node) from compromised/escalated counts
    priv_targets = priv[1:] if priv.size > 1 else np.array([], dtype=np.int32)
    compromised = int((priv_targets >= 1).sum())

    # Stage 4: best-effort obs-based fallback.
    # CyberBattleChain terminates (terminated=True) when own_atleast_percent is met.
    # The authoritative stage-4 signal is the terminated flag in the rollout loop.
    # Here we also check customer_data_found (other CBS topologies) and a configurable
    # node-ownership threshold so stage_from_cbs alone can still return 4 in those cases.
    import os as _os
    cbs_win_nodes = int(_os.environ.get("CBS_WIN_NODES", "3") or 3)
    if (int(obs.get("customer_data_found", 0) or 0) > 0
            or compromised >= cbs_win_nodes):
        return 4

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

    Host index 0 is the entry host — the agent starts there with on_host=1,
    so we deliberately exclude it from the on_host foothold check (mirroring
    how CBS excludes node 0 from the compromised count).  Escalation anywhere
    (including the entry host) still counts as a real achievement.
    """
    HOST_ATTRS = 7
    obs_vec = cw_raw_to_red_vector(obs_vec)

    n = int(obs_vec.size)
    standalone_len = n % HOST_ATTRS
    max_hosts = (n - standalone_len) // HOST_ATTRS if n >= HOST_ATTRS else 0

    any_impacted = False
    any_on_non_entry_host = False   # on_host=1 on host index > 0 only
    escalated_count = 0
    n_valid_hosts = 0               # total non-padding hosts visible in obs

    for i in range(max_hosts):
        base = i * HOST_ATTRS
        chunk = obs_vec[base: base + HOST_ATTRS]
        if np.all(chunk == -1):
            continue
        n_valid_hosts += 1
        if chunk[6] == 1:
            any_impacted = True
        if chunk[5] == 1:
            escalated_count += 1
        if chunk[4] == 1 and i > 0:    # exclude entry host (index 0)
            any_on_non_entry_host = True

    # Stage 4: any host impacted
    if any_impacted:
        return 4
    # Stage 3: lateral spread — escalated on 2+ distinct hosts
    if escalated_count >= 2:
        return 3
    # Stage 2: foothold — escalated anywhere OR moved to a non-entry host
    if escalated_count >= 1 or any_on_non_entry_host:
        return 2
    # Stage 1: recon done — more than just the entry host is visible.
    # Mirrors CBS: stage_from_cbs returns 1 when discovered_node_count > 1.
    # In CW, ping_sweep adds new hosts to the observation, so n_valid_hosts > 1
    # means recon has found something — no need for the 'discovered' attribute.
    if n_valid_hosts > 1:
        return 1
    return 0
