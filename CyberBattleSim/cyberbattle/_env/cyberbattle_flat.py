# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""CyberBattle environment based on a flat network structure with credential reuse"""

from ..samples.flatnetwork.flatnetwork import new_environment
from . import cyberbattle_env


class CyberBattleFlat(cyberbattle_env.CyberBattleEnv):
    """CyberBattle environment based on a flat network with credential reuse.
    
    This scenario tests whether the agent prioritizes credential actions before exploit.
    - Topology: Flat enterprise network where most machines can directly communicate
    - Vulnerability distribution: Same SSH or RDP login credentials reused on multiple machines
    - Success probability: Credential reuse: 0.6, Exploit: 0.3
    - Node values: Low
    - Termination: Successfully establishing a secure, initial position
    """

    def __init__(self, num_nodes=20, credential_reuse_prob=0.6, exploit_success_prob=0.3, **kwargs):
        self.num_nodes = num_nodes
        self.credential_reuse_prob = credential_reuse_prob
        self.exploit_success_prob = exploit_success_prob
        super().__init__(
            initial_environment=new_environment(num_nodes, credential_reuse_prob, exploit_success_prob),
            **kwargs
        )

    @property
    def name(self) -> str:
        return f"CyberBattleFlat-{self.num_nodes}"

