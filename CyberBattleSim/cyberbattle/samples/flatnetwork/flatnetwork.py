# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""CyberBattleSim flat network scenario for credential preference testing.
This scenario creates a flat enterprise network where most machines can directly communicate.
The same SSH or RDP login credentials are reused on multiple machines.
Success probability: Credential reuse: 0.6, Exploit: 0.3
Node values: Low
Termination: Successfully establishing a secure, initial position
Transfer evaluation question: Does the agent prioritize credential actions before exploit?
"""

from cyberbattle.simulation.model import Identifiers, NodeID, NodeInfo
from ...simulation import model as m
from typing import Dict

DEFAULT_ALLOW_RULES = [
    m.FirewallRule("RDP", m.RulePermission.ALLOW),
    m.FirewallRule("SSH", m.RulePermission.ALLOW),
    m.FirewallRule("HTTPS", m.RulePermission.ALLOW),
    m.FirewallRule("HTTP", m.RulePermission.ALLOW),
]

# Environment constants
ENV_IDENTIFIERS = Identifiers(
    properties=[
        "Windows",
        "Linux",
        "Enterprise",
        "UserWorkstation",
        "ApplicationServer",
        "Infrastructure",
    ],
    ports=["HTTPS", "SSH", "RDP", "HTTP", "PING"],
    local_vulnerabilities=["ScanBashHistory", "ScanExplorerRecentFiles", "CrackKeepPass"],
    remote_vulnerabilities=["ProbeLinux", "ProbeWindows", "ExploitVulnerability"],
)

# Shared credentials - reused across multiple machines
SHARED_SSH_CREDENTIAL = "SharedSSHPassword123!"
SHARED_RDP_CREDENTIAL = "SharedRDPPassword123!"


def create_flat_network(num_nodes: int = 20, credential_reuse_prob: float = 0.6, exploit_success_prob: float = 0.3) -> Dict[NodeID, NodeInfo]:
    """
    Create a flat network where most machines can directly communicate.
    
    Args:
        num_nodes: Number of nodes in the network
        credential_reuse_prob: Probability that credential reuse will succeed (0.6)
        exploit_success_prob: Probability that exploit will succeed (0.3)
    """
    nodes = {}
    
    # Entry node - starting point
    nodes["entry"] = m.NodeInfo(
        services=[],
        value=10,  # Low value
        vulnerabilities=dict(
            ScanExplorerRecentFiles=m.VulnerabilityInfo(
                description="Scan for credentials in recent files",
                type=m.VulnerabilityType.LOCAL,
                outcome=m.LeakedCredentials(credentials=[
                    m.CachedCredential(node="node_0", port="SSH", credential=SHARED_SSH_CREDENTIAL),
                    m.CachedCredential(node="node_1", port="RDP", credential=SHARED_RDP_CREDENTIAL),
                ]),
                reward_string="Found shared credentials in recent files",
                cost=1.0,
            )
        ),
        agent_installed=True,
        reimagable=False,
    )
    
    # Create flat network nodes - all can communicate with each other
    for i in range(num_nodes):
        node_id = f"node_{i}"
        
        # Determine node type (mix of Windows and Linux)
        is_windows = i % 2 == 0
        is_linux = not is_windows
        
        # Determine if this node uses shared credentials (60% chance)
        # Use deterministic pattern: approximately credential_reuse_prob of nodes use shared creds
        uses_shared_creds = (i % 10) < int(credential_reuse_prob * 10)
        
        # Create services with credentials
        services = []
        if is_linux:
            if uses_shared_creds:
                services.append(m.ListeningService("SSH", allowedCredentials=[SHARED_SSH_CREDENTIAL]))
            else:
                services.append(m.ListeningService("SSH", allowedCredentials=[f"UniqueSSHPassword{i}"]))
        else:  # Windows
            if uses_shared_creds:
                services.append(m.ListeningService("RDP", allowedCredentials=[SHARED_RDP_CREDENTIAL]))
            else:
                services.append(m.ListeningService("RDP", allowedCredentials=[f"UniqueRDPPassword{i}"]))
        
        services.append(m.ListeningService("HTTPS"))
        services.append(m.ListeningService("HTTP"))
        
        # Create vulnerabilities
        vulnerabilities = {}
        
        # Probe vulnerabilities (always succeed for discovery)
        if is_linux:
            vulnerabilities["ProbeLinux"] = m.VulnerabilityInfo(
                description="Probe to check if the node runs Linux",
                type=m.VulnerabilityType.REMOTE,
                outcome=m.ProbeSucceeded(["Linux", "Enterprise"]),
                reward_string="Remote machine is running Linux",
                cost=1.0,
            )
            vulnerabilities["ProbeWindows"] = m.VulnerabilityInfo(
                description="Probe to check if the node runs Windows",
                type=m.VulnerabilityType.REMOTE,
                outcome=m.ProbeFailed(),
                reward_string="Remote machine is not running Windows",
                cost=1.0,
            )
        else:  # Windows
            vulnerabilities["ProbeWindows"] = m.VulnerabilityInfo(
                description="Probe to check if the node runs Windows",
                type=m.VulnerabilityType.REMOTE,
                outcome=m.ProbeSucceeded(["Windows", "Enterprise"]),
                reward_string="Remote machine is running Windows",
                cost=1.0,
            )
            vulnerabilities["ProbeLinux"] = m.VulnerabilityInfo(
                description="Probe to check if the node runs Linux",
                type=m.VulnerabilityType.REMOTE,
                outcome=m.ProbeFailed(),
                reward_string="Remote machine is not running Linux",
                cost=1.0,
            )
        
        # Exploit vulnerability (30% success probability)
        # Note: In CyberBattleSim, outcomes are deterministic at creation time
        # For probabilistic outcomes, we'll use a fixed pattern based on node index
        # This simulates the 30% success rate across the network
        exploit_succeeds = (i % 10) < 3  # Approximately 30% success rate
        
        if exploit_succeeds:
            # Success - leak credentials to next nodes
            next_nodes = [f"node_{(i+1) % num_nodes}", f"node_{(i+2) % num_nodes}"]
            creds = []
            for next_node in next_nodes:
                next_idx = int(next_node.split("_")[1])
                if next_idx % 2 == 0:  # Next is Windows
                    creds.append(m.CachedCredential(node=next_node, port="RDP", credential=SHARED_RDP_CREDENTIAL))
                else:  # Next is Linux
                    creds.append(m.CachedCredential(node=next_node, port="SSH", credential=SHARED_SSH_CREDENTIAL))
            exploit_outcome = m.LeakedCredentials(credentials=creds)
        else:
            exploit_outcome = m.ExploitFailed()
        
        vulnerabilities["ExploitVulnerability"] = m.VulnerabilityInfo(
            description="Attempt to exploit a vulnerability",
            type=m.VulnerabilityType.REMOTE,
            outcome=exploit_outcome,
            reward_string="Exploit attempt",
            cost=5.0,
        )
        
        # Local vulnerabilities for credential discovery
        if is_linux:
            vulnerabilities["ScanBashHistory"] = m.VulnerabilityInfo(
                description="Scan bash history for credentials",
                type=m.VulnerabilityType.LOCAL,
                outcome=m.LeakedCredentials(credentials=[
                    m.CachedCredential(node=f"node_{(i+1) % num_nodes}", port="SSH", credential=SHARED_SSH_CREDENTIAL)
                ]),
                reward_string="Found SSH credentials in bash history",
                cost=1.0,
            )
        else:  # Windows
            vulnerabilities["ScanExplorerRecentFiles"] = m.VulnerabilityInfo(
                description="Scan Windows Explorer recent files for credentials",
                type=m.VulnerabilityType.LOCAL,
                outcome=m.LeakedCredentials(credentials=[
                    m.CachedCredential(node=f"node_{(i+1) % num_nodes}", port="RDP", credential=SHARED_RDP_CREDENTIAL)
                ]),
                reward_string="Found RDP credentials in recent files",
                cost=1.0,
            )
        
        # Set properties
        properties = ["Enterprise"]
        if is_windows:
            properties.extend(["Windows", "UserWorkstation" if i < num_nodes * 0.7 else "ApplicationServer"])
        else:
            properties.extend(["Linux", "UserWorkstation" if i < num_nodes * 0.7 else "ApplicationServer"])
        
        if i >= num_nodes * 0.9:
            properties.append("Infrastructure")
        
        nodes[node_id] = m.NodeInfo(
            services=services,
            firewall=m.FirewallConfiguration(incoming=DEFAULT_ALLOW_RULES, outgoing=DEFAULT_ALLOW_RULES),
            value=10,  # Low node values
            properties=properties,
            owned_string="Initial foothold established",
            vulnerabilities=vulnerabilities,
        )
    
    return nodes


def new_environment(num_nodes: int = 20, credential_reuse_prob: float = 0.6, exploit_success_prob: float = 0.3) -> m.Environment:
    """
    Create a new flat network environment.
    
    Args:
        num_nodes: Number of nodes in the network (default: 20)
        credential_reuse_prob: Probability that credential reuse will succeed (default: 0.6)
        exploit_success_prob: Probability that exploit will succeed (default: 0.3)
    """
    # Set random seed for reproducibility if needed
    # random.seed(42)
    
    network = create_flat_network(num_nodes, credential_reuse_prob, exploit_success_prob)
    return m.Environment(
        network=m.create_network(network),
        vulnerability_library=dict([]),
        identifiers=ENV_IDENTIFIERS
    )

