# Cyberwheel Domain Analysis

## Quick Reference: Object Types and Features

### **Router**
- **Type**: Network infrastructure node
- **Features**: Firewall rules, routing tables, interface IPs, default routes
- **Purpose**: Traffic management between subnets

### **Subnet**
- **Type**: Broadcast domain container
- **Features**: IP range (CIDR), DHCP pool, firewall rules, DNS server, connected hosts list
- **Purpose**: Network segment containing hosts

### **Host**
- **Type**: Network endpoint (workstation/server/decoy)
- **Features**:
  - **Identity**: Name, IP, MAC, OS (windows/macos/linux), host type
  - **State**: is_compromised, isolated, restored
  - **Network**: Subnet membership, firewall rules, routes, DNS, interfaces
  - **Security**: Services (port/protocol/version), CVEs, vulnerabilities
  - **Attack**: Processes, command history, artifacts
- **Purpose**: Attackable machine with services and vulnerabilities

### **Service**
- **Type**: Running application on host
- **Features**: Name, port, protocol (TCP/UDP/ICMP), version, CVEs, decoy flag
- **Purpose**: Attack surface entry point

### **CVE**
- **Type**: Vulnerability identifier
- **Features**: Links services to exploitable techniques
- **Purpose**: Enables attack techniques

---

## 1. Network Environments: Objects, Relations, and Features

### Network Topology Structure

Cyberwheel networks are represented as directed graphs (NetworkX DiGraph) with three primary object types:

#### **Network Objects**

1. **Routers**
   - **Purpose**: Manage network traffic between subnets
   - **Features**:
     - Firewall rules (source, destination, port, protocol)
     - Routing tables
     - Interface IP addresses (one per connected subnet)
     - Default routes
   - **Relations**: Connected to subnets via network edges

2. **Subnets**
   - **Purpose**: Represent broadcast domains and manage traffic between hosts
   - **Features**:
     - IP range (CIDR notation, e.g., 192.168.0.0/24)
     - Available IP addresses (DHCP pool)
     - Firewall rules
     - DNS server configuration
     - Connected hosts list
   - **Relations**: 
     - Connected to one router (parent router)
     - Contains multiple hosts
     - Connected to router via network graph edges

3. **Hosts**
   - **Purpose**: Machines/devices that belong to a subnet
   - **Features**:
     - **Basic Attributes**:
       - Name (unique identifier)
       - IP address (assigned via DHCP)
       - MAC address
       - Operating System (windows, macos, linux)
       - Host type (workstation, server, decoy)
     - **Security State**:
       - `is_compromised`: Boolean indicating if attacker has compromised the host
       - `isolated`: Boolean for network isolation status
       - `restored`: Boolean for restoration status
     - **Network Configuration**:
       - Subnet membership
       - Firewall rules
       - Default route and routing table
       - DNS server
       - Network interfaces
     - **Services & Vulnerabilities**:
       - List of running services (each with port, protocol, version)
       - CVE list (Common Vulnerabilities and Exposures)
       - Vulnerabilities list
     - **Attack Metadata**:
       - Processes list
       - Command history
       - Attack artifacts
   - **Relations**:
     - Belongs to one subnet (parent subnet)
     - Connected to subnet via network graph edge
     - Can have interfaces to other hosts (for special connectivity)

#### **Network Graph Structure**

- **Graph Type**: Directed graph (DiGraph) using NetworkX
- **Nodes**: Routers, Subnets, Hosts
- **Edges**: Network connectivity (router↔subnet, subnet↔host)
- **Network Organization**:
  - Routers connect subnets
  - Subnets contain hosts
  - Hosts can be isolated (edges removed while maintaining node)

#### **Host Types**

Hosts are categorized by type with different characteristics:

1. **Workstation/User Hosts**
   - Typically entry points for attackers
   - Lower privilege levels
   - May contain user data

2. **Server Hosts**
   - High-value targets
   - Often contain critical services
   - Primary objectives for many attack strategies

3. **Decoy Hosts**
   - Deployed by blue agent (defender)
   - Designed to attract and detect attackers
   - Can be dynamically added/removed
   - Have exploitable services to appear attractive

#### **Services**

Each service on a host has:
- Name (e.g., SSH, HTTP, RDP)
- Port number
- Protocol (TCP, UDP, ICMP)
- Version information
- Associated CVEs (vulnerabilities)
- Decoy flag (if service is part of a decoy)

### Network Relations Summary

```
Router
  └── Subnet (1:N)
        └── Host (1:N)
              └── Service (1:N)
                    └── CVE (1:N)
```

**Traffic Flow**: Host → Subnet → Router → Router → Subnet → Host

**Firewall Enforcement**: Applied at Router, Subnet, and Host levels (all must allow for traffic to pass)

---

## 2. Attacker (Red Agent) Action Space, Objectives, and Strategies

### Action Space

The red agent follows a **killchain-based attack sequence** with the following action types:

#### **Core Actions (Required Sequence)**

1. **ARTPingSweep**
   - **Purpose**: Discover hosts in a target subnet
   - **Target**: Subnet
   - **Effect**: Reveals all hosts in the subnet
   - **Success**: Always succeeds (information gathering)
   - **Reward**: Typically 0 (information gathering step)

2. **ARTPortScan**
   - **Purpose**: Scan a target host for open ports and services
   - **Target**: Specific host
   - **Effect**: Reveals services, ports, and vulnerabilities on the host
   - **Success**: Always succeeds (information gathering)
   - **Reward**: Typically 0 (information gathering step)

3. **ARTDiscovery**
   - **Purpose**: Execute discovery techniques to learn about the host
   - **Target**: Specific host
   - **Effect**: 
     - Executes Atomic Red Team discovery techniques
     - Reveals host type (Server/Workstation)
     - Exposes CVEs on the host
   - **Success**: Depends on technique validity (OS, killchain phase, CVE match)
   - **Reward**: Immediate reward (e.g., 20 points)

4. **ARTLateralMovement**
   - **Purpose**: Move from current host to target host
   - **Target**: Destination host
   - **Effect**: 
     - Attempts to compromise and move to target host
     - Updates current_host location
   - **Success**: Depends on technique validity and network connectivity
   - **Reward**: Typically 0 (movement step)

5. **ARTPrivilegeEscalation**
   - **Purpose**: Escalate privileges to root/admin level
   - **Target**: Current host
   - **Effect**: 
     - Executes privilege escalation techniques
     - Escalates privileges to root
     - Marks host as escalated
   - **Success**: Depends on technique validity
   - **Reward**: Immediate reward (varies by strategy)

6. **ARTImpact**
   - **Purpose**: Cause impact on the target (e.g., data encryption, service disruption)
   - **Target**: Current host
   - **Effect**: 
     - Executes impact techniques (e.g., data encryption)
     - Marks host as impacted
     - Achieves primary objective
   - **Success**: Depends on technique validity
   - **Reward**: High immediate reward (e.g., 100 points) - primary objective

#### **Action Execution Logic**

The red agent follows a **sequential killchain** on each host:
1. PingSweep subnet (if not scanned)
2. PortScan host (if not scanned)
3. LateralMovement to host (if not on host)
4. Discovery → PrivilegeEscalation → Impact (killchain phases)

**Technique Selection**: Actions use Atomic Red Team (ART) techniques that are validated based on:
- Target host's OS compatibility
- Killchain phase match
- CVE overlap between technique and host

### Objectives

The red agent's objectives vary by strategy but generally include:

1. **Server Downtime Strategy**:
   - Find and impact all servers in the network
   - Prioritize unimpacted servers
   - Explore unknown hosts that might be servers

2. **Exfiltration Strategy**:
   - Steal data from specific target hosts (leader hosts)
   - Focus on data exfiltration rather than service disruption

3. **Brute Force Strategy**:
   - Systematically attack all hosts
   - No specific prioritization

4. **BFS/DFS Strategies**:
   - Breadth-first or depth-first exploration
   - Systematic network traversal

### Strategies

Red agent strategies determine **target selection** logic:

1. **ServerDowntime** (Default)
   - Prioritizes servers over workstations
   - Continues attacking current host if it's a server or unknown
   - Selects from unimpacted servers pool
   - Falls back to unknown hosts if no servers available

2. **Exfiltration**
   - Targets specific "leader" hosts (configured or random)
   - Focuses on data exfiltration objectives

3. **DFSImpact** / **BFSServerDowntime** / **BFSExfiltration**
   - Graph traversal strategies (depth-first or breadth-first)
   - Systematic exploration patterns

4. **BruteForce**
   - No prioritization, attacks all hosts systematically

### Observation Space

The red agent has a **limited, expanding view** of the network:

**Host-level Attributes** (per known host):
- `type`: Host type (Server/User/Unknown)
- `sweeped`: Whether subnet has been ping-swept
- `scanned`: Whether host has been port-scanned
- `discovered`: Whether host type is known
- `on_host`: Whether attacker is currently on this host
- `escalated`: Whether privileges have been escalated
- `impacted`: Whether host has been impacted

**Standalone Attributes**:
- Quadrant attribute (positional information)

**Knowledge Expansion**:
- Initially knows only entry host
- Discovers hosts through:
  - PingSweep (discovers all hosts in scanned subnet)
  - PortScan (reveals host details)
  - Network changes (new decoys appear if subnet is scanned)

---

## 3. Defender (Blue Agent) Action Space, Objectives, and Strategies

### Action Space

The blue agent uses a **discrete action space** with configurable actions:

#### **Primary Actions**

1. **Nothing**
   - **Purpose**: No action taken
   - **Type**: Standalone action
   - **Reward**: 0 (no cost, no benefit)

2. **DeployDecoyHost**
   - **Purpose**: Deploy a decoy host to attract and detect attackers
   - **Type**: Subnet action (requires subnet selection)
   - **Target**: A subnet in the network
   - **Effect**: 
     - Creates a new decoy host with exploitable services
     - Decoy appears in network and can be discovered by attacker
     - If attacker interacts with decoy, alerts are generated
   - **Constraints**: 
     - Limited by `max_decoys` parameter
     - Cannot exceed decoy limit
   - **Reward**: Typically 0 immediate, 0 recurring
   - **Shared Data**: May use isolate_data for coordination

3. **RemoveDecoyHost**
   - **Purpose**: Remove a deployed decoy host
   - **Type**: Subnet action (requires subnet selection)
   - **Target**: A subnet containing a decoy
   - **Effect**: 
     - Removes decoy from network
     - Frees up decoy slot
   - **Reward**: Typically -1.0 immediate (cost of removal)

4. **IsolateDecoy** (if implemented)
   - **Purpose**: Isolate a decoy to contain attacker
   - **Type**: Subnet action
   - **Effect**: Network isolation of decoy host

#### **Action Space Structure**

- **Discrete Action Space**: Integer actions mapped to (action_type, target) pairs
- **Action Types**:
  - `standalone`: No target required (e.g., Nothing)
  - `subnet`: Requires subnet selection (e.g., DeployDecoyHost, RemoveDecoyHost)
- **Action Selection**: Action space converts discrete integer to (action_class, target_subnet) tuple

### Objectives

The blue agent's primary objectives:

1. **Detect Attacks**: 
   - Use decoys to attract attackers
   - Generate alerts when attackers interact with decoys
   - Monitor network for suspicious activity

2. **Slow/Stop Attackers**:
   - Deploy decoys strategically to waste attacker time
   - Isolate compromised hosts
   - Prevent attackers from reaching critical targets

3. **Minimize Impact**:
   - Prevent attackers from impacting servers
   - Protect high-value assets
   - Maintain network availability

4. **Resource Management**:
   - Manage limited decoy budget
   - Balance decoy deployment with removal
   - Optimize decoy placement

### Strategies

Blue agent strategies are **learned through RL** (no predefined heuristics for RL agent):

- **Decoy Placement**: Learn optimal subnet selection for decoy deployment
- **Decoy Management**: Learn when to deploy vs. remove decoys
- **Timing**: Learn optimal timing for defensive actions
- **Resource Allocation**: Learn to balance decoy budget

### Observation Space

The blue agent has **full network visibility** with alert-based information:

**Alert-based Attributes** (per host):
- **Current Alerts**: Binary indicator (0/1) for each host if alert triggered this step
- **Alert History**: Binary indicator (0/1) for each host if alert triggered previously
- **Alert Structure**: 2 × num_hosts (current + history)

**Standalone Attributes**:
- `num_decoys_deployed`: Count of currently deployed decoys

**Alert Generation**:
- Alerts generated by detectors when:
  - Attacker performs actions on hosts with detectors
  - Attacker interacts with decoys
  - Suspicious activity detected
- Detector system uses SIEM-like functionality (Sysmon, Elasticsearch)

**Observation Vector Structure**:
```
[alert_current_host1, alert_current_host2, ..., alert_history_host1, alert_history_host2, ..., num_decoys_deployed]
```

---

## 4. Strategic Knowledge/Skills Required for Good Performance

### Attacker (Red Agent) - Key Learnable Skills

**Network Exploration**:
- Subnet scanning strategy (which subnets to scan first)
- Host reconnaissance (port scanning, service discovery)
- Network topology mapping (connectivity, firewall rules)

**Attack Execution**:
- Target prioritization (servers vs. workstations, objective-based)
- Killchain sequencing (Discovery → Privilege Escalation → Impact)
- Technique selection (OS/CVE compatibility, exploit matching)
- Lateral movement planning (when to move vs. continue)

**Defensive Countermeasures**:
- Decoy recognition (identify and avoid/exploit decoys)
- Alert avoidance (stealthy attack patterns, detector awareness)
- Path planning (efficient routes to targets, exploration vs. exploitation)

**Strategic Decision Making**:
- Resource management (when to persist vs. switch targets)
- Partial observability handling (make decisions with limited network view)

### Defender (Blue Agent) - Key Learnable Skills

**Threat Detection**:
- Alert interpretation (pattern recognition, false positive filtering)
- Attack pattern recognition (killchain phases, lateral movement detection)
- Attack prediction (anticipate next moves, understand attacker strategies)

**Decoy Strategy**:
- Optimal placement (which subnets, attacker discovery paths)
- Timing (when to deploy/remove, early vs. late intervention)
- Attractiveness (make decoys appear as high-value targets)

**Defense Planning**:
- Critical asset protection (prioritize servers, high-value hosts)
- Resource allocation (decoy budget management, proactive vs. reactive)
- Multi-step planning (deployment sequences, long-term consequences)

**Strategic Decision Making**:
- Response timing (optimal intervention points)
- State estimation (infer attacker knowledge/position from alerts)

### Shared Skills (Both Agents)

**Network Understanding**:
- Topology awareness (structure, connectivity, firewall/routing)
- Service-vulnerability mapping (exploitable services, CVE-to-technique relationships)

**Adversarial Reasoning**:
- Opponent modeling (learn opponent's patterns, adapt strategies)
- Game-theoretic thinking (zero-sum balance, risk/reward, mixed strategies)

**Partial Observability**:
- Information gathering (efficient exploration, alert inference)
- State estimation (maintain mental models of unobserved components)

---

## Summary

Cyberwheel is a **multi-agent adversarial RL environment** where:

- **Network**: Hierarchical structure (Routers → Subnets → Hosts → Services) with firewall enforcement
- **Red Agent**: Follows killchain-based attacks (Discovery → Privilege Escalation → Impact) with limited, expanding network view
- **Blue Agent**: Deploys decoys and uses detectors to detect and slow attackers with full network visibility but alert-based information
- **Key Skills**: Network exploration, attack planning, decoy strategy, threat detection, and adversarial reasoning under partial observability

The environment emphasizes **strategic decision-making** where both agents must learn complex policies to achieve their objectives while adapting to their opponent's behavior.

