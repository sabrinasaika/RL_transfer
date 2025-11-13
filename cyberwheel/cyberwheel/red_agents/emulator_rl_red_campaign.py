import yaml

from typing_extensions import Tuple, Type
from typing import Iterable

from cyberwheel.red_agents import RLRedCampaign
import cyberwheel.red_agents.action_space as action_space
from cyberwheel.red_agents.red_agent_base import RedAgentResult
from cyberwheel.network.network_base import Network, Host
from cyberwheel.red_actions.red_base import RedActionResults
from cyberwheel.red_actions.technique import Technique
from cyberwheel.red_actions.art_techniques import RemoteSystemDiscovery, NetworkServiceDiscovery, SudoandSudoCaching, DataEncryptedforImpact, LinuxLateralMovement
from cyberwheel.red_actions.actions import Nothing

import numpy as np

class EmulatorRLRedCampaign(RLRedCampaign):
    def __init__(self, network: Network, args) -> None:
        super().__init__(network, args)
    
    def add_host(self, h: Host, sweeped = True):
        ip = str(h.ip_address)
        self.mapping[ip] = h
        if ip not in self.observation.obs:
            self.observation.add_host(ip, ip_as_key=True, on_host=False, sweeped=sweeped)
        if ip not in self.action_space.hosts:
            self.action_space.add_host(ip)
            #print(self.observation)

    def from_yaml(self) -> None:
        config = self.args.agent_config["red"]
        
        action_classes = [
            RemoteSystemDiscovery,
            NetworkServiceDiscovery,
            SudoandSudoCaching,
            DataEncryptedforImpact,
            LinuxLateralMovement
        ]
        self.atomic_test = {
            "Remote System Discovery": RemoteSystemDiscovery.get_atomic_test("96db2632-8417-4dbb-b8bb-a8b92ba391de"),
            "Network Service Discovery": NetworkServiceDiscovery.get_atomic_test("515942b0-a09f-4163-a7bb-22fefb6f185f"),
            "Sudo and Sudo Caching": SudoandSudoCaching.get_atomic_test("150c3a08-ee6e-48a6-aeaf-3659d24ceb4e"),
            "Data Encrypted for Impact": DataEncryptedforImpact.get_atomic_test("08cbf59f-85da-4369-a5f4-049cffd7709f"),
            "LinuxLateralMovement": LinuxLateralMovement.get_atomic_test("uuid")
        }
        self.entry_host: Host = "random"
        self.current_host : Host = self.network.get_random_user_host() #self.network.hosts[self.entry_host]

        self.leader = "random_server"
        self.leader_host = self.network.get_random_server_host() #self.network.hosts["server1"]

        self.action_space = getattr(action_space, config["action_space"])(action_classes, str(self.current_host.ip_address))

        self.killchain = []
        self.reward_map = {
            "nothing": (0.0, 0.0),
            "Remote System Discovery": (0.0, 0.0),
            "Network Service Discovery": (0.0, 0.0),
            "Sudo and Sudo Caching": (0.0, 0.0),
            "Data Encrypted for Impact": (100.0, 10.0),
            "LinuxLateralMovement": (0.0, 0.0)
        }
    
    def select_action(self, action: int) -> RedAgentResult:
        art_action, target_host_ip = self.action_space.select_action(
            action
        )  # Selects ART Action, should include the action and target host (based on view?)
        #print(art_action)
        target_host = self.mapping[target_host_ip] if target_host_ip != "nothing" else self.current_host
        success = self.validate_action(art_action, str(target_host.ip_address))

        return RedAgentResult(
            art_action, 
            self.current_host, 
            target_host, 
            success, 
            self.get_observation_space()
        )

    def resolve_action(self, action_results: RedActionResults) -> Iterable:
        self.handle_action(action_results) # TODO: This handles info learned - pings sweeped, ports scanned, ssh success, sudo completed, data encrypted
        return self.get_observation_space()

    def validate_action(self, action, target_host: str) -> bool:
        if action == Nothing:
            return True
        if target_host not in self.observation.obs:
            return False
        host_view = self.observation.obs[target_host]
        #print(f"On Host: {host_view['on_host']}\nSweeped: {host_view['sweeped']}\nScanned: {host_view['scanned']}\nDiscovered: {host_view['discovered']}\nEscalated: {host_view['escalated']}\nImpacted: {host_view['impacted']}\n")
        if action == RemoteSystemDiscovery:  # valid if host["sweeped"] == False
            return not host_view["sweeped"]
        elif (
            action == NetworkServiceDiscovery
        ):  # valid if host["scanned"] == False and host["sweeped"] == True
            return host_view["sweeped"] and not host_view["scanned"]
        elif (
            action == LinuxLateralMovement
        ):  # valid if host["scanned"] && host["sweeped"] && host["discovered"] && !host.on_target
            return (
                host_view["sweeped"]
                and host_view["scanned"]
                and host_view["discovered"]
                and not host_view["on_host"]
            )
        elif (
            action == SudoandSudoCaching
        ):  # valid if host["scanned"] && host["sweeped"] && host["discovered"] && host.on_target && !host["escalated"]
            return (
                host_view["sweeped"]
                and host_view["scanned"]
                and host_view["discovered"]
                and host_view["on_host"]
                and not host_view["escalated"]
            )
        elif (
            action == DataEncryptedforImpact
        ):  # valid if host["scanned"] && host["sweeped"] && host["discovered"] && host.on_target && host["escalated"]
            return (
                host_view["sweeped"]
                and host_view["scanned"]
                and host_view["discovered"]
                and host_view["on_host"]
                and host_view["escalated"]
                and not host_view["impacted"]
            )
        else:
            return False
    
    def handle_action(self, result: RedActionResults) -> None:
        if not result.attack_success:
            return
        action = result.action
        src_host = str(result.src_host.ip_address)
        target_host = str(result.target_host.ip_address)
        if action == RemoteSystemDiscovery:  # Adds pingsweeped hosts to obs
            self.observation.update_host(target_host, sweeped=True)
            hosts = {str(host.ip_address): host for host in result.discovered_hosts}
            for h in hosts.keys() - self.observation.obs.keys():
                self.add_host(hosts[h])
        elif action == NetworkServiceDiscovery:  # Scans target host # TODO: Get from results.metadata (list of services)
            self.observation.update_host(target_host, scanned=True)
            self.observation.update_host(target_host, discovered=True)
            self.observation.update_host(target_host, type=self.mapping[target_host].host_type.type)
        elif action == LinuxLateralMovement:  # Moves to target host # TODO: Get by running command to see if on target host
            self.observation.update_host(target_host, on_host=True)
            self.observation.update_host(src_host, on_host=False)
            self.current_host = result.target_host
        elif action == SudoandSudoCaching: # TODO : Get by running command to see if in sudo mode on host
            self.observation.update_host(target_host, escalated=True)
        elif action == DataEncryptedforImpact: # TODO: Get by running command to see if file encrypted on host
            self.observation.update_host(target_host, impacted=True)

    def get_observation_space(self):
        """
        Takes red agent view of network and transforms it into the obs vector.
        """
        return np.array(self.observation.obs_vec, dtype=np.int64)
    
    def reset(self, network: Network, service_mapping: dict):
        super().reset(network, service_mapping)
        #entry_host = network.get_random_user_host()
        #self.network = network
        #self.current_host = entry_host
        ip = str(self.current_host.ip_address)
        self.mapping = {ip: self.current_host}
        self.action_space.reset(ip)
        return self.observation.reset(network, ip, ip_as_key=True)
        
        

    def run_action(self, target_host: Host, art_action) -> Tuple[RedActionResults, Type[Technique]]:
        self.leader = ["server01", "server02", "server03", "decoy01", "decoy02"]

        technique = art_action()
        atomic_test = self.atomic_test[art_action.get_name()]

        mitre_id = technique.mitre_id
        technique_name = technique.name

        action_results = RedActionResults(self.current_host, target_host)
        action_results.modify_alert(dst=target_host, src=self.current_host)

        # TODO: Checking if technique will work: OS match, CVE in cve_list, Killchain check
        action_results.add_successful_action()

        processes = []
        for dep in atomic_test.dependencies:
            processes.extend(dep.get_prerequisite_command)
            processes.extend(dep.prerequisite_command)
        if atomic_test.executor != None:
            processes.extend(atomic_test.executor.command)
            processes.extend(atomic_test.executor.cleanup_command)

        for p in processes:
            target_host.run_command(atomic_test.executor, p, "root")
        action_results.add_metadata(
            target_host.name,
            {
                "commands": processes,
                "mitre_id": mitre_id,
                "technique": technique_name,
            },
        )
        action_results.action = art_action



        return action_results, art_action

    def get_reward_map(self):
        return self.reward_map
