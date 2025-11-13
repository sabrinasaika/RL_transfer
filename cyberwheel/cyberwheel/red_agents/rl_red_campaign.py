import yaml

from typing_extensions import Tuple, Type

from cyberwheel.red_agents import ARTCampaign
from cyberwheel.red_agents.red_agent_base import RedAgentResult
import cyberwheel.red_agents.action_space as action_space
from cyberwheel.observation.red_observation import RedObservation
from cyberwheel.network.network_base import Network, Host
from cyberwheel.red_actions.red_base import RedActionResults
from cyberwheel.red_actions.technique import Technique
from cyberwheel.red_actions.art_techniques import RemoteSystemDiscovery, NetworkServiceDiscovery, SudoandSudoCaching, DataEncryptedforImpact, LinuxLateralMovement
from cyberwheel.red_actions.actions import Nothing
from cyberwheel.utils import HybridSetList

from copy import copy

import numpy as np
import random

class RLRedCampaign(ARTCampaign):
    def __init__(self, network: Network, args) -> None:
        super().__init__(network, args)
        self.observation = RedObservation(args, network)
        self.observation.add_host(self.current_host.name, on_host=True)
        
    def from_yaml(self) -> None:
        config = self.args.agent_config["red"]
        
        action_classes = [
            RemoteSystemDiscovery,
            NetworkServiceDiscovery,
            LinuxLateralMovement,
            SudoandSudoCaching,
            DataEncryptedforImpact,
        ]
        self.atomic_test = {
            "Remote System Discovery": RemoteSystemDiscovery.get_atomic_test("96db2632-8417-4dbb-b8bb-a8b92ba391de"),
            "Network Service Discovery": NetworkServiceDiscovery.get_atomic_test("515942b0-a09f-4163-a7bb-22fefb6f185f"),
            "LinuxLateralMovement": LinuxLateralMovement.get_atomic_test("uuid"),
            "Sudo and Sudo Caching": SudoandSudoCaching.get_atomic_test("150c3a08-ee6e-48a6-aeaf-3659d24ceb4e"),
            "Data Encrypted for Impact": DataEncryptedforImpact.get_atomic_test("08cbf59f-85da-4369-a5f4-049cffd7709f"),
        }
        self.entry_host: Host = "random"
        self.current_host : Host = self.network.get_random_user_host() #self.network.hosts[self.entry_host]

        self.leader = "random_server"
        self.leader_host = self.network.get_random_server_host() #self.network.hosts["server1"]

        self.action_space = getattr(action_space, config["action_space"])(action_classes, self.current_host.name)

        self.killchain = []
        self.reward_map = {
            "nothing": (0.0, 0.0),
            "Remote System Discovery": (0.0, 0.0),
            "Network Service Discovery": (5.0, 0.0),
            "Sudo and Sudo Caching": (0.0, 0.0),
            "Data Encrypted for Impact": (100.0, 0.0),
            "LinuxLateralMovement": (0.0, 0.0)
        }

    def act(self, action: int) -> RedAgentResult:
        self.handle_network_change() #TODO: Implement when developing static blue agent
        art_action, target_host_name = self.action_space.select_action(
            action
        )  # Selects ART Action, should include the action and target host (based on view?)
        source_host = self.current_host
        target_host = self.network.hosts[target_host_name] if target_host_name != "nothing" and target_host_name in self.network.hosts else "invalid"
        success = False
        if self.validate_action(art_action, target_host_name):
            action_results, action = self.run_action(target_host, art_action)
            success = action_results.attack_success
            self.handle_action(action_results)
        else:
            action_results = RedActionResults(source_host, target_host)
            
        return RedAgentResult(
            art_action, 
            source_host, 
            target_host, 
            success, 
            self.get_observation_space(),
            action_results=action_results
        )  # Returns what ARTAgent act() should, probably. Or the observation space? 
    
    def handle_network_change(self):
        current_hosts = self.network.all_hosts
        new_hosts = current_hosts.data_set - self.tracked_hosts.data_set
        removed_hosts = self.tracked_hosts.data_set - current_hosts.data_set
        #print(f"What I see: {self.tracked_hosts.data_set}")
        #print(f"All New Hosts: {current_hosts.data_set}")
        for h in new_hosts:
            host = self.network.hosts[h]
            if host.subnet.name in self.observation.known_subnets:
                #self.service_mapping[h] = self.get_valid_techniques_by_host(
                #    host, self.all_kcps
                #)
                #print(f"adding {h} to red obs and action space")
                self.observation.add_host(h, sweeped=True)
                self.action_space.add_host(h)
                self.tracked_hosts.add(h)
        for h in removed_hosts:
            #print(f"removing {h} from action space")
            self.action_space.remove_host(h)
            self.tracked_hosts.remove(h)

    def validate_action(self, action, target_host: str) -> bool:
            if action == Nothing:
                return True
            if target_host == "invalid" or target_host not in self.network.hosts:
                return False
            host_view = self.observation.obs[target_host]
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
        src_host = result.src_host.name
        target_host = result.target_host.name
        if action == RemoteSystemDiscovery:  # Adds pingsweeped hosts to obs
            self.observation.update_host(target_host, sweeped = True)
            current_subnet = result.target_host.subnet
            sweeped_hosts = random.sample(current_subnet.connected_hosts, len(current_subnet.connected_hosts))
            hosts = {h.name for host in sweeped_hosts for h in host.interfaces}
            #print(result.metadata)
            hosts |= {host.name for host in sweeped_hosts}
            #print(hosts)
            #interfaced_hosts = result.metadata["interfaced_hosts"]
            for h in hosts - self.observation.obs.keys():
                in_current_subnet = self.network.hosts[h].subnet.name == current_subnet.name
                self.observation.add_host(h, sweeped=in_current_subnet)
                self.action_space.add_host(h)
            #for h in set(interfaced_hosts):
            #    self.observation[h] = HostView(h)
            #    self.action_space.add_host(h)
        elif action == NetworkServiceDiscovery:  # Scans target host
            self.observation.update_host(target_host, scanned = True)
            self.observation.update_host(target_host, discovered = True)
            self.observation.update_host(target_host, type = self.network.get_node_from_name(
                target_host
            ).host_type.type)
        elif action == LinuxLateralMovement:  # Moves to target host
            self.observation.update_host(target_host, on_host = True)
            self.observation.update_host(src_host, on_host = False)
            self.current_host = result.target_host
        elif action == SudoandSudoCaching:
            self.observation.update_host(target_host, escalated = True)
        elif action == DataEncryptedforImpact:
            self.observation.update_host(target_host, impacted = True)

    def get_observation_space(self):
        """
        Takes red agent view of network and transforms it into the obs vector.
        """
        return np.array(self.observation.obs_vec, dtype=np.int64)
    
    def reset(self, network: Network, service_mapping: dict):
        super().reset(network, service_mapping)
        self.action_space.reset(self.current_host.name)
        self.tracked_hosts = HybridSetList()
        return self.observation.reset(network, self.current_host.name)
        

    def run_action(self, target_host: Host, art_action) -> Tuple[RedActionResults, Type[Technique]]:
        #self.leader = ["server01", "server02", "server03", "decoy01", "decoy02"]
        if art_action == Nothing:
            return RedActionResults(self.current_host, target_host), Nothing

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