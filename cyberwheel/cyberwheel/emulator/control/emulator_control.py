"""
Module defines the class and functions to perform setup actions in the emulator before an experiment begins.
"""

from __future__ import annotations
from importlib.resources import files
from cyberwheel.emulator.utils import read_config
from cyberwheel.emulator.actions.blue_actions import (
    EmulateDeployDecoyHost,
    EmulateRemoveDecoyHost,
)
from cyberwheel.emulator.actions.red_actions import (
    EmulatePing,
    EmulatePingSweep,
    EmulatePortScan,
    EmulateSudoandSudoCaching,
    EmulateDataEncryptedForImpact,
    EmulateLateralMovement,
)
from cyberwheel.emulator.detectors import EmulatorDectector
from cyberwheel.blue_agents.blue_agent import BlueAgentResult
from cyberwheel.red_actions.red_base import RedActionResults
from cyberwheel.detectors.alert import Alert
from cyberwheel.network.host import Host
from cyberwheel.network.network_base import Network
from typing import Any, Dict, List, Iterable
import subprocess
import random
import json


NETWORK_CONFIG_PATH = files("cyberwheel.data.configs").joinpath("network")
EMULATOR_CONFIG_PATH = files("cyberwheel.emulator").joinpath("configs")
EMULATOR_CONFIG = "emulator_config.yaml"


class EmulatorControl:
    """
    Class setup emulator before an experiment begins.
    """

    emu_config = read_config(str(EMULATOR_CONFIG_PATH), EMULATOR_CONFIG)
    HOST_USERNAME = emu_config["firewheel"]["host"]["username"]
    HOST_PASSWORD = emu_config["firewheel"]["host"]["password"]
    ELASTIC_USERNAME = emu_config["elastic"]["username"]
    ELASTIC_PASSWORD = emu_config["elastic"]["password"]
    SIEM_HOSTNAME = emu_config["firewheel"]["siem"]["hostname"]
    KIBANA_PORT = emu_config["kibana"]["port"]

    def __init__(self, network: Network, network_config_name: str):
        self.network = network
        self.net_config_name = network_config_name
        self.net_config = read_config(str(NETWORK_CONFIG_PATH), network_config_name)
        self.detector = EmulatorDectector(
            network_config=network_config_name, network=network
        )

    def init_hosts(self) -> bool:
        """Setup hosts and run scripts before an experiment begins."""
        # Action #1: Enroll non-decoy hosts' agent to fleet
        # print("\tEnrolling elastic agents into fleet server...")
        all_host_names = self._get_host_names()
        decoy_names = self._get_decoy_host_names()
        #non_decoy_names = all_host_names - decoy_names # [name for name in all_host_names if name not in decoy_names]
        enrolled_host_names = self._get_enrolled_host_names()
        successfully_enrolled = True

        for host_name in all_host_names:
            if host_name not in enrolled_host_names and host_name not in decoy_names:
                successfully_enrolled = self._enroll_agent_to_fleet(host_name)

        if not successfully_enrolled:
            print("Error with enrolling agent(s) into fleet.")
            return False

        # Add more setup actions here...

        # print("done")
        return True

    def reset(self) -> bool:
        """Sequence of actions to reset the emulator for each episode"""

        # print("removing decoys...")
        decoy_names = self._get_decoy_host_names()
        success = self._reset_decoys(decoy_names)

        return success

    def run_blue_action(
        self,
        action_name: str,
        src_host_name: str,
        id: str = "",
    ) -> BlueAgentResult:
        """Lookup and execute blue actions in the emulator."""

        shell_cmd = ""
        # print(f"executing emulator blue action: {action_name}, id: {id}")

        match action_name:
            case "deploy_decoy":
                action = EmulateDeployDecoyHost(network=self.network, configs={})

                # random pick decoy within subnet
                decoy_names = self._get_decoy_host_names()
                #random_int = random.randint(0, len(decoy_names) - 1)
                random_decoy_hostname = decoy_names.pop()

                shell_cmd = action.build_emulator_cmd(random_decoy_hostname)
                deploy_return = action.emulator_execute(shell_cmd)
                deploy_return.host = None
                if deploy_return.success:
                    host_name = random_decoy_hostname
                    emu_host_ip = self.get_ip_address(host_name)
                    decoys_reserve = self.network.decoys_reserve
                    for d in decoys_reserve:
                        if d.name == host_name:
                            d.set_ip_from_str(emu_host_ip)
                            deploy_return.host = d
                            break

                # Enroll Decoy's agent into Fleet to enable detector
                enrolled_host_names = self._get_enrolled_host_names()
                if random_decoy_hostname not in enrolled_host_names:
                    self._enroll_agent_to_fleet(random_decoy_hostname)
                else:
                    #print(
                    #    f"{random_decoy_hostname}'s agent is already enrolled into fleet, skipping.\n"
                    #)
                    pass

                return BlueAgentResult(action_name, deploy_return.id, deploy_return.success, deploy_return.recurring, target=deploy_return.host)

            case "remove_decoy_host":
                action = EmulateRemoveDecoyHost(network=self.network, configs={})
                shell_cmd = action.build_emulator_cmd(src_host_name)
                action_result = action.emulator_execute(shell_cmd)
                return BlueAgentResult(action_name, action_result.id, action_result.success, action_result.recurring, target=action_result.host)
            case "nothing":
                return BlueAgentResult(action_name, "nothing", False, 0)
            case _:
                #print("ERROR: This action does not exist!")
                return BlueAgentResult(action_name, "invalid", False, 0)

    def run_red_action(
        self,
        action_name: str,
        src_host: Host,
        dst_host: Host,
        id: str = "",
        options: Dict[str, Any] = {},
    ) -> RedActionResults:
        """Lookup and execute red actions in the emulator."""

        shell_cmd = ""
        # print(f"executing emulator red action: {action_name}, id: {id}")

        match action_name:
            case "Remote System Discovery":
                """
                This action will execute ping sweep and subsequent pings
                if the host as multiple network interfaces with connected hosts.
                Extra interfaces are defined in the network config.
                """
                action = EmulatePingSweep(
                    src_host=src_host, target_host=dst_host, network=self.network
                )

                # Get ip_range of from subnet src_host is on
                src_host_subnet = src_host.subnet
                ip_range = src_host_subnet.ip_range

                # Limit ping sweep range to 'xxx.xxx.xxx.2-20' to prevent long action time.
                # If number of hosts on the subnet is greater, update end_host.
                if not options:
                    options = {
                        "start_host": 2,
                        "end_host": 10,
                    }  # will go to 2-254 if not defined

                # NOTE: ip_range will come from src_host if not provided
                shell_cmd = action.build_emulator_cmd(
                    start_host=options["start_host"],
                    end_host=options["end_host"],
                    ip_range=ip_range,
                )
                action.emulator_execute(shell_cmd)

                if not self._host_has_multi_interfaces(src_host):
                    return action.action_results

                # Host has another interface defined in config and will ping each connected host
                interfaces = self.net_config["interfaces"]
                connected_hosts = interfaces[src_host.name]

                # Collect all discovered hosts from original ping sweep
                all_discovered_hosts: list[Host] = []
                all_discovered_hosts.extend(action.action_results.discovered_hosts)

                # Execute ping
                shell_cmd = action.build_emulator_cmd()
                for host_name in connected_hosts:
                    conn_host = self.network.get_node_from_name(host_name)
                    ping = EmulatePing(
                        src_host=src_host, target_host=conn_host, network=self.network
                    )

                    shell_cmd = ping.build_emulator_cmd()
                    ping.emulator_execute(shell_cmd)
                    all_discovered_hosts.extend(ping.action_results.discovered_hosts)

                # Combine all discovered hosts
                action.action_results.discovered_hosts = all_discovered_hosts
                return action.action_results
            case "Network Service Discovery":
                action = EmulatePortScan(src_host=src_host, target_host=dst_host)
                shell_cmd = action.build_emulator_cmd()
                action.emulator_execute(shell_cmd)
                return action.action_results
            case "Sudo and Sudo Caching":
                action = EmulateSudoandSudoCaching(
                    src_host=src_host, target_host=dst_host
                )
                shell_cmd = action.build_emulator_cmd()
                action.emulator_execute(shell_cmd)
                return action.action_results
            case "Data Encrypted for Impact":
                action = EmulateDataEncryptedForImpact(
                    src_host=src_host, target_host=dst_host
                )
                shell_cmd = action.build_emulator_cmd()
                action.emulator_execute(shell_cmd)
                return action.action_results
            case "LinuxLateralMovement":
                action = EmulateLateralMovement(src_host=src_host, target_host=dst_host)
                shell_cmd = action.build_emulator_cmd()
                action.emulator_execute(shell_cmd)
                return action.action_results
            case _:
                #print(f"ERROR: This attack {action_name} does not exist.")
                results = RedActionResults(src_host=src_host, target_host=dst_host)
                results.attack_success = False
                return results

    def get_siem_obs(self) -> Iterable[Alert]:
        """
        Returns alerts converted from SIEM logs.

        Queries the last 5 minutes of activity and filters for red action activity.
        Any action done to a decoy generates an alert.
        """

        # print("\n")
        alerts = self.detector.obs()
        print(f"{len(alerts)} New SIEM Alerts:")
        for alert in alerts:
            print(f"Activity Detected on Host [{alert.src_host.name}]")
        return alerts

    def get_ip_address(self, host_name: str) -> str:
        """Returns emulator IP address."""
        host_user = self.emu_config["firewheel"]["host"]["username"]
        host_pwd = self.emu_config["firewheel"]["host"]["password"]

        command_arr = [
            f"sshpass -p {host_pwd}",
            f"firewheel ssh {host_user}@{host_name}",
            "ip -4 -brief address show | grep ens2 | awk '{print $3}'",
        ]
        cmd = " ".join(command_arr)

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True,
            text=True,
            check=True,
        )

        if result.returncode != 0:
            #print(f"ERROR: {result.stderr}")
            return ""
        elif result.stdout == "":
            return ""
        else:
            ip = result.stdout.split("/")[0]
            return ip

    def _get_host_names(self) -> List[str]:
        """Returns list of all host names."""
        # Firewheel host names cannot have "_" and are replaced with "-"
        # when creating the VM.
        return {name.replace("_", "-") for name in self.net_config["hosts"]}

    def _get_decoy_host_names(self) -> List[str]:
        """Returns list of all decoy host names."""
        # Firewheel host names cannot have "_" and are replaced with "-"
        # when creating the VM.
        return {name.replace("_", "-") for name in self.net_config["decoys"]}

    def _enroll_agent_to_fleet(self, host_name: str):
        """Enroll elastic agent to fleet server."""

        host_user = EmulatorControl.emu_config["firewheel"]["host"]["username"]
        host_pwd = EmulatorControl.emu_config["firewheel"]["host"]["password"]
        url = EmulatorControl.emu_config["fleet"]["server-url"]
        token = EmulatorControl.emu_config["fleet"]["enrollment-token"]

        cmd = f"""sshpass -p {host_pwd} firewheel ssh {host_user}@{host_name} \
        'echo {host_pwd} | sudo -S elastic-agent enroll -f \
        --url={url} \
        --enrollment-token={token} \
        --insecure'
        """

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True,
            text=True,
            check=True,
            # capture_output=True,
        )

        if result.returncode != 0:
            #print(f"ERROR: {result.stderr}")
            return False
        else:
            print(f"Successfully enrolled {host_name}'s elastic agent to fleet.")
            return True

    def _reset_decoys(self, hostnames: list[str]) -> bool:
        """Reset decoys by turning off network "ens2" interface"""

        for decoy_hostname in hostnames:
            action = EmulateRemoveDecoyHost(network=self.network, configs={})
            shell_cmd = action.build_emulator_cmd(decoy_hostname=decoy_hostname)
            result = action.emulator_execute(shell_cmd)
            if result.success:
                continue
            else:
                return False

        return True

    def _host_has_multi_interfaces(self, src_host: Host) -> bool:
        """Return True if host as multiple interfaces"""
        interfaces = self.net_config["interfaces"]

        if not interfaces:
            return False

        if src_host.name in interfaces.keys():
            # print(f"{src_host.name} has interface to {interfaces[src_host.name]}")
            return True

        return False
    

    def _get_enrolled_host_names(self) -> list[str]:
        """Retun a list of agent hostnames enrolled in fleet"""
        # Calls the Elastic API within the SIEM host, to retrieve hostnames enrolled in the SIEM
        cmd = f"""curl -X GET http://{self.ELASTIC_USERNAME}:{self.ELASTIC_PASSWORD}@localhost:{self.KIBANA_PORT}/api/fleet/agents"""

        result = self.run_command_on_host(self.SIEM_HOSTNAME, cmd)
        if not result:
            return []   # print(result.stdout)

        resDict = json.loads(result.stdout)
        return [item["local_metadata"]["host"]["name"] for item in resDict["items"]]

    def run_command_on_host(self, hostname, cmd):
        cmd = f"""sshpass -p {self.HOST_PASSWORD} firewheel ssh {self.HOST_USERNAME}@{hostname} \
        '{cmd}'
        """

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True,
            text=True,
            check=True,
        )

        if result.returncode != 0:
            #print(f"ERROR: {result.stderr}")
            return None
        return result