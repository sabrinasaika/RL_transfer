"""
Module defines the class to execute ping sweep action in the emulator.
"""

from __future__ import annotations
import os
from cyberwheel.emulator.actions import stdout_to_list
from .emulate_red_action_base import EmulateRedAction
from cyberwheel.network.network_base import Network

file_path = os.path.realpath(__file__)
dir_name = os.path.dirname(file_path)


class EmulatePingSweep(EmulateRedAction):
    """
    Class to exeucte ping sweep in the emulator.
    """

    name = "Remote System Discovery"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = EmulatePingSweep.name
        self.network = kwargs.get("network", None)

    def build_emulator_cmd(
        self,
        start_host: int = 2,
        end_host: int = 254,
        ip_range: str = "",
    ):
        """
        Construct shell command to execute ping sweep script on emulator host VM.

        Argruments:
            start_host - start ip range value (e.g. 1)
            end_host - end ip range value (e.g. start_host< value <=254)
            subnet - host subnet (e.g. 192.168.1)

        Returns:
            shell_cmd - full shell command that runs in a subprocess.
        """
        # if not provided subnet, use source host subnet ip_range for scan
        if not ip_range:
            src_host_subnet = self.src_host.subnet
            ip_range = src_host_subnet.ip_range

        # remove CIDR
        ip_split = ip_range.split(".")
        subnet = ".".join(ip_split[:-1])  # drop last element

        action_cmd_arr = [
            f"'for ip in $(seq {start_host} {end_host});",
            f"do ping -c 1 {subnet}.$ip;",
            f"[ $? -eq 0 ] && echo {subnet}.$ip UP || :;",
            r"done | grep UP | cut -d \" \" -f 1'",
        ]
        action_cmd = " ".join(action_cmd_arr)

        shell_cmd = self.prefix_emulator_cmd(action_cmd)
        return shell_cmd

    def emulator_execute(self, shell_cmd: str):
        """
        Execute ping sweep in the emulator. Discovered IP addresses are added to
        discovered hosts in RedActionResults.

        Argrument:
            shell_cmd - shell command to execute a ping sweep in emulator host VM.

        Returns:
            RedActionResults
        """

        # Execute ping sweep in emulator VM
        #print(f"executing shell command: {shell_cmd}")
        result = super().emulator_execute(shell_cmd=shell_cmd)
        #result = self.run_cmd(shell_cmd)

        # Capture output after executing command
        discovered_ips: list[str] = []
        if self.action_results.attack_success:
            discovered_ips = stdout_to_list(result.stdout)
            #print("discovered ips: ", discovered_ips)

        """
            Add discovered hosts to red action results.
            The Host performing ping sweep is included.
            Decoys are not included.
        """
        for discovered_ip in discovered_ips:
            # NOTE: network.get_all_hosts() does not get decoys
            self.action_results.add_host(self.network.get_node_from_ip(discovered_ip))
            #for host in self.network.get_all_hosts():
            #    if str(host.ip_address) == discovered_ip:
            #        self.action_results.add_host(host)
        #print(
        #    f"added discovered hosts to action results: {[host.name for host in self.action_results.discovered_hosts]}"
        #)

        return self.action_results
