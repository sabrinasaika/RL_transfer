"""
This plugin reads in the Cyberwheel network topology config
and creates the toplogy in Firehweel.
"""

import yaml
import os
from netaddr import IPNetwork
from base_objects import Switch
from linux.ubuntu_cyberwheel import (
    Ubuntu1604DesktopSiemCyberwheel,
    Ubuntu2204DesktopHostCyberwheel,
)
from vyos.helium118 import Helium118
from firewheel.control.experiment_graph import AbstractPlugin, Vertex
from typing import List

# Get network configuration file name from environment variable
NETWORK_CONFIG = os.environ["NETWORK_CONFIG"]

# Name of the interface used within Firewheel experiment;
# not the interface used by Firewheel ssh command.
HOST_INTERNAL_INTERFACE = "ens2"


class Plugin(AbstractPlugin):
    """cyberwheel.topology plugin."""

    def run(self):
        """Run method documentation."""

        # TODO - add check to ensure config exist
        config = read_config(NETWORK_CONFIG)

        # Create an external-facing network
        # self.external_network = IPNetwork("1.0.0.0/24")
        # Create an external-facing network iterator
        # external_network_iter = self.external_network.iter_hosts()

        # Create an internal facing network
        core_router = config.get("routers").get("core_router")
        core_router_networks = core_router.get("routes")[0].get("dest")
        internal_networks = IPNetwork(core_router_networks)  # e.g. 10.0.0.0/8

        # Break the internal network in to various subnets
        self.internal_subnets = internal_networks.subnet(24)

        # Create an internal switch
        internal_switch_name = "cyberwheel-internal-switch"
        internal_switch = Vertex(self.g, name=internal_switch_name)
        internal_switch.decorate(Switch)

        # Grab a subnet to use for connections to the internal switch
        internal_switch_network = next(self.internal_subnets)
        # Create a generator for the network
        internal_switch_network_iter = internal_switch_network.iter_hosts()
        print(
            f"\ncreated switch {internal_switch_name}, network: {core_router_networks}\n"
        )

        # Build Subnets
        subnets = config.get("subnets")
        for name, values in subnets.items():
            subnet_name = name.replace("_", "-")

            # Extract subnet's ip range
            subnet_ip = values.get("ip_range")
            subnet_network = IPNetwork(subnet_ip)

            # Break up subnet network into smaller subnets
            # internal_subnets = subnet_network.subnet(24)

            # Create subnets
            print(f"creating {subnet_name} with ip range {subnet_ip}...")
            num_hosts = num_hosts_in_subnet(name, config)  # use original subnet name

            # Skip subnet if no hosts
            if num_hosts == 0:
                print(f"no hosts in {subnet_name}, skip building subnet.\n")
                continue

            host_names = get_host_names_in_subnet(name, config)
            decoys = config.get("decoys")
            subnet_router = self.build_subnet(
                subnet_name, subnet_network, host_names, decoys
            )
            print(f"finished creating {subnet_name}")

            # Connect subnet to internal switch
            subnet_router.ospf_connect(
                internal_switch,
                next(internal_switch_network_iter),
                internal_switch_network.netmask,
            )
            print(f"connected {subnet_name} router to {internal_switch_name}\n")

            # Connect all connected subnets together
            subnet_router.redistribute_ospf_connected()

        # Add SIEM Subnet
        siem_network = IPNetwork("192.168.100.0/24")
        # siem_router = self.build_siem("siem", next(self.internal_subnets))
        siem_router = self.build_siem("siem", siem_network)
        siem_router.ospf_connect(
            internal_switch,
            next(internal_switch_network_iter),
            internal_switch_network.netmask,
        )
        siem_router.redistribute_ospf_connected()
        print(f"connected siem router to {internal_switch_name}\n")

    def build_subnet(
        self,
        subnet_name: str,
        network: IPNetwork,
        host_names: List[str],
        decoys: List[str] = [],
    ):
        """Build subnet

        Args:
            subnet_name: the name of the hosts subnet.
            network: the network object.
            host_names: names of the hosts in a specified subnet.

        Returns:
            vyos.Helium118: the subnet router object.
        """

        # Create subnet router
        full_subnet_name = f"{subnet_name}"
        subnet_router = Vertex(self.g, name=full_subnet_name)
        subnet_router.decorate(Helium118)

        # Create hosts switch
        subnet_switch_name = f"{subnet_name}-switch"
        subnet_switch = Vertex(self.g, name=subnet_switch_name)
        subnet_switch.decorate(Switch)

        # Create a network generator
        network_iter = network.iter_hosts()

        # Connet the router to the switch
        subnet_ip = next(network_iter)
        subnet_router.connect(subnet_switch, subnet_ip, network.netmask)
        print(
            f"connected {subnet_name} router to {subnet_switch_name}, {subnet_ip} {network.netmask}"
        )

        # Redistributes routes directly connected subnets to OSPF peers.
        subnet_router.redistribute_ospf_connected()

        # Create hosts
        for host_name in host_names:
            # Create a new host which are Ubuntu Desktops
            host_name = f"{host_name}"
            host = Vertex(self.g, name=host_name)
            host.decorate(Ubuntu2204DesktopHostCyberwheel)

            # Connect the host ot the switch
            host_ip = next(network_iter)
            host.connect(subnet_switch, host_ip, network.netmask)
            print(
                f"connected {host_name} to {subnet_switch_name}, {host_ip} {network.netmask}"
            )

            # Turn internal interface off for decoy hosts to initially hide
            if host_name in decoys:
                delay = -50
                host.run_executable(
                    delay, "/usr/sbin/ip", f"link set {HOST_INTERNAL_INTERFACE} down"
                )
                print(
                    f"turned off interface {HOST_INTERNAL_INTERFACE} for decoy host: {host_name}"
                )

        return subnet_router

    def build_siem(self, name: str, network: IPNetwork):
        """Build siem

        Args:
            name (str): the name of the user hosts subnet.
            network (netaddr.IPNetwork): the subnet for the user hosts.

        Returns:
            vyos.Helium118: The subnet router.
        """

        # Create SIEM subnet router
        siem_name = f"{name}-subnet"
        siem_router = Vertex(self.g, name=siem_name)
        siem_router.decorate(Helium118)

        # Create SIEM switch
        siem_switch_name = f"{name}-switch"
        siem_switch = Vertex(self.g, name=siem_switch_name)
        siem_switch.decorate(Switch)

        # Create a network generator
        network_iter = network.iter_hosts()

        # Connet the router to the switch
        siem_ip = next(network_iter)
        siem_router.connect(siem_switch, siem_ip, network.netmask)
        print(
            f"connected {siem_name} router to {siem_switch_name}, {siem_ip} {network.netmask}"
        )

        # Redistributes routes directly connected subnets to OSPF peers.
        siem_router.redistribute_ospf_connected()

        # Create SIEM host
        host_name = f"{name}"
        host = Vertex(self.g, name=host_name)
        host.decorate(Ubuntu1604DesktopSiemCyberwheel)

        # Connect the host ot the switch
        host_ip = "192.168.100.2"  # static ip address for siem
        host.connect(siem_switch, host_ip, network.netmask)
        print(
            f"connected {host_name} to {siem_switch_name}, {host_ip} {network.netmask}"
        )

        return siem_router


################################ Helper Functions #####################################


def read_config(name: str):
    """Read network config from YAML file"""
    cwd = os.getcwd()

    with open(f"{NETWORK_CONFIG}", "r", encoding="utf-8") as file:
        data = yaml.load(file, Loader=yaml.SafeLoader)
        return data


def num_hosts_in_subnet(subnet_name: str, config):
    """
    Returns number of hosts in a subnet

    Args:
        subnet_name: name of the subnet
        config: network configuration (scenario) YAML file

    Returns:
        num_hosts: number hosts in the subnet
    """

    hosts = config.get("hosts")
    num_hosts = sum(
        1 for _, values in hosts.items() if values.get("subnet") == subnet_name
    )
    return num_hosts


def get_host_names_in_subnet(subnet_name: str, config) -> List[str]:
    """
    Returns host names in a subnet

    Args:
        subnet_name: name of the subnet
        config: network configuration (scenario) YAML file

    Returns:
        host_names: names of hosts in a subnet
    """

    hosts = config.get("hosts")

    replace_underscore = lambda name: name.replace("_", "-")
    host_names = [
        replace_underscore(name)
        for name, values in hosts.items()
        if values.get("subnet") == subnet_name
    ]

    return host_names
