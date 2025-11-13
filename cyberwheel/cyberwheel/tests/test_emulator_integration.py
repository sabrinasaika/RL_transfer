"""
Module to test the EmulatorController class and integration methods.

NOTES: ensure target decoy host is deployed (interface turned on) before running
red actions.
"""

import unittest
from cyberwheel.emulator.control import EmulatorControl
from cyberwheel.network.network_base import Network
from cyberwheel.network.host import Host
from importlib.resources import files


################### Build Network From Config ###################
NETWORK_CONFIG = "emulator_integration_config.yaml"
config_path = files("cyberwheel.data.configs.network").joinpath(NETWORK_CONFIG)
network = Network.create_network_from_yaml(config_path)
user_subnet = next(iter(network.subnets.values()))

####################### TEST NETWORK #########################
# from cyberwheel.network.router import Router
# from cyberwheel.network.subnet import Subnet

# network = Network(name="test")
# router = Router(name="core_router")
# user_subnet = Subnet(name="user_subnet", ip_range="192.168.0.0/24", router=router)
# server_subnet = Subnet(name="server_subnet", ip_range="192.168.1.0/24", router=router)


class TestEmulatorIntegration(unittest.TestCase):
    """Unit tests for the the emulator controller actions"""

    emulator = EmulatorControl(network=network, network_config_name=NETWORK_CONFIG)

    # get host IP addresses from emulator
    for name, h in emulator.network.hosts.items():
        print(f"retrieving ip address from emulator for {name}")
        host_name = name.replace("_", "-")
        emu_host_ip = emulator.get_ip_address(host_name)
        h.set_ip_from_str(emu_host_ip)

    def test_run_deploy_decoy_host(self) -> None:
        """
        Test executing a blue action, deploy decoy host, in the emulator.
        """
        action_name = "deploy_decoy"
        src_host_name = "subnet"  # ignored, chooses decoy in subnet

        blue_action_return = self.emulator.run_blue_action(action_name, src_host_name)
        self.assertTrue(blue_action_return.success)

    def test_run_remove_decoy_host(self) -> None:
        """
        Test executing a blue action, remove decoy host, in the emulator.
        """
        action_name = "remove_decoy_host"
        src_host_name = "decoy01"

        blue_action_return = self.emulator.run_blue_action(action_name, src_host_name)
        self.assertTrue(blue_action_return.success)

    def test_run_ping_sweep(self) -> None:
        """
        Test executing a red action, ping sweep, in the emulator.
        """
        action_name = "Remote System Discovery"
        src_host = Host(name="user01", subnet=user_subnet, host_type=None)

        # NOTE: ping sweep range defined in emulator_control.py
        red_action_return = self.emulator.run_red_action(
            action_name, src_host=src_host, dst_host=src_host
        )
        self.assertTrue(red_action_return.attack_success)

    def test_run_multi_subnet_ping_sweep(self) -> None:
        """
        Test executing multi-subnet ping sweep, in the emulator.
        """
        action_name = "Remote System Discovery"
        src_host = Host(name="user01", subnet=user_subnet, host_type=None)
        options = {
            "start_host": 2,
            "end_host": 10,
        }

        red_action_return = self.emulator.run_red_action(
            action_name, src_host=src_host, dst_host=src_host, options=options
        )
        print(
            f"all discovered hosts: {[host.name for host in red_action_return.discovered_hosts]}"
        )
        self.assertTrue(red_action_return.attack_success)

    def test_run_port_scan(self) -> None:
        """
        Test executing a red action, port scan, in the emulator.
        """
        action_name = "Network Service Discovery"
        src_host = Host(name="user01", subnet=user_subnet, host_type=None)
        target_host = Host(name="user02", subnet=user_subnet, host_type=None)
        target_host.set_ip_from_str("192.168.0.3")

        red_action_return = self.emulator.run_red_action(
            action_name, src_host=src_host, dst_host=target_host
        )
        self.assertTrue(red_action_return.attack_success)

    def test_run_sudo_and_sudo_caching(self) -> None:
        """
        Test executing a red action, sudo and sudo caching, in the emulator.
        """
        action_name = "Sudo and Sudo Caching"
        src_host = Host(name="decoy01", subnet=user_subnet, host_type=None)
        target_host = Host(name="decoy01", subnet=user_subnet, host_type=None)
        target_host.set_ip_from_str("192.168.0.8")

        red_action_return = self.emulator.run_red_action(
            action_name, src_host=src_host, dst_host=target_host
        )
        self.assertTrue(red_action_return.attack_success)

    def test_run_data_encrypted_for_impact(self) -> None:
        """
        Test executing a red action, data encrypted for impact, in the emulator.
        """
        action_name = "Data Encrypted for Impact"
        src_host = Host(name="decoy01", subnet=user_subnet, host_type=None)
        target_host = Host(name="decoy01", subnet=user_subnet, host_type=None)
        target_host.set_ip_from_str("192.168.0.8")

        red_action_return = self.emulator.run_red_action(
            action_name, src_host=src_host, dst_host=target_host
        )
        self.assertTrue(red_action_return.attack_success)

    def test_run_lateral_movement(self) -> None:
        """
        Test executing a red action, lateral movement, in the emulator.
        """
        action_name = "LinuxLateralMovement"
        src_host = Host(name="user01", subnet=user_subnet, host_type=None)
        target_host = Host(name="decoy01", subnet=user_subnet, host_type=None)
        target_host.set_ip_from_str("192.168.0.5")

        red_action_return = self.emulator.run_red_action(
            action_name, src_host=src_host, dst_host=target_host
        )
        self.assertTrue(red_action_return.attack_success)

    def test_run_get_siem_obs(self) -> None:
        """
        Test executing querying the SIEM and converting hits to alerts.
        """
        alerts = self.emulator.get_siem_obs()
        self.assertIsNotNone(alerts)

    def test_get_network_subnets(self) -> None:
        """
        Test retrieving subnets
        """
        print(list(self.emulator.network.subnets.values()))
        self.assertTrue(True)
