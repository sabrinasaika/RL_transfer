"""
Module to test the red actions in the emulator.
"""

import unittest
from cyberwheel.emulator.actions.red_actions import (
    EmulatePing,
    EmulatePingSweep,
    EmulatePortScan,
    EmulateSudoandSudoCaching,
    EmulateDataEncryptedForImpact,
    EmulateLateralMovement,
)
from importlib.resources import files
from cyberwheel.network.host import Host
from cyberwheel.network.network_base import Network
from cyberwheel.network.router import Router
from cyberwheel.network.subnet import Subnet

# TEST variables
config_path = files("cyberwheel.data.configs.network").joinpath(
    "emulator_integration_config.yaml"
)
network = Network.create_network_from_yaml(config_path)
router = Router(name="192.168.1.0")
subnet = Subnet(name="192.168.1.0", ip_range="192.168.1.0", router=router)


class TestEmulatorRedActions(unittest.TestCase):
    """Unit tests for the the emulator red actions."""

    def test_ping(self) -> None:
        """Test single ping."""
        src_host = Host(name="user01", subnet=subnet, host_type=None)
        target_host = Host(name="user02", subnet=subnet, host_type=None)
        target_host.set_ip_from_str("192.168.0.3")

        red_action = EmulatePing(src_host, target_host=target_host, network=network)
        print(red_action.__class__.get_name())

        ping_sweep_cmd = red_action.build_emulator_cmd()

        results = red_action.emulator_execute(ping_sweep_cmd)
        self.assertTrue(results.attack_success)

    def test_ping_sweep(self) -> None:
        """Test ping sweep in emulator."""
        src_host = Host(name="user01", subnet=subnet, host_type=None)
        red_action = EmulatePingSweep(src_host, target_host=src_host, network=network)
        print(red_action.__class__.get_name())

        ping_sweep_cmd = red_action.build_emulator_cmd(
            start_host=3, end_host=9, ip_range="192.168.0.0/24"
        )

        results = red_action.emulator_execute(ping_sweep_cmd)
        self.assertTrue(results.attack_success)

    def test_port_scan(self) -> None:
        """Test port scan in emulator."""
        src_host = Host(name="user01", subnet=subnet, host_type=None)
        target_host = Host(name="decoy01", subnet=subnet, host_type=None)
        target_host.set_ip_from_str("192.168.0.5")

        red_action = EmulatePortScan(src_host, target_host)
        print(red_action.__class__.get_name())

        port_scan_cmd = red_action.build_emulator_cmd()
        results = red_action.emulator_execute(port_scan_cmd)
        self.assertTrue(results.attack_success)

    def test_sudo_and_sudo_caching(self) -> None:
        """Test sudo and sudo cashing in emulator."""
        src_host = Host(name="user01", subnet=subnet, host_type=None)
        target_host = Host(name="decoy01", subnet=subnet, host_type=None)
        target_host.set_ip_from_str("192.168.0.5")

        red_action = EmulateSudoandSudoCaching(src_host, target_host)
        print(red_action.__class__.get_name())

        sudo_caching_cmd = red_action.build_emulator_cmd()
        results = red_action.emulator_execute(sudo_caching_cmd)
        self.assertTrue(results.attack_success)

    def test_data_encrypted_for_impact(self) -> None:
        """Test data encrypted for impact in emulator."""
        src_host = Host(name="user01", subnet=subnet, host_type=None)
        target_host = Host(name="user02", subnet=subnet, host_type=None)
        target_host.set_ip_from_str("192.168.0.3")

        red_action = EmulateDataEncryptedForImpact(src_host, target_host)
        print(red_action.__class__.get_name())

        data_encrypted_cmd = red_action.build_emulator_cmd()
        results = red_action.emulator_execute(data_encrypted_cmd)
        self.assertTrue(results.attack_success)

    def test_lateral_movement(self) -> None:
        """Test data lateral movement in emulator."""
        attacker = Host(name="user01", subnet=subnet, host_type=None)
        user_host = Host(name="decoy01", subnet=subnet, host_type=None)
        user_host.set_ip_from_str("192.168.0.8")

        red_action = EmulateLateralMovement(src_host=attacker, target_host=user_host)
        print(red_action.__class__.get_name())

        data_encrypted_cmd = red_action.build_emulator_cmd()
        results = red_action.emulator_execute(data_encrypted_cmd)
        self.assertTrue(results.attack_success)
