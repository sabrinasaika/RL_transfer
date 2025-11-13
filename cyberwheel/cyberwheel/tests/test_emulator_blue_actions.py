"""
Module to test the Blue Agent actions in the emulator.
"""

import unittest
from cyberwheel.emulator.actions.blue_actions import (
    EmulateDeployDecoyHost,
    EmulateRemoveDecoyHost,
)
from cyberwheel.network.network_base import Network


class TestEmulatorBlueActions(unittest.TestCase):
    """Unit tests for the the emulator blue actions"""

    def test_deploy_decoy_host(self) -> None:
        """Test deploying decoy host"""
        action = EmulateDeployDecoyHost(network=Network("test"), configs={})
        decoy_hostname = "decoy01"
        shell_cmd = action.build_emulator_cmd(decoy_hostname)

        action_result = action.emulator_execute(shell_cmd)
        self.assertTrue(action_result.success)

    def test_remove_decoy_host(self) -> None:
        """Test removing decoy host"""
        action = EmulateRemoveDecoyHost(network=Network("test"), configs={})
        decoy_hostname = "decoy01"
        shell_cmd = action.build_emulator_cmd(decoy_hostname)

        action_result = action.emulator_execute(shell_cmd)
        self.assertTrue(action_result.success)
