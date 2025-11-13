"""
Module defines the class to deploy a decoy in the emulator.

Note decoys are currenlty pre-deployed when starting an experiment.
This action emulates deploying a decoy by turning on the network interface.
Once turned on the decoy machine will be visible to other machines in the subnet.
"""

from __future__ import annotations
from .emulate_blue_action_base import EmulateBlueAction
from cyberwheel.blue_actions.blue_action import BlueActionReturn


class EmulateDeployDecoyHost(
    EmulateBlueAction
):  # pylint: disable=too-few-public-methods
    """
    Class to deploy decoy in the emulator.
    """

    name = "deploy_decoy"

    def build_emulator_cmd(self, decoy_hostname: str) -> str:
        """
        Construct shell command to execute deploy decoy host in firewheel.
        This comand turn on the network interface which will allow the (pre-made)
        decoy to be visitble to other hosts.

        Returns:
            shell_cmd - full shell command that runs in a subprocess.
        """
        host_username = EmulateBlueAction.emu_config["firewheel"]["host"]["username"]
        interface = EmulateBlueAction.emu_config["firewheel"]["interface"]

        action_cmd_arr = [
            f"firewheel ssh {host_username}@{decoy_hostname}",
            f"'echo ubuntu | sudo -S ip link set {interface} up'",
        ]
        action_cmd = " ".join(action_cmd_arr)

        shell_cmd = self.prefix_emulator_cmd(action_cmd)
        return shell_cmd

    def emulator_execute(self, shell_cmd: str) -> BlueActionReturn:
        """
        Exeucute deploying a decoy in the emulator.
        """
        # Execute Deploy Decoy Host VM
        #print(f"executing shell command: {shell_cmd}")
        result = self.run_cmd(shell_cmd)

        isSuccessful = False  # assume False
        if result.returncode != 0:
            #print(result.stderr)
            pass
        else:
            isSuccessful = True
            #print(result.stdout)

        return BlueActionReturn(self.name, isSuccessful, 0)
