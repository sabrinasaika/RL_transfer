"""
Module defines the class to execute lateral movement action in the emulator.

For this action, the attacker (source host) ssh's into the target host.
It is assumed the attacker knows the target's login password.
"""

from __future__ import annotations
import os
from .emulate_red_action_base import EmulateRedAction

file_path = os.path.realpath(__file__)
dir_name = os.path.dirname(file_path)


class EmulateLateralMovement(EmulateRedAction):
    """
    Class to exeucte ping sweep in the emulator.
    """

    name = "LinuxLateralMovement"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = EmulateLateralMovement.name

    def build_emulator_cmd(self):
        """
        Construct shell command to execute lateral movement in a linux VM.

        Returns:
            shell_cmd - full shell command that runs in a subprocess.
        """
        host_user = EmulateRedAction.emu_config["firewheel"]["host"]["username"]
        host_pwd = EmulateRedAction.emu_config["firewheel"]["host"]["password"]

        # This command ssh's into a target host, then prints the host's name to the terminal
        cmd_arr = [
            f"sshpass -p {host_pwd}",
            f"ssh -o StrictHostKeyChecking=accept-new {host_user}@{self.target_host.ip_address}",
            "hostname",
        ]
        action_cmd = " ".join(cmd_arr)
        full_action_cmd = self.prefix_emulator_cmd(action_cmd)
        return full_action_cmd

