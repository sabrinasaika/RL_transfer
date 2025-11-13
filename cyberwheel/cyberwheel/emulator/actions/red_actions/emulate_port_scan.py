"""
Module defines the class to execute port scan action in the emulator.
"""

from __future__ import annotations
import os
from cyberwheel.emulator.actions import stdout_to_list
from .emulate_red_action_base import EmulateRedAction

file_path = os.path.realpath(__file__)
dir_name = os.path.dirname(file_path)


class EmulatePortScan(EmulateRedAction):
    """
    Class to exeucte ping sweep in the emulator.
    """

    name = "Network Service Discovery"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = EmulatePortScan.name

    def build_emulator_cmd(self):
        """
        Construct shell command to execute ping sweep script on emulator host VM.

        Returns:
            shell_cmd - full shell command that runs in a subprocess.
        """

        action_cmd_arr = [
            f"'echo ubuntu | sudo -S nmap -sS {self.target_host.ip_address}"
            r" | grep open | cut -d \" \" -f 1'"
        ]
        action_cmd = " ".join(action_cmd_arr)
        shell_cmd = self.prefix_emulator_cmd(action_cmd)
        return shell_cmd

