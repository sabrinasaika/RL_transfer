"""
Module defines the class to execute sudo and sudo caching in the emulator.
"""

from __future__ import annotations
import os
from .emulate_red_action_base import EmulateRedAction

file_path = os.path.realpath(__file__)
dir_name = os.path.dirname(file_path)


class EmulateSudoandSudoCaching(EmulateRedAction):
    """
    Class to exeucte sudo and sudo caching in the emulator.
    """

    name = "Sudo and Sudo Caching"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = EmulateSudoandSudoCaching.name

    def build_emulator_cmd(self):
        """
        Construct shell command to execute sudo and sudo caching.

        Returns:
            shell_cmd - full shell command that runs in a subprocess.
        """
        host_user = EmulateRedAction.emu_config["firewheel"]["host"]["username"]
        host_pwd = EmulateRedAction.emu_config["firewheel"]["host"]["password"]

        action_cmd_arr = [
            f"'sshpass -p {host_pwd}",
            f"ssh -o StrictHostKeyChecking=accept-new {host_user}@{self.target_host.ip_address}",
            f'"echo {host_pwd} | sudo -S -l;',
            f"echo {host_pwd} | sudo -S cat /etc/sudoers\"'",
            # Part of the attack to use vim to edit the sudoers file.
            # f"echo {host_pwd} | sudo -S vim /etc/sudoers\"'",
        ]
        action_cmd = " ".join(action_cmd_arr)
        shell_cmd = self.prefix_emulator_cmd(action_cmd)
        return shell_cmd