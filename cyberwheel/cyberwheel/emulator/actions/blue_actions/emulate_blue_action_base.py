"""
Module defines the class to execute actions in the emulator.
"""

from __future__ import annotations
import os
import subprocess
from abc import ABC, abstractmethod
from subprocess import CompletedProcess
from importlib.resources import files
from typing import Any

from cyberwheel.blue_actions.blue_action import SubnetAction, BlueActionReturn
from cyberwheel.emulator.utils import read_config
from cyberwheel.network.subnet import Subnet

EMULATOR_CONFIG_PATH = files("cyberwheel.emulator").joinpath("configs")
EMULATOR_CONFIG = "emulator_config.yaml"


class EmulateBlueAction(SubnetAction, ABC):
    """
    Abstract class to exeucte actions in the emulator.
    User needs to implement the following abstract methods:

        - shell_script_name()  - the name of the shell script that contains the action.
        - build_emulator_cmd() - the shell command that ssh's into the emulator
                                 and calls the action script.
        - emulator_execute()   - runs the shell command.
    """

    emu_config = read_config(EMULATOR_CONFIG_PATH, EMULATOR_CONFIG)

    @abstractmethod
    def build_emulator_cmd(self, *args, **kwargs) -> str | type[NotImplementedError]:
        """Construct the full emulator command."""
        raise NotImplementedError

    @abstractmethod
    def emulator_execute(
        self, shell_cmd: str
    ) -> BlueActionReturn | type[NotImplementedError]:
        """
        Execute red action in the emulator.

        Argrument:
            shell_cmd - shell command that executes shell script in emulator host VM.

        Returns
            BlueActionReturn
        """
        raise NotImplementedError

    def prefix_emulator_cmd(self, action_cmd: str) -> str:
        """
        Pre-fixes the 'sshpass -p <password>' and 'firewheel ssh' command.

        Argrument:
            action_cmd - action command to execute which includes the shell script.

        Returns:
            final_cmd - full shell command as a string - sshpass + firewheel + action command.
        """
        host_pwd = EmulateBlueAction.emu_config["firewheel"]["host"]["password"]

        command_arr = [
            f"sshpass -p {host_pwd}",
            action_cmd,
        ]
        final_cmd = " ".join(command_arr)
        return final_cmd

    def run_cmd(self, shell_cmd: str) -> CompletedProcess[Any]:
        """
        Run shell command that executes a script on emulator host VM.

        Argrument:
            shell_cmd: shell command to run in a subprocess.

        Returns:
           result: stdout or stderr from executing the shell command.
        """

        result = subprocess.run(
            shell_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True,
            text=True,
            check=True,
            # capture_output=True,
        )
        return result

    def execute(self, subnet: Subnet, **kwargs) -> BlueActionReturn:
        """Not used in emulator."""
        return BlueActionReturn()
