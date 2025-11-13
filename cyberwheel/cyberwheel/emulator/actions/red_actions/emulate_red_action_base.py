"""
Module defines the class to execute actions in the emulator.
"""

from __future__ import annotations
import os
from abc import ABC, abstractmethod
from importlib.resources import files
from cyberwheel.red_actions.red_base import ARTAction, RedActionResults
import subprocess
from subprocess import CompletedProcess
from typing import Any
from cyberwheel.emulator.utils import read_config


EMULATOR_CONFIG_PATH = files("cyberwheel.emulator").joinpath("configs")
EMULATOR_CONFIG = "emulator_config.yaml"


class EmulateRedAction(ARTAction, ABC):
    """
    Abstract class to exeucte actions in the emulator.
    User needs to implement the following abstract methods:

        - shell_script_name() - the name of the shell script that contains the action.
        - build_emulator_cmd() - the shell command that ssh's into the emulator and calls the action script.
        - emulator_execute() - runs the shell command.
    """

    emu_config = read_config(EMULATOR_CONFIG_PATH, EMULATOR_CONFIG)

    def __init__(self, src_host=None, target_host=None, network=None, ssh_pool=None):
        super().__init__(src_host=src_host, target_host=target_host)
        self.src_host = src_host
        self.target_host = target_host
        self.network = network

    @abstractmethod
    def build_emulator_cmd(self) -> str | type[NotImplementedError]:
        """Construct the full emulator command."""
        raise NotImplementedError

    def emulator_execute(
        self, shell_cmd: str
    ) -> RedActionResults | type[NotImplementedError]:
        """
        Execute red action in the emulator.

        Argrument:
            shell_cmd - shell command that executes shell script in emulator host VM.

        Returns
            RedActionResults
        """
        result = self.run_cmd(shell_cmd)
        self.action_results.attack_success = result.returncode == 0
        return result


    def prefix_emulator_cmd(self, action_cmd: str) -> str:
        """
        Pre-fixes the 'sshpass -p <password>' and 'firewheel ssh' command.

        Argrument:
            action_cmd - action command to execute which includes the shell script.

        Returns:
            final_cmd - full shell command as a string - sshpass + firewheel + action command.
        """
        host_user = EmulateRedAction.emu_config["firewheel"]["host"]["username"]
        host_pwd = EmulateRedAction.emu_config["firewheel"]["host"]["password"]

        self.action_results.action_cmd = f"{self.src_host.name} > {action_cmd}"

        # Ensure hostname is replaced with dash
        src_host_name = self.src_host.name.replace("_", "-")

        command_arr = [
            f"sshpass -p {host_pwd}",
            f"firewheel ssh {host_user}@{src_host_name}",
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

    def sim_execute(self) -> type[NotImplementedError]:
        """Not used in emulator."""
        return NotImplementedError

    def perfect_alert(self):
        """Not used in emulator."""
