# Emulator Red Actions

This README describes how to add new emulator red actions. Each red action has its own class and action shell script. The class is reponsible for building the shell command to ssh into host within Firewheel (the attacker VM) and execute the the action on the host . The shell script is the action itself that host VM runs (e.g. ping sweep).

## Important Notes

- The `scripts` folder and individual shell scripts must be copied to the attacker host's **/home** directory, in Firewheel, before executing an experiment.

## How to Add New Red Actions

1. Create a new shell script in the `scripts/` folder.
    - The file should begin with with which operating system it is for (e.g. `linux_<script_name>.sh`).
    - If the script returns captures values, they should output to stdout and each value be on a new line.
    - See `linux_ping_sweep.sh` as an example.
2. Create a new class for the action.
    - The name should start with "emulate" (e.g. `emulate_<action_name>.py`).
    - The class needs to inherit `EmulateRedAction`
    - Implement the following methods:
        - `shell_script_name()` - returns the name of the action shell script. It should include the folder name (e.g. scripts/<name>.sh).
        - `build_emulator_command()` - returns the shell command that will ssh into host and call the action script with `bash`. The command should include any argurments the action script requires. Use the `prefix_emulator_cmd(cmd)` to prefix `sshpass` and `firewheel ssh` to construct the full command. An usage is bellow:
            ```
            cmd = f"bash {self.shell_script_name}
            full_cmd = self.prefix_emulator_cmd(cmd)
            return full_cmd
            ```
        - `emulator_execute()` - runs the shell command, captures output if needed and returns a RedActionResults.
    - See `emulate_ping_sweep.py` as an example