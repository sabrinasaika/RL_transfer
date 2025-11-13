"""
Red Actions Emulation Execution File

Runs actions in the emulator defined in the campaing action log (e.g. brute-force-campaing-test.csv).
You may defined the number of steps to run as an argurment (see command below). Default is 4 steps.

peotry run python test_emulator_red_action_campain.py <num_of_steps>
"""
import csv
import sys
from cyberwheel.network.host import Host
from cyberwheel.network.router import Router
from cyberwheel.network.subnet import Subnet
from cyberwheel.red_actions.red_base import RedActionResults

from cyberwheel.emulator.actions.red_actions import (
    EmulatePingSweep,
    EmulatePortScan,
    EmulateSudoandSudoCaching,
    EmulateDataEncryptedForImpact,
)

TOTAL_STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 4
print(f"Total Steps For Tests: {TOTAL_STEPS}\n")

# User IP range. Defined in network config
router = Router(name="192.168.0.0")
subnet = Subnet(name="192.168.0", ip_range="192.168.0.0", router=router)

with open('./test_files/brute-force-campaign-test.csv', mode ='r', encoding='utf8') as file:
    action_log = csv.DictReader(file)
    for step, line in enumerate(action_log):
        if step >= TOTAL_STEPS:
            break

        action = line["red_action_type"]
        src_hostname = line["red_action_src"]
        target_hostname = line["red_action_dest"]

        print(f"step: {step + 1}")
        print(f"action: {action}, source host: {src_hostname}, target host: {target_hostname}\n")

        src_host = Host(name=src_hostname, subnet=subnet, host_type=None)
        target_host = Host(name=target_hostname, subnet=subnet, host_type=None)
        results = RedActionResults

        match action:
            case "RemoteSystemDiscovery":
                print("executing Remote System Discoery...")

                red_action = EmulatePingSweep(src_host, target_host=src_host)

                START_HOST = 2
                END_HOST = 6
                print(f"scanning from {subnet.name + str(START_HOST)} to {subnet.name+ str(END_HOST)}")

                ping_sweep_cmd = red_action.build_emulator_cmd(
                    start_host=START_HOST, end_host=END_HOST, subnet=subnet.name
                )
                results = red_action.emulator_execute(ping_sweep_cmd)

            case "NetworkServiceDiscovery":
                print("execute Network Service Discovery...")

                target_host.set_ip_from_str("192.168.0.4") # user03
                red_action = EmulatePortScan(src_host, target_host)
                port_scan_cmd = red_action.build_emulator_cmd()
                results = red_action.emulator_execute(port_scan_cmd)

            case "SudoandSudoCaching":
                print("execute Sudo and Sudo Caching...")

                target_host.set_ip_from_str("192.168.0.4") # user03
                red_action = EmulateSudoandSudoCaching(src_host, target_host)
                sudo_caching_cmd = red_action.build_emulator_cmd()
                results = red_action.emulator_execute(sudo_caching_cmd)

            case "DataEncryptedforImpact":
                print("execute Data Encrypted for Impact...")

                target_host.set_ip_from_str("192.168.0.4") # user03
                red_action = EmulateDataEncryptedForImpact(src_host, target_host)
                data_encrypted_cmd = red_action.build_emulator_cmd()
                results = red_action.emulator_execute(data_encrypted_cmd)

            case _:
                print("action not found")

        print(f"Action Successfull: {results.attack_success}\n")
