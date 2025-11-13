"""
Module defines the Emulator Dectectory class.
"""

from __future__ import annotations
from cyberwheel.detectors.alert import Alert
from cyberwheel.detectors.detector_base import Detector
from cyberwheel.emulator.utils import read_config
from cyberwheel.network.host import Host, HostType
from cyberwheel.network.network_base import Network
from pprint import pprint
from subprocess import CompletedProcess
from typing import Any, Dict, Iterable, List, cast
from importlib.resources import files
import json
import re
import subprocess

# Librares for testing
# from cyberwheel.network.router import Router
# from cyberwheel.network.subnet import Subnet

# TEST variables
# decoy_ips = ["192.168.0.5", "192.168.0.6"]
# router = Router(name="192.168.0.0")
# subnet = Subnet(name="192.168.0.0", ip_range="192.168.0.0/24", router=router)

# Paths and file locations
NETWORK_CONFIG_PATH = files("cyberwheel.data.configs").joinpath("network")
EMULATOR_CONFIG_PATH = files("cyberwheel.emulator").joinpath("configs")
EMULATOR_CONFIG = "emulator_config.yaml"
QUERY_FILE = files("cyberwheel.emulator.detectors").joinpath("query.txt")


class EmulatorDectector(Detector):
    """
    Class to communicate with SIEM (elasticsearch) within the emulator.
    The detector, Sysmon, fowards information to the SIEM.
    """

    emu_config = read_config(str(EMULATOR_CONFIG_PATH), EMULATOR_CONFIG)

    def __init__(self, network_config: str, network: Network):
        self.network_config = read_config(str(NETWORK_CONFIG_PATH), network_config)
        self.network = network
        self.alert_ids = set()  # Keep history of alerts

    def query_to_json(self, result: CompletedProcess[str]) -> Any | None:
        """Converts SIEM query reponse to JSON."""
        return json.loads(result.stdout)

    def submit_test_query(self) -> CompletedProcess[str] | None:
        """SSHs into the host with the SIEM and submits a query."""
        siem_pwd = EmulatorDectector.emu_config["firewheel"]["siem"]["password"]
        siem_user = EmulatorDectector.emu_config["firewheel"]["siem"]["username"]
        siem_hostname = EmulatorDectector.emu_config["firewheel"]["siem"]["hostname"]

        cmd_arr = [
            f"sshpass -p {siem_pwd} firewheel ssh {siem_user}@{siem_hostname}",
            f"curl -u elastic:elastic -X GET http://localhost:9200",
        ]
        cmd = " ".join(cmd_arr)

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True,
            text=True,
            check=True,
        )

        if result.returncode != 0:
            error = {"error": result.stderr}
            print(json.dumps(error))
            return None
        else:
            print(result.stdout)

        return result

    def submit_obs_query(self) -> Dict[Any, Any]:
        """Shells into the SIEM VM and submits a query to get the oberstation state."""
        siem_pwd = EmulatorDectector.emu_config["firewheel"]["siem"]["password"]
        siem_user = EmulatorDectector.emu_config["firewheel"]["siem"]["username"]
        siem_hostname = EmulatorDectector.emu_config["firewheel"]["siem"]["hostname"]

        cmd_arr = [
            f"sshpass -p {siem_pwd} firewheel ssh {siem_user}@{siem_hostname}",
            f"$(cat {QUERY_FILE})",
        ]
        cmd = " ".join(cmd_arr)
        # print("Querying SIEM logs in the last 5 minutes...")
        # print(f"{cmd}\n")

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            shell=True,
        )

        if result.returncode != 0:
            error = {"error": result.stderr}
            print(json.dumps(error))
            return {}
        else:
            response = json.loads(result.stdout)
            # pprint(response)

        return response

    def parse_query_response(self, response: Dict[Any, Any]) -> List[Dict[Any, Any]]:
        hits = response["hits"]["hits"]
        parsed_hits = []
        # print("parsing logs into hits...\n")

        for hit in hits:
            # pprint(hit)
            id = hit["_id"]
            source = hit["_source"]
            timestamp = source["@timestamp"]
            hostname = source["host"]["hostname"]
            src_ip = source["host"]["ip"][2]
            command = source["process"]["command_line"]
            # command_args = source["process"]["args"]
            target_ip = self._get_target_ip((command))

            if not target_ip:
                continue

            parsed_hits.append(
                {
                    "id": id,
                    "timestamp": timestamp,
                    "hostname": hostname,
                    "src_ip": src_ip,
                    "target_ip": target_ip,
                    "command": command,
                }
            )

        return parsed_hits

    def create_alerts(self, hits: List[Dict[Any, Any]]) -> Iterable[Alert]:
        """
        Take the parsed logs (hits) and creates Alerts.
        Currenlty, Alerts are created when a target host is a decoy.
        """
        decoy_hits = self._find_decoy_hits((hits))
        alerts = []
        prev_dst_ip = ""

        for hit in decoy_hits:
            dst_ip = hit["target_ip"]

            # skip hits with same target ip (we're assuming its from in the same action)
            # if prev_dst_ip == dst_ip:
            #    continue

            # if 'ccencrypt' in hit["command"]:
            #    print("Impact Detected:")
            #    pprint(hit)

            # skip hits that have already been
            if hit["id"] in self.alert_ids:
                #print(f"found duplicate decoy hit, skipping id {hit['id']}")
                continue

            #print("found new decoy hit, creating alert...")
            #pprint(hit)
            self.alert_ids.add(hit["id"])  # save id

            # get the source emulator Host name
            src_emu_hostname = hit["hostname"]
            # hosts defined with underscores in config file (dashes used in emulator)
            src_hostname = src_emu_hostname.replace("-", "_")

            decoys: list[Host] = self.network.decoys_reserve
            decoy_hostnames: list[str] = [decoy.name for decoy in decoys]

            # Check if the src_host is a decoy itself
            src_host = None
            if src_hostname in decoy_hostnames:
                src_host = [decoy for decoy in decoys if decoy.name == src_hostname][0]
            else:
                src_host = self.network.get_node_from_name(src_hostname)

            # Get the destination (target) decoy Host object
            target_decoy_host = [
                decoy for decoy in decoys if decoy.ip_address == dst_ip
            ]

            # Ensure decoy Host exist in network topology.
            # If no decoy Host, create a temporary Host to create the Alert.
            if len(target_decoy_host) > 0 and isinstance(target_decoy_host[0], Host):
                target_decoy_host = target_decoy_host[0]  # assume first option
            else:
                # print(
                #    f"Cannot find decoy Host with {dst_ip} in network topology, creating Host for alert."
                # )
                dst_host_type = HostType()
                dst_host_type.decoy = True
                temp_dst_host = Host(
                    name="dst_host",
                    subnet=next(iter(self.network.subnets.values())),
                    host_type=dst_host_type,
                )
                temp_dst_host.set_ip_from_str(dst_ip)
                target_decoy_host = temp_dst_host

            # Create the Alert
            new_alert = Alert()
            new_alert.add_src_host(cast(Host, src_host))
            new_alert.add_dst_host(target_decoy_host)
            alerts.append(new_alert)

            # Keep track of previous dst_ip for comparison
            prev_dst_ip = dst_ip

        return alerts

    def obs(self, perfect_alerts: Iterable[Alert] = []) -> Iterable[Alert]:
        """
        Creates an array of alerts using information from the SIEM's query response.
        """
        response = self.submit_obs_query()
        hits = self.parse_query_response(response)
        alerts = self.create_alerts(hits)
        return alerts

    def _get_target_ip(self, command: str) -> str:
        ip_candidates = re.findall(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", command)

        if len(ip_candidates) == 0:
            return ""

        return ip_candidates[0]  # assuming only 1 target

    def _find_decoy_hits(self, hits: List[Dict[Any, Any]]) -> List[Dict[Any, Any]]:
        decoy_host_names = self.network_config["decoys"]
        decoy_ips = []

        for host_name in decoy_host_names:
            ip = self._get_ip_address(host_name)
            if ip:
                decoy_ips.append(ip)

        decoy_hits = [hit for hit in hits if hit["target_ip"] in decoy_ips]
        return decoy_hits

    def _get_ip_address(self, host_name: str) -> str:
        host_user = self.emu_config["firewheel"]["host"]["username"]
        host_pwd = self.emu_config["firewheel"]["host"]["password"]

        command_arr = [
            f"sshpass -p {host_pwd}",
            f"firewheel ssh {host_user}@{host_name}",
            "ip -4 -brief address show | grep ens2 | awk '{print $3}'",
        ]
        cmd = " ".join(command_arr)

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True,
            text=True,
            check=True,
        )

        if result.returncode != 0:
            print(result.stderr)
            return ""
        elif result.stdout == "":
            return ""
        else:
            ip = result.stdout.split("/")[0]
            return ip
