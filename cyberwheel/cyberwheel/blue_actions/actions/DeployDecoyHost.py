import json

from typing import Dict, List

from cyberwheel.blue_actions.blue_action import (
    SubnetAction,
    generate_id,
    BlueActionReturn,
)
from cyberwheel.network.network_base import Network
from cyberwheel.network.host import HostType, HostTypes
from cyberwheel.network.subnet import Subnet


def get_host_types() -> List[Dict[str, any]]:
    with open("resources/metadata/host_definitions.json", "rb") as f:
        host_defs = json.load(f)
    return host_defs["host_types"]


class DeployDecoyHost(SubnetAction):
    """
    This class represents the action for deploying a decoy Host in the network.
    """
    def __init__(self, network: Network, configs: Dict[str, any], **kwargs) -> None:
        super().__init__(network, configs)
        self.define_configs()
        self.define_services()

    def execute(self, subnet: Subnet, **kwargs) ->  BlueActionReturn:
        """
        This executes the action to deploy a decoy host.

        When ran, this function will add a decoy Host to the
        network with a UUID name.
        """
        seed = kwargs.get("seed", None)
        emulation = kwargs.get("emulation", False)
        name = generate_id(seed=seed)
        if "server" in self.type.lower():
            host_type = HostType(
                name="Server", type=HostTypes.DECOY_SERVER, services=self.services, decoy=True, cve_list=self.cves
            )
        else:
            host_type = HostType(
                name="Workstation",
                type=HostTypes.DECOY_USER,
                services=self.services,
                decoy=True,
                cve_list=self.cves,
            )
        
        decoy_limit_exceeded = len(self.network.decoys) > self.max_decoys
        if decoy_limit_exceeded:
            return BlueActionReturn("decoy_limit_exceeded", False, 0)

        if emulation:
            self.host = self.network.enable_decoy_host(name, subnet, host_type)
            self.decoy_list.append(name)
            return BlueActionReturn(name, True, 1)
        else:
            self.host = self.network.create_decoy_host(name, subnet, host_type)
            #print("deployed new host")
            #print(f"Deploying Decoy: {name}")
            return BlueActionReturn(name, True, 0, target=subnet.name)
        

#class IsolateDecoyHost(SubnetAction):
#    def __init__(self, network: Network, configs: Dict[str, any], **kwargs) -> None:
#        super().__init__(network, configs)
#        self.define_configs()
#        self.define_services()
#        self.isolate_data = kwargs.get("isolate_data", [])
#
#    def execute(self, subnet: Subnet, **kwargs) -> BlueActionReturn:
#        name = generate_id()
#        host_type = HostType(
#            name=name, type=HostTypes.DECOYservices=self.services, decoy=True, cve_list=self.cves
#        )
#        self.host = self.network.create_decoy_host(name, subnet, host_type)
#        return BlueActionReturn(
#            name, self.isolate_data.append_decoy(self.host, subnet), 1
#        )

