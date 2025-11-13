from cyberwheel.blue_agents.blue_agent import BlueAgent, BlueAgentResult
from cyberwheel.network.network_base import Network
from cyberwheel.network.host import HostType, HostTypes
from cyberwheel.blue_actions.blue_action import generate_id, BlueActionReturn
from importlib.resources import files
from cyberwheel.network.service import Service

import random
import yaml

class RandomBlueAgent(BlueAgent):
    """
    This agent deploys n number of decoys randomly in the network.
    """

    def __init__(self, network: Network, args) -> None:
        self.args = args
        self.network = network
        super().__init__()
        self.deploy_steps = [random.randint(0, round(args.num_steps / 4)) for _ in range(args.max_decoys)]
        self.subnets = list(network.subnets.values())
        self.current_step = 0
        host_defs = files(f"cyberwheel.data.configs.host_definitions").joinpath('host_defs_services.yaml')
        services = files(f"cyberwheel.data.configs.services").joinpath('windows_exploitable_services.yaml')
        decoy_info = files(f"cyberwheel.data.configs.decoy_hosts").joinpath('decoy_server_hosts.yaml')

        with open(host_defs, "r") as f:
            self.host_info = yaml.safe_load(f)
        with open(services, "r") as f:
            self.service_info = yaml.safe_load(f)
        with open(decoy_info, "r") as f:
            self.decoy_info = yaml.safe_load(f)
        
        self.type = list(self.decoy_info.values())[0]["type"]

        type_info = self.host_info["host_types"][self.type]
        self.services = set()
        self.cves = set()
        
        for s in type_info["services"]:
            service = Service.create_service_from_dict(self.service_info[s])
            self.services.add(service)
            self.cves.update(service.vulns)

    def act(self, action=None) -> BlueAgentResult:
        agent_result = BlueAgentResult("nothing", -1, True, 0)
        if self.current_step in self.deploy_steps and len(self.network.decoys) <= self.args.max_decoys:
            name = generate_id()
            target_subnet = random.choice(self.subnets)
            host_type = HostType(
                name="Server", type=HostTypes.DECOY_SERVER, services=self.services, decoy=True, cve_list=self.cves
            )
            self.network.create_decoy_host(name, target_subnet, host_type)
            agent_result = BlueAgentResult('deploy_decoy', name, True, 0, target=target_subnet.name)
            #print(self.network.decoys.keys())
            #print(self.current_step)
        self.current_step += 1
        return agent_result

    def get_reward_map(self):
        return {"nothing": (0, 0), "deploy_decoy": (0, 0)}

    def reset(self, network: Network):
        self.deploy_steps = [random.randint(0, round(self.args.num_steps / 4)) for _ in range(self.args.max_decoys)]
        self.subnets = list(network.subnets.values())
        self.current_step = 0
        return
