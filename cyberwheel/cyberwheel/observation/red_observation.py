import numpy as np

from typing import Iterable

from cyberwheel.observation.observation import Observation
from cyberwheel.network.network_base import Network, HostTypes
from cyberwheel.observation.observation_attributes import ObservationAttribute, StandaloneQuadrantAttribute

class RedObservation(Observation):

    def __init__(self, args, network: Network):
        self.args = args
        self.network = network
        
        self.define_attributes()
        self.max_size = (self.args.max_num_hosts * len(self.attributes["host"])) + len(self.attributes["standalone"])
        self._init_vars()
    
    def _init_vars(self):
        self.size : int = 0
        self.obs_vec : list[int] = [-1] * self.max_size
        self.known_subnets = []
        self.obs : dict[str, dict] = {}
        self.standalone_obs: dict[str, int] = {}
        self.obs_index: dict[str, int] = {}

        for attr in self.attributes["standalone"]:
            self.standalone_obs[attr.name] = 0

    def define_attributes(self):
        """
        Any attribute in the red observation must first be defined here.
        """
        self.attributes: dict[str, list[ObservationAttribute]] = {}

        self.attributes["host"] = [ # Define all host-level observations
            ObservationAttribute("type", 'int'),
            ObservationAttribute("sweeped", 'bool'),
            ObservationAttribute("scanned", 'bool'),
            ObservationAttribute("discovered", 'bool'),
            ObservationAttribute("on_host", 'bool'),
            ObservationAttribute("escalated", 'bool'),
            ObservationAttribute("impacted", 'bool'),
        ]
        self.attributes["subnet"] = [] # Define all subnet-level observations
        self.attributes["standalone"] = [ # Define all standalone observations
            StandaloneQuadrantAttribute(),
        ]
        
    def add_host(self, host: str, **kwargs):
        ip_as_key = kwargs.get("ip_as_key", False)
        self.obs[host] = {attr.name: int(kwargs.get(attr.name, 0)) for attr in self.attributes["host"]}
        self.obs_index[host] = self.size
        num_host_attributes = len(self.attributes["host"])
        self.obs_vec[self.size:(self.size + num_host_attributes)] = list(self.obs[host].values())
        self.size += num_host_attributes
        if ip_as_key:
            self.known_subnets.append(self.network.get_node_from_ip(host))
        else:
            self.known_subnets.append(self.network.hosts[host].subnet.name)

    def update_host(self, host: str, **kwargs):
        for attr in self.attributes["host"]:
            self.obs[host][attr.name] = attr.get_obs_value(kwargs, default=self.obs[host][attr.name])
        host_index = self.obs_index[host]
        self.obs_vec[host_index:host_index+len(self.attributes["host"])] = list(self.obs[host].values())
    
    def update_obs(self, **kwargs):
        i = len(self.attributes["standalone"])
        for attr in self.attributes["standalone"]:
            obs_value = attr.get_obs_value(kwargs)
            self.obs_vec[self.max_size - i] = obs_value
            self.standalone_obs[attr] = obs_value
            i -= 1
             
    def reset(self, network: Network, entry_host: str, ip_as_key=False) -> Iterable:
        self.network = network
        #self.define_attributes()
        self._init_vars()
        self.add_host(entry_host, ip_as_key=ip_as_key, on_host=True)
        return np.array(self.obs_vec)