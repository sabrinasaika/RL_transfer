import numpy as np

from typing import Dict, Iterable
from importlib.resources import files

from cyberwheel.detectors.alert import Alert
from cyberwheel.network.host import Host
from cyberwheel.observation.observation import Observation
from cyberwheel.observation.observation_attributes import ObservationAttribute
from cyberwheel.network.network_base import Network
from cyberwheel.detectors.handler import DetectorHandler

import time


class BlueObservation(Observation):
    def __init__(self, args, network, detector: DetectorHandler) -> None: #mapping: Dict[Host, int], detector_config: str) -> None:
        self.args = args
        self.network = network
        self.detector = detector

        self.define_attributes()
        self.max_size = (2 * self.args.max_num_hosts) + len(self.attributes["standalone"])
        self._init_vars()

    def define_attributes(self):
        self.attributes: dict[str, list[ObservationAttribute]] = {}
        self.attributes["standalone"] = [ # Define all standalone observations
            ObservationAttribute("num_decoys_deployed", 'int'),
        ]
    
    def _init_vars(self):
        self.mapping = {host: i for i, host in enumerate(sorted(list(self.network.hosts.keys())))}
        self.len_alerts = len(self.mapping) * 2
        self.obs_vec = np.full(self.max_size, -1)
        for i in range(self.len_alerts):
            self.obs_vec[i] = 0
    
    def create_obs_vector(self, alerts: Iterable[Alert], **kwargs) -> Iterable:
        barrier = self.len_alerts // 2

        # Refresh the non-history portion of the obs_vec
        for i in range(barrier):
            self.obs_vec[i] = 0
        
        # Trip alerts in observation space
        for alert in alerts:
            alerted_host = alert.src_host
            if not alerted_host or alerted_host.name not in self.mapping:
                continue
            index = self.mapping[alerted_host.name]
            self.obs_vec[index] = 1
            self.obs_vec[index + barrier] = 1

        # Add standalone observation attributes to the end of the obs space
        i = len(self.attributes["standalone"])
        for attr in self.attributes["standalone"]:
            obs_value = attr.get_obs_value(kwargs, default=-1)
            self.obs_vec[self.max_size - i] = obs_value
            i -= 1
        return self.obs_vec

    def reset(self, network) -> Iterable:
        self.network = network
        self._init_vars()
        self.detector.reset()
        return self.obs_vec