from typing import Dict

from cyberwheel.blue_actions.blue_action import StandaloneAction, generate_id, BlueActionReturn
from cyberwheel.network.network_base import Network

class Nothing(StandaloneAction):
    """
    This class represents the blue action for doing nothing.
    """
    def __init__(self, network: Network, configs: Dict[str, any], **kwargs) -> None:
        super().__init__(network, configs)

    def execute(self, action=None, **kwargs) ->  BlueActionReturn:
        #print("does nothing")
        return BlueActionReturn(generate_id(), True)                                                                                                                                                                                                                   
