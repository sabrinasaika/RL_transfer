from cyberwheel.network.network_base import Network
from cyberwheel.reward.reward_base import Reward
from cyberwheel.utils.hybrid_set_list import HybridSetList

from cyberwheel.red_agents.red_agent_base import RedAgentResult
from cyberwheel.blue_agents.blue_agent import BlueAgentResult

from cyberwheel.reward import red_reward_functions
from cyberwheel.reward import blue_reward_functions


class RLReward(Reward):
    def __init__(
        self,
        args,
        red_agent,
        blue_agent,
        valid_targets: list[str] | str,
        network: Network,
    ) -> None:
        super().__init__(red_agent.get_reward_map(), blue_agent.get_reward_map())
        self.args = args
        self.blue_recurring_reward = []
        self.red_recurring_reward = []

        self.valid_targets = valid_targets
        self.network = network
        self.blue_agent = blue_agent
        self.red_agent = red_agent
        self.blue_reward_function = getattr(blue_reward_functions, self.args.blue_reward_function)
        self.red_reward_function = getattr(red_reward_functions, self.args.red_reward_function)

        self.current_step = 0

    def calculate_reward(self, blue_agent_result: BlueAgentResult, red_agent_result: RedAgentResult) -> int | float:
        """
        TODO: Add function header
        """
        # Define which hosts in the network the red agent will get rewarded/penalized for attacking
        valid_targets = self.get_valid_targets()

        # Calculate red and blue rewards
        r, r_recurring = self.red_reward_function(self, blue_agent_result=blue_agent_result, red_agent_result=red_agent_result, valid_targets=valid_targets)
        
        b, b_recurring = self.blue_reward_function(self, blue_agent_result=blue_agent_result, red_agent_result=red_agent_result, valid_target=valid_targets)

        # Handle recurring rewards
        self.blue_recurring_reward += [b_recurring] if b_recurring != 0 else []
        self.red_recurring_reward += [r_recurring] if r_recurring != 0 else []

        if blue_agent_result.recurring == -1 and len(self.blue_recurring_reward) > 0:
            self.blue_recurring_reward.pop(0)

        # Step forward
        self.current_step += 1

        return b + sum(self.blue_recurring_reward), r + sum(self.red_recurring_reward)

    
    def get_valid_targets(self) -> HybridSetList:
        if self.valid_targets == "servers":
            valid_targets = self.network.server_hosts.data_set
        elif self.valid_targets == "users":
            valid_targets = self.network.user_hosts.data_set
        elif self.valid_targets == "all":
            valid_targets = self.network.hosts.keys()
        elif self.valid_targets == "leader":
            valid_targets = {self.red_agent.leader_host.name}
        elif type(self.valid_targets) is list:
            valid_targets = set(self.valid_targets)
        elif type(self.valid_targets) is str:
            valid_targets = set([self.valid_targets])
        else:
            valid_targets = self.network.hosts.keys()
        return valid_targets | set(self.network.decoys)

    def reset(self) -> None:
        self.blue_recurring_reward = []
        self.red_recurring_reward = []
        self.current_step = 0
