from cyberwheel.blue_agents.blue_agent import BlueAgent, BlueAgentResult


class InactiveBlueAgent(BlueAgent):
    """
    This agent does nothing.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__()

    def act(self, action=None) -> BlueAgentResult:
        return BlueAgentResult("nothing", -1, True, 0)

    def get_reward_map(self):
        return {"nothing": (0, 0)}

    def reset(self, network=None):
        return
