from cyberwheel.red_agents.strategies.red_strategy import RedStrategy
from cyberwheel.network.host import Host


class Exfiltration(RedStrategy):
    """
    The Exfiltration strategy is to find and impact a specific host in the network.
    Once it discovers it (runs Discovery on it), it will impact it.
    """
    @classmethod
    def select_target(cls, agent_obj) -> Host:
        """
        It should continue impacting the current host if: it is Unknown or if it is the Target. Otherwise it should move to another host.
        It should prioritize attacking other Servers that are unimpacted in its view. Then it should prioritize Unknown hosts in its view.
        """
        current_host_type = agent_obj.history.hosts[agent_obj.current_host.name].type
        target_host = agent_obj.current_host
        # print(agent_obj.history.hosts)
        # print(agent_obj.history.hosts[agent_obj.current_host.name].is_leader)

        # If current host is Unknown, or if it is known AND leader, keep attacking
        if (
            agent_obj.history.hosts[agent_obj.current_host.name].is_leader
            or agent_obj.history.hosts[agent_obj.current_host.name].type == "Unknown"
        ):
            target_host = agent_obj.current_host
        elif len(agent_obj.unknowns) > 0:
            random_host_name = ""
            target_host = None
            while not target_host and len(agent_obj.unknowns) > 0:
                random_host_name = agent_obj.unknowns.get_random()
                if random_host_name in agent_obj.network.hosts:
                    target_host = agent_obj.network.hosts[random_host_name]
                else:
                    target_host = None
                    #print(f"{random_host_name} not in network")
                    agent_obj.unknowns.remove(random_host_name)
        target_host = agent_obj.current_host if not target_host else target_host
        
        return target_host
