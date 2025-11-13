from cyberwheel.red_agents.strategies.red_strategy import RedStrategy
from cyberwheel.network.host import Host
from cyberwheel.red_actions.art_techniques import SudoandSudoCaching


class ServerDowntime(RedStrategy):
    """
    The Server Downtime strategy is to find and attack all of the Servers it can find in the network.
    Once it finds a server, it will try to impact it. Once impacted, it will look for another server.
    """
    @classmethod
    def select_target(cls, agent_obj) -> Host:
        current_host_type = agent_obj.history.hosts[agent_obj.current_host.name].type

        """
        It should continue impacting the current host if: it is Unknown or if it is a Server that has not been impacted yet. Otherwise it should move to another host.
        It should prioritize attacking other Servers that are unimpacted in its view. Then it should prioritize Unknown hosts in its view.
        If there are no unimpacted Servers or Unknown hosts in its view, it has succeeded. Maybe give this a very large cost to signify failure on the blue agent side.
        """

        target_host = None
        pool = None
        if (
            current_host_type == "Unknown"
            or agent_obj.current_host.name in agent_obj.unimpacted_servers
        ):
            return agent_obj.current_host
        elif len(agent_obj.unimpacted_servers) > 0:
            pool = agent_obj.unimpacted_servers
        elif len(agent_obj.unknowns) > 0:
            pool = agent_obj.unknowns
        else:
            return agent_obj.current_host
        
        while not target_host and len(pool) > 0:
            random_host_name = pool.get_random()
            if random_host_name in agent_obj.network.hosts:
                target_host = agent_obj.network.hosts[random_host_name]
            else:
                target_host = None
                #print(f"{random_host_name} not in network")
                pool.remove(random_host_name)
        target_host = target_host if target_host else agent_obj.current_host
        return target_host
