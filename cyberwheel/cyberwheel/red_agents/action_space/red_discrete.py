from gymnasium import Space
from gymnasium.spaces import Discrete
from gymnasium.core import ActType

from cyberwheel.red_actions.actions import ARTKillChainPhase, Nothing

class RedDiscreteActionSpace:
    def __init__(self, actions: list[ARTKillChainPhase], entry_host: str) -> None:
        self._action_space_size: int = len(actions) + 1
        self.num_hosts = 1
        self.num_actions = len(actions)
        self.actions = actions
        self.hosts = [entry_host]
        self.host_index_map = {entry_host: 0}

    def select_action(self, action: ActType) -> tuple[ARTKillChainPhase, str]:
        try:
            action = int(action)
        except:
            raise TypeError(
                f"provided action is of type {type(action)} and is unsupported by the chosen ActionSpaceConverter"
            )
        
        if action == self.max_size - 1:
            return Nothing, "nothing"

        #action -= 1
        
        action_index = action % self.num_actions
        host_index = action // self.num_actions

        action_name = self.actions[action_index]
        host_name = self.hosts[host_index]

        return action_name, host_name
    
    def get_action_mask(self, current_host: str):
        action_mask = [False] * self.max_size

        #action_mask[:action_space_size] = True # Valid actions
        #action_mask[action_space_size:] = False # Invalid actions
        
        for h, i in self.host_index_map.items():
            if h == 'available':
                continue
            elif h == current_host:
                action_mask[i:(i+self.num_actions)] = [True] * self.num_actions
            else:
                action_mask[i:(i+self.num_actions-2)] = [True] * (self.num_actions - 2)
            #print(f"Host {h} - {action_mask[i:(i+self.num_actions)]}")
            #print(self.num_actions)

        #for i in range(len(self.hosts)):
        #    if self.hosts[i] == 'available':
        #        continue
        #
        #    host_index = i * self.num_actions
        #    if self.hosts[i] == current_host:
        #        action_mask[host_index:(host_index+self.num_actions)] = True # allows all actions that can target other hosts
        #    else:
        #        action_mask[host_index:(host_index+self.num_actions-2)] = True # allows all actions that can target other hosts
        #self.hosts.index(host_index)
        action_mask[self.max_size - 1] = True # allows nothing action
        #print(len(action_mask))

        return action_mask

    def add_host(self, host_name: str) -> None:
        if 'available' in self.hosts:
            i = self.hosts.index('available')
            self.hosts[self.hosts.index('available')] = host_name
            self.num_hosts += 1
        else:
            i = self._action_space_size
            self._action_space_size += len(self.actions)
            self.hosts += [host_name]
            self.num_hosts += 1
        self.host_index_map[host_name] = i
    
    def remove_host(self, host_name: str) -> None:
        i = self.hosts.index(host_name)
        self.hosts[i] = 'available'
        del self.host_index_map[host_name]

    def get_shape(self) -> tuple[int, ...]:
        return (self._action_space_size,)

    def create_action_space(self, max_size: int) -> Space:
        self.max_size = max_size
        return Discrete(max_size)

    def reset(self, entry_host: str) -> None:
        self._action_space_size: int = len(self.actions)
        self.num_hosts = 1
        self.hosts = [entry_host]
        self.host_index_map = {entry_host: 0}