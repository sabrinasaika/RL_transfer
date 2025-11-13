import gymnasium as gym
import numpy as np
import importlib

from typing import Iterable, Any 
from gymnasium import spaces

from cyberwheel.cyberwheel_envs.cyberwheel import Cyberwheel
from cyberwheel.network.network_base import Network
from cyberwheel import red_agents, blue_agents
from cyberwheel.utils import YAMLConfig, HybridSetList
from cyberwheel.utils.set_seed import set_seed
from cyberwheel.reward import RLReward

import pandas as pd
import random

import time


class CyberwheelRL(gym.Env, Cyberwheel):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        args: YAMLConfig,
        network: Network = None,
        evaluation: bool = False,
        networks : dict = {},
    ):
        """
        The CyberwheelRL class is used to define the Cyberwheel environment. It allows you to use a YAML
        file to configure the actions, rewards, and logic of the blue agent. Given various configurations, it
        will initiate the environment with the red agent, blue agent, reward functions, and network state.
        Important member variables:

        * `args`: required
            - YAMLConfig instance defining the environment state.

        * `network`: optional
            - The Network object to use throughout the environment. This prevents longer start-up times when training with multiple environments.
            - If not passed, it will build the network with the config file passed.
            - Default: None
        """
        super().__init__(args, network=network)

        if len(networks) == 0:
            networks = {network.name: network}
        self.networks = networks
        self.evaluation = evaluation

        self.reward_calculator = RLReward(
            args,
            red_agent=self.red_agent, 
            blue_agent=self.blue_agent,
            valid_targets=self.args.valid_targets,
            network=self.network
        )
    
    def initialize_agents(self) -> None:
        max_net = self.args.network_size_compatibility
        self.args.max_num_hosts = 100 if max_net == 'small' else 1000 if max_net == 'medium' else 10000 # if max_net == 'large'
        self.args.max_num_subnets = 10 if max_net == 'small' else 100 if max_net == 'medium' else 1000 #if max_net == 'large'

        self.blue_agent = getattr(blue_agents, self.args.agent_config["blue"]["class"])(network=self.network, args=self.args)
        self.red_agent = getattr(red_agents, self.args.agent_config["red"]["class"])(network=self.network, args=self.args)

        self.blue_max_action_space_size = self.args.max_num_subnets * self.blue_agent.action_space.num_actions if self.args.agent_config["blue"]["rl"] else None #self.blue_agent.action_space._action_space_size if self.args.agent_config["blue"]["rl"] else None
        self.red_max_action_space_size = self.args.max_num_hosts * self.red_agent.action_space.num_actions * 2 if self.args.agent_config["red"]["rl"] else None

        self.max_blue_attr_value = self.args.max_decoys + 2 if self.args.agent_config["blue"]["rl"] else None # Max obs attribute is limited to when num_decoys_deployed exceeds max_decoys allowed
        self.max_red_attr_value = 4 if self.args.agent_config["red"]["rl"] else None # Max obs attribute is limited to the 'quadrant' attribute, which goes up to 4.


        obs_dict = {}
        act_dict = {}
        if self.args.agent_config["blue"]["rl"]:
            obs_dict["blue"] = spaces.Box(
                low  = np.full(self.blue_agent.observation.max_size, -1, dtype=np.int32),
                high = np.full(self.blue_agent.observation.max_size, self.max_blue_attr_value, dtype=np.int32),
                dtype=np.int32
            )
            act_dict["blue"] = self.blue_agent.create_action_space(self.blue_max_action_space_size)

        if self.args.agent_config["red"]["rl"]:
            obs_dict["red"] = spaces.Box(
                low  = np.full(self.red_agent.observation.max_size, -1, dtype=np.int32),
                high = np.full(self.red_agent.observation.max_size,  self.max_red_attr_value, dtype=np.int32),
                dtype=np.int32
            )
            act_dict["red"] = self.red_agent.action_space.create_action_space(self.red_max_action_space_size)

        self.observation_space = spaces.Dict(obs_dict)
        self.action_space = spaces.Dict(act_dict)

        self.red_reward_sign = -1
        self.blue_reward_sign = 1

    def step(self, action: dict[str, int]) -> tuple[Iterable, int | float, bool, bool, dict[str, Any]]:
        """
        Steps through environment.
        1. Blue agent runs action
        2. Red agent runs action
        3. Calculate reward based on red/blue actions and network state
        4. Get obs from Red or Blue Observation
        5. Return obs and related metadata
        """
        blue_agent_result = self.blue_agent.act(action["blue"]) if "blue" in action and action["blue"] != None else self.blue_agent.act()
        red_agent_result = self.red_agent.act(action["red"]) if "red" in action and action["red"] != None else self.red_agent.act()

        blue_obs_vec = self.blue_agent.get_observation_space(red_agent_result) if self.args.agent_config["blue"]["rl"] else None
        red_obs_vec = self.red_agent.get_observation_space() if self.args.agent_config["red"]["rl"] else None
        
        blue_reward, red_reward = self.reward_calculator.calculate_reward(
            blue_agent_result=blue_agent_result,
            red_agent_result=red_agent_result
        ) # TODO: Double check that the signs are correct

        # End episode either at horizon or on key red successes (prevents drowning successes in later failures)
        terminal_actions = {"ARTPrivilegeEscalation", "ARTImpact"}
        red_success_terminal = (
            red_agent_result.success
            and hasattr(red_agent_result.action, "get_name")
            and red_agent_result.action.get_name() in terminal_actions
        )
        done = self.current_step == self.max_steps - 1 or red_success_terminal        

        self.current_step += 1
        info = {
            "red_reward": red_reward,
            "blue_reward": blue_reward,
            }
        
        if self.evaluation:
            tgt_decoy = red_agent_result.target_host.decoy if red_agent_result.target_host != "invalid" else False
            decoy_attacked = red_agent_result.success and (tgt_decoy or red_agent_result.src_host.decoy)
            info = {
                "red_action": red_agent_result.action.get_name(),
                "red_action_src": red_agent_result.src_host.name,
                "red_action_dst": red_agent_result.target_host.name if red_agent_result.target_host != "invalid" else "invalid",
                "red_action_success": red_agent_result.success,
                "blue_action": blue_agent_result.name,
                "blue_action_id": blue_agent_result.id,
                "blue_action_target": blue_agent_result.target,
                "blue_action_success": blue_agent_result.success,
                "blue_action_src": blue_agent_result.target,
                "blue_action_dst": blue_agent_result.target,
                "killchain": self.red_agent.killchain,
                "network": self.network,
                "history": self.red_agent.history,
                "commands": [], #red_agent_result.action_results.metadata.get("commands", []),
                "decoy_attacked": decoy_attacked,
                "red_reward": red_reward,
                "blue_reward": blue_reward,
            }
        obs = {"blue": blue_obs_vec, "red": red_obs_vec}
        reward = blue_reward + red_reward
        return obs, reward, done, False, info

    def reset(self, seed=None, options=None) -> tuple[Iterable, dict]:
        if seed is not None:
            set_seed(seed)
        self.current_step = 0

        self.network.reset()
        self.network = self.network if self.evaluation else random.choice(list(self.networks.values()))

        self.red_agent.reset(self.network, self.args.service_mapping[self.network.name])
        self.blue_agent.reset(self.network)
        self.reward_calculator.reset()
        obs = {
            "blue": self.blue_agent.observation.obs_vec if self.args.agent_config["blue"]["rl"] else None, 
            "red": self.red_agent.observation.obs_vec if self.args.agent_config["red"]["rl"] else None,
        }
        info = {}
        return obs, info

        
    def close(self) -> None:
        pass

    @property
    def blue_action_space_size(self):
        return self.blue_agent.action_space._action_space_size

    @property
    def red_action_space_size(self):
        return self.red_agent.action_space._action_space_size
    
    @property
    def action_mask(self):
        mask = {}
        if self.args.agent_config["blue"]["rl"]:
            mask["blue"] = self.blue_agent.action_space.get_action_mask()
        if self.args.agent_config["red"]["rl"]:
            mask["red"] = self.red_agent.action_space.get_action_mask(self.red_agent.current_host.name)   
        return mask

    @property
    def blue_action_mask(self):
        return self.blue_agent.action_space.get_action_mask()