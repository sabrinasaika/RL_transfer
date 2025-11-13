
import gymnasium as gym
import time
import importlib
import pandas as pd
import torch
import wandb
import os
import random
import yaml

from importlib.resources import files
from tqdm import tqdm

from cyberwheel.network.network_base import Network
from cyberwheel.utils import RLPolicy, get_service_map
from cyberwheel.runners.visualizer import Visualizer
from cyberwheel.runners.rl_trainer import RLTrainer
from cyberwheel.utils.set_seed import set_seed


class RLEvaluator(RLTrainer):
    def __init__(self, args):
        super().__init__(args)
        if self.args.download_model:
            self.api = wandb.Api()
            self.run = self.api.run(
                f"{self.args.wandb_entity}/{self.args.wandb_project_name}/runs/{self.args.run}"
            )

    def configure_evaluation(self):
        if self.args.deterministic:
            set_seed(self.seed)
            torch.backends.cudnn.deterministic = True
        else:
            set_seed(random.randint(0, 999999999))
            torch.backends.cudnn.deterministic = False

        self.device = torch.device("cpu")
        print(f"Using device {self.device}")

        # Load networks from yaml here
        network_configs = []
        if isinstance(self.args.network_config, str):
            network_configs.append(self.args.network_config)
        else:
            for config in self.args.network_config:
                network_configs.append(config)
        
        self.networks = {}
        self.args.service_mapping = {}
        for config in network_configs:
            network_config = files("cyberwheel.data.configs.network").joinpath(
                config
            )

            print(f"Building network: {config} ...")

            network = Network.create_network_from_yaml(network_config)
            network_name = network.name
            self.networks[network_name] = [network]

            print("Mapping attack validity to hosts...", end=" ")
            self.args.service_mapping[network_name] = get_service_map(network)
            print("done")
        
        self.args.agent_config = {}
        #print(self.networks)

        for agent_type in self.args.agents:
            self.args.agent_config[agent_type] = {}
            agent_yaml = self.args.agents[agent_type]
            agent_config = files(f"cyberwheel.data.configs.{agent_type}_agent").joinpath(agent_yaml)
            with open(agent_config, "r") as yaml_file:
                self.args.agent_config[agent_type] = yaml.safe_load(yaml_file)
            if self.args.agent_config[agent_type]["rl"]:
                self.agents[agent_type] = None

        self.env = self.make_env(0, evaluation=True, net_name=list(self.networks.keys())[0])()
        self.policy = {}

    def load_models(self):
        for agent in self.agents:
            self.policy[agent] = RLPolicy(self.agents[agent]["max_action_space_size"], self.agents[agent]["obs"].shape).to(self.device)
            agent_filename = f"{agent}_{self.args.checkpoint}.pt"

            # If download from W&B, use API to get run data.
            if self.args.download_model:
                model = self.run.file(agent_filename)
                model.download(
                    files("cyberwheel.data.models").joinpath(self.args.experiment_name), exist_ok=True
                )

            # Load model from models/ directory
            self.policy[agent].load_state_dict(
                torch.load(
                    files(f"cyberwheel.data.models.{self.args.experiment_name}").joinpath(agent_filename),
                    map_location=self.device,
                )
            )
            self.policy[agent].eval()

    def _initialize_environment(self):
        print("Resetting the environment...")

        self.episode_rewards = []
        self.total_reward = 0
        self.steps = 0
        self.obs = self.env.reset()

        print("Playing environment...")

        # Set up dirpath to store action logs CSV
        if self.args.graph_name != None:
            self.now_str = self.args.graph_name
        else:
            self.now_str = f"{self.args.experiment_name}_evaluate_{self.args.network_config.split('.')[0]}_{self.args.red_agent}_{self.args.reward_function}reward"
        self.log_file = files("cyberwheel.data.action_logs").joinpath(f"{self.now_str}.csv")

        self.actions_df = {
                "episode": [],
                "step": [],
        }
        self.action_mask = {}
        self.rewards = {}
        
        """
        for agent in self.agents[agent]:
            data[agent]["action_name"] = []
            data[agent]["action_src"] = []
            data[agent]["action_dest"] = []
            data[agent]["action_success"] = []
            data[agent]["reward"] = []
        """

        with open(self.log_file, 'w'): # Create an empty CSV for new action logs, overwrite previous
            pass
    
    def mask_actions(self, new_action_mask, action_mask):
        new_mask = torch.tensor(
            new_action_mask,
            dtype=torch.bool,
            device=action_mask.device,
        )
        return new_mask

    def evaluate(self):
        for agent in self.agents:
            self.action_mask[agent] = torch.zeros(self.agents[agent]["max_action_space_size"], dtype=torch.bool).to(self.device)
            self.rewards[agent] = [0] * self.args.num_episodes

        self.start_time = time.time()
        for episode in range(self.args.num_episodes):
            obs, _ = self.env.reset()
            for step in range(self.args.num_steps):
                action = None
                actions = {}
                action_masks = self.env.action_mask

                for agent in self.agents:
                    agent_obs = torch.Tensor(obs[agent]).to(self.device)
                    tmp_mask = action_masks[agent]
                    self.action_mask[agent] = self.mask_actions(tmp_mask, self.action_mask[agent])
                    action, _, _, _ = self.policy[agent].get_action_and_value(agent_obs, action_mask=self.action_mask[agent])
                    actions[agent] = action

                obs, rew, done, _, info = self.env.step(actions)

                actions_df = {
                    "episode": [episode],
                    "step": [step],
                    "reward": [rew],
                }
                self.total_reward += rew
                for agent in self.agents:
                    name_key = f"{agent}_action"
                    succ_key = f"{agent}_action_success"
                    src_key  = f"{agent}_action_src"
                    dst_key  = f"{agent}_action_dst"
                    rew_key  = f"{agent}_reward"
                    actions_df[f"{agent}_action_name"] = [info.get(name_key, "")]
                    actions_df[f"{agent}_action_success"] = [info.get(succ_key, False)]
                    actions_df[f"{agent}_action_src"] = [info.get(src_key, "")]
                    actions_df[f"{agent}_action_dest"] = [info.get(dst_key, "")]
                    actions_df[f"{agent}_reward"] = [info.get(rew_key, 0.0)]

                actions_df = pd.DataFrame(actions_df)
                actions_df.to_csv(self.log_file, mode='a', header = os.path.getsize(self.log_file) == 0, index=False)
                #return # TODO

"""
    def evaluate(self):
        
        for episode in tqdm(range(self.args.num_episodes)):
            obs, _ = self.env.reset()
            for step in range(self.args.num_steps):
                self.blue_obs = torch.Tensor(obs["blue"]).to(self.device)
                self.red_obs = torch.Tensor(obs["red"]).to(self.device)

                tmp_blue_mask = self.env.blue_action_mask
                tmp_red_mask = self.env.red_action_mask

                self.blue_action_mask = self.mask_actions(tmp_blue_mask, self.blue_action_mask)
                self.red_action_mask = self.mask_actions(tmp_red_mask, self.red_action_mask)

                blue_action, _, _, _ = self.blue_agent.get_action_and_value(
                    self.blue_obs, action_mask=self.blue_action_mask
                )
                red_action, _, _, _ = self.red_agent.get_action_and_value(
                    self.red_obs, action_mask=self.red_action_mask
                )

                action = {"blue": blue_action, "red": red_action}

                obs, rew, done, _, info = self.env.step(action)

                blue_reward = info["blue_reward"]
                red_reward = info["red_reward"]

                blue_action = info["blue_action"]
                red_action_type = info["red_action"]
                red_action_src = info["red_action_src"]
                red_action_dest = info["red_action_dst"]
                red_action_success = info["red_action_success"]
                blue_action_success = info["blue_action_success"]
                self.red_obs = obs["red"]
                self.blue_obs = obs["blue"]

                actions_df = pd.DataFrame(
                {
                    "episode": [episode],
                    "step": [step],
                    "red_action_success": [red_action_success],
                    "red_action_type": [red_action_type],
                    "red_action_src": [red_action_src],
                    "red_action_dest": [red_action_dest],
                    "blue_action": [blue_action],
                    "blue_success": [blue_action_success],
                    "blue_reward": [blue_reward],
                    "red_reward": [red_reward],
                    "reward": [rew],
                })
                actions_df.to_csv(self.log_file, mode='a', header = os.path.getsize(self.log_file) == 0, index=False)

                # If generating graphs for dash server view
                if self.args.visualize:
                    # visualize(net, episode, step, now_str, history, killchain)
                    pass

                self.total_reward += rew
                self.steps += 1

            self.steps = 0
            self.episode_rewards.append(self.total_reward)
            self.total_reward = 0

        # Save action metadata to CSV in action_logs/

        self.total_time = time.time() - self.start_time
        print("charts/SPS", int(2000 / self.total_time))
        #self.total_reward = sum(self.episode_rewards)
        #self.episodes = len(self.episode_rewards)
        print(f"Total Time Elapsed: {self.total_time}")
"""