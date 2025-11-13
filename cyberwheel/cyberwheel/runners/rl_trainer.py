import torch
import math
import random
import gymnasium as gym
from gymnasium import spaces
import time
import os
import importlib
import numpy as np
import yaml

from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter
from torch import optim, nn
from importlib.resources import files
from statistics import mean, median

from cyberwheel.utils import RLPolicy, get_service_map
from cyberwheel.utils.set_seed import set_seed
from cyberwheel.runners.rl_handler import RLHandler
from cyberwheel.network.network_base import Network

class RLTrainer:
    def __init__(self, args):
        self.args = args
        m = importlib.import_module("cyberwheel.cyberwheel_envs")
        self.env = getattr(m, args.environment)
        self.args.deterministic = os.getenv("CYBERWHEEL_DETERMINISTIC", "False").lower() in ('true', '1', 't')
        self.seed = args.seed
        self.define_vars = True

        self.agents = {}
        self.static_agents = []
        self.rl_agents = []

    def make_env(self, rank, evaluation: bool = False, net_name: str = ""):
        """
        Utility function for multiprocessed env.

        :param env_id: the environment ID
        :param num_env: the number of environments you wish to have in subprocesses
        :param rank: index of the subprocess
        """

        def _init():
            if evaluation: # Only evaluate on one network at a time
                env = self.env(self.args, network=self.networks[net_name][rank])
            else:
                env = self.env(self.args, network=random.choice(list(self.networks.values()))[rank])

            for agent in self.agents:
                if not self.define_vars:
                    continue
                if not self.args.agent_config[agent]["rl"] and (agent == 'red' or agent == 'blue'):
                    continue
                    #self.agents[agent] = {"max_action_space_size": 0, "max_obs_space_size": 0, "max_attrs": 0}
                    #self.agents[agent]["obs"] = spaces.Space()
                elif agent == 'red':
                    self.agents[agent] = {"max_action_space_size": env.red_agent.action_space.max_size, "max_obs_space_size": env.red_agent.observation.max_size, "max_attrs": env.max_red_attr_value}
                    self.rl_agents.append(agent)
                elif agent == 'blue':
                    self.agents[agent] = {"max_action_space_size": env.blue_agent.action_space.max_size, "max_obs_space_size": env.blue_agent.observation.max_size, "max_attrs": env.max_blue_attr_value}
                    self.rl_agents.append(agent)
                else:
                    raise Exception("Agent Not Recognized!")
                    #print("Agent Not Recognized!")
                self.agents[agent]["obs"] = spaces.Box(
                    low  = np.full(self.agents[agent]["max_obs_space_size"], -1, dtype=np.int32),
                    high = np.full(self.agents[agent]["max_obs_space_size"], self.agents[agent]["max_attrs"], dtype=np.int32),
                    dtype=np.int32
                )
                    
            self.define_vars = False
            env.reset()
            #env = gym.wrappers.RecordEpisodeStatistics(env)  # This tracks the rewards of the environment that it wraps. Used for logging
            return env

        return _init

    def mask_actions(self, new_action_mask, action_mask):
        new_mask = torch.tensor(
            new_action_mask,
            dtype=torch.bool,
            device=action_mask.device,
        )
        return new_mask
    
    def evaluate(self, agents, env):
        """Evaluate 'agent'"""
        # We evaluate on CPU because learning is already happening on GPUs.
        # You can evaluate small architectures on CPU, but if you increase the neural network size,
        # you may need to do fewer evaluations at a time on GPU.
        eval_device = torch.device("cpu")
        self.episode_decoy_attacks = []
        self.episode_decoys_deployed = []

        agent_info = {agent: {
                "episode_rewards": [0] * self.args.eval_episodes,
                "action_masks": torch.zeros(self.handler.agents[agent]["max_action_space_size"], dtype=torch.bool).to(eval_device)
            } for agent in self.rl_agents
        }

        # Standard evaluation loop to estimate mean episodic return
        for episode in range(self.args.eval_episodes):
            num_decoy_attacks = 0
            obs, _ = env.reset()
            for step in range(self.args.num_steps):
                action = None
                actions = {}
                action_masks = env.action_mask

                for agent in self.rl_agents:
                    agent_obs = torch.Tensor(obs[agent]).to(eval_device)
                    tmp_mask = action_masks[agent]
                    agent_info[agent]["action_masks"] = self.mask_actions(tmp_mask, agent_info[agent]["action_masks"])
                    action, _, _, _ = agents[agent].get_action_and_value(agent_obs, action_mask=agent_info[agent]["action_masks"])
                    actions[agent] = action

                obs, rew, done, _, info = env.step(actions)

                for agent in self.agents:
                    agent_info[agent]["episode_rewards"][episode] += info[f"{agent}_reward"] if f"{agent}_reward" in info else 0

                if "decoy_attacked" in info and info["decoy_attacked"]:
                    num_decoy_attacks += 1

            self.episode_decoy_attacks.append(num_decoy_attacks)
            self.episode_decoys_deployed.append(len(env.network.decoys))

        episodic_return = {agent: float(sum(agent_info[agent]["episode_rewards"])) / self.args.eval_episodes for agent in self.agents}
        return episodic_return
    
    def run_evals(self, models, globalstep):
        """Evaluate 'model' on tasks listed in 'eval_queue' in a separate process"""
        eval_device = torch.device("cpu")
        loaded_models = {agent: torch.load(models[agent], map_location=eval_device) for agent in self.agents}
        eval_agents = {}

        results = {}
        for network_name in self.networks:
            env = self.make_env(0, evaluation=True, net_name = network_name)()
            for agent in self.agents:
                eval_agent = None
                # Load the agent

                eval_agent = RLPolicy(action_space_shape=self.handler.agents[agent]["max_action_space_size"], obs_space_shape=self.handler.agents[agent]["shape"]).to(eval_device)
                eval_agent.load_state_dict(loaded_models[agent])
                eval_agent.eval()
                eval_agents[agent] = eval_agent

            # Evaluate the agent
            result = self.evaluate(eval_agents, env)
            # Store evaluation parameters and results
            results[network_name] = result
            
        return results
    
    def wandb_setup(self):
        # Initialize Weights and Biases tracking
        import wandb

        self.run = wandb.init(
            project=self.args.wandb_project_name,  # Can be whatever you want
            entity=self.args.wandb_entity,
            sync_tensorboard=True,  # Data logged to the tensorboard SummaryWriter will be sent to W&B
            config=vars(self.args),  # Saves args as the run's configuration
            name=self.args.experiment_name,  # Unique run name
            monitor_gym=False,  # Does not attempt to render any episodes
            save_code=False,
        )
        
        self.run.define_metric("episodic_runtime", summary="mean")

    def configure_training(self):
        self.writer = SummaryWriter(
            files("cyberwheel.data.runs").joinpath(self.args.experiment_name)
        )  # Logs data to tensorboard and W&B
        self.writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s"
            % ("\n".join([f"|{key}|{value}|" for key, value in vars(self.args).items()])),
        )
        # Seeding
        if self.args.deterministic:
            set_seed(self.seed)
            torch.backends.cudnn.deterministic = True
        else:
            torch.backends.cudnn.deterministic = False
            #torch.set_num_threads(1)

        # Environment setup

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
            self.networks[network_name] = [deepcopy(network) for i in range(self.args.num_envs)]

            print("Mapping attack validity to hosts...", end=" ")
            self.args.service_mapping[network_name] = get_service_map(network)
            print("done")


        self.args.agent_config = {}

        for agent_type in self.args.agents:
            self.args.agent_config[agent_type] = {}
            agent_yaml = self.args.agents[agent_type]
            agent_config = files(f"cyberwheel.data.configs.{agent_type}_agent").joinpath(agent_yaml)
            with open(agent_config, "r") as yaml_file:
                self.args.agent_config[agent_type] = yaml.safe_load(yaml_file)
            if self.args.agent_config[agent_type]["rl"]:
                self.agents[agent_type] = None
            else:
                self.static_agents.append(agent_type)
        
        print("Defining environment(s) and beginning training:", end="\n\n")

        env_funcs = [self.make_env(i) for i in range(self.args.num_envs)]

        self.envs = (
            gym.vector.AsyncVectorEnv(env_funcs)
            if self.args.async_env
            else gym.vector.SyncVectorEnv(env_funcs)
        )

        assert isinstance(
            self.envs.single_action_space, gym.spaces.Dict
        ), "only discrete action space is supported"

        # Create agent and optimizer

        self.handler = RLHandler(self.envs, self.args, self.agents, static_agents=self.static_agents)

        self.handler.define_multiagent_variables()

    def train(self, update):
        # Tracking runtimes and processing times
        train_start_time = time.time()
        train_start_process_time = time.process_time()
        episode_start = time.time_ns()
        episode_process_start = time.process_time_ns()

        # Resetting environments 
        self.handler.reset()

        # Run an episode in each environment. This loop collects experience which is later used for optimization.
        for step in range(0, self.args.num_steps):
            # Set determinism if applicable
            if self.args.deterministic:
                set_seed(self.seed)
            self.seed += self.args.num_envs

            # Update action masking for blue agent and red agent
            self.handler.update_action_masks(step)

            # Get action and value estimates, step through environment and update obs
            self.handler.step_multiagent(step)

        # Tracking runtimes and processing times      
        end_time = time.time_ns()
        end_process_time = time.process_time_ns()
        episode_time = (end_time - episode_start) / (10**9)
        episode_process_time = (end_process_time - episode_process_start) / (10**9)

        # Logging stuff
        self.handler.log_stuff(self.writer, episode_time, episode_process_time)
        
        # Calculate advantages used to optimize the policy and returns which are compared to values to optimize the critic.
        self.handler.compute_gae()

        # Flatten the batch
        self.handler.flatten_batch()

        # Optimizing the policy and value network 
        b_inds = np.arange(self.args.batch_size)

        # Iterate over multiple epochs which each update the policy using all of the batch data
        for epoch in range(self.args.update_epochs):
            np.random.shuffle(b_inds)

            # For each epoch, split the batch into minibatches for smaller updates
            for start in range(0, self.args.batch_size, self.args.minibatch_size):
                end = start + self.args.minibatch_size
                mb_inds = b_inds[start:end]

                self.handler.update_policy(mb_inds)
                self.handler.calculate_loss(mb_inds)
                self.handler.backpropagate(update)

            if self.args.target_kl is not None:
                if self.handler.approx_kl > self.args.target_kl:
                    break

        self.handler.calculate_explained_variance()

        # Infrequently save the model and evaluate the agent
        if (update - 1) % self.args.save_frequency == 0:
            start_eval = time.time()
            start_process_eval = time.process_time()

            # Save the model
            
            agent_paths = self.handler.save_models()

            # Run evaluation
            print("Evaluating Agent...")

            eval_return = self.run_evals(agent_paths, self.handler.global_step) # TODO: globalstep or agent?

            for network_name in eval_return:
                self.writer.add_scalar("charts/eval_time", int(time.time() - start_eval), self.handler.global_step)
                self.writer.add_scalar("charts/eval_process_time", int(time.process_time() - start_process_eval), self.handler.global_step)
                mean_decoys_attacked = mean(self.episode_decoy_attacks)
                median_decoys_attacked = median(self.episode_decoy_attacks)
                mean_decoys_deployed = mean(self.episode_decoys_deployed)
                median_decoys_deployed = median(self.episode_decoys_deployed)
                self.writer.add_scalar(
                    f"evaluation/{network_name}_mean_decoys_attacked",
                    mean_decoys_attacked,
                    self.handler.global_step,
                )
                self.writer.add_scalar(
                    f"evaluation/{network_name}_median_decoys_attacked",
                    median_decoys_attacked,
                    self.handler.global_step,
                )
                self.writer.add_scalar(
                    f"evaluation/{network_name}_mean_decoys_deployed",
                    mean_decoys_deployed,
                    self.handler.global_step
                )
                self.writer.add_scalar(
                    f"evaluation/{network_name}_median_decoys_deployed",
                    median_decoys_deployed,
                    self.handler.global_step
                )
                for agent in self.agents:
                    self.writer.add_scalar(
                        f"evaluation/{network_name}_blue_episodic_return",
                        eval_return[network_name][agent],
                        self.handler.global_step,
                    )

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        #print(f"Actor: {self.optimizer.param_groups[0]['lr']}")
        #print(f"Critic: {self.optimizer.param_groups[1]['lr']}")
        sps = int((self.args.num_steps * self.args.num_envs) / (time.time() - train_start_time))
        print("SPS:", sps)
        process_sps = int((self.args.num_steps * self.args.num_envs) / (time.process_time() - train_start_process_time))
        self.writer.add_scalar("charts/SPS", sps, self.handler.global_step)
        self.writer.add_scalar("charts/process_SPS", process_sps, self.handler.global_step)
        
        self.handler.log_training_metrics(self.writer)


    def close(self) -> None:
        self.envs.close()
        self.writer.close()