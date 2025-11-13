from torch import nn, optim

from cyberwheel.utils import RLPolicy
from gymnasium.vector import VectorEnv, AsyncVectorEnv

from importlib.resources import files

import numpy as np
import torch
import os

class RLHandler:

    def __init__(self, envs: VectorEnv, args, agents: dict, static_agents=[]):
        self.envs = envs
        self.args = args
        # Use a GPU if available. You can choose a specific GPU with CUDA, for example by setting 'device' to "cuda:0"
        self.device = self.args.device
        print(f"Using device '{self.device}'")

        #self.agents = agents.keys()
        #print(agents)
        self.agents = {}
        self.static_agents = static_agents

        for agent in agents:
            self.agents[agent] = agents[agent]
            self.agents[agent]["shape"] = self.agents[agent]["obs"].shape
            self.agents[agent]["policy"] = RLPolicy(self.agents[agent]["max_action_space_size"], self.agents[agent]["shape"]).to(self.device)
            self.agents[agent]["optimizer"] = optim.Adam([
                { 'params': list(self.agents[agent]["policy"].actor.parameters()),  'lr': float(self.args.actor_lr),  'eps': 1e-5 },
                { 'params': list(self.agents[agent]["policy"].critic.parameters()), 'lr': float(self.args.critic_lr), 'eps': 1e-5 },
            ])
            if self.args.anneal_lr == 'cosine_restarts':
                self.agents[agent]["scheduler"] = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.agents[agent]["optimizer"], T_0=int(self.args.save_frequency), T_mult=int(self.args.restart_Tmult), eta_min=float(self.args.min_lr), last_epoch=-1) if self.args.anneal_lr == 'cosine_restarts' else None
            # TODO: Reconfigure LR


    def define_multiagent_variables(self):
        reset = self.envs.reset(seed=[i for i in range(self.args.num_envs)])[0]

        for agent in self.agents:
            agent_dict = self.agents[agent]
            self.agents[agent]["obs"] = torch.zeros((self.args.num_steps, self.args.num_envs) + agent_dict["obs"].shape).to(self.device)
            self.agents[agent]["actions"] = torch.zeros((self.args.num_steps, self.args.num_envs)).to(self.device)
            self.agents[agent]["logprobs"] = torch.zeros((self.args.num_steps, self.args.num_envs)).to(self.device)
            self.agents[agent]["values"] = torch.zeros((self.args.num_steps, self.args.num_envs)).to(self.device)
            self.agents[agent]["action_masks"] = torch.zeros((self.args.num_steps, self.args.num_envs, agent_dict["max_action_space_size"]), dtype=torch.bool).to(self.device)
            self.agents[agent]["resets"] = np.array(reset[agent]) # TODO: Need to update with determinism
            self.agents[agent]["next_obs"] = torch.Tensor(self.agents[agent]["resets"]).to(self.device)
            self.agents[agent]["rewards"] = torch.zeros((self.args.num_steps, self.args.num_envs)).to(self.device)
        self.dones = torch.zeros((self.args.num_steps, self.args.num_envs)).to(self.device)
        self.next_done = torch.zeros(self.args.num_envs).to(self.device)
        self.global_step = 0

    def mask_actions(self, new_action_mask, action_mask):
        new_mask = torch.tensor(
            new_action_mask,
            dtype=torch.bool,
            device=action_mask.device,
        )
        return new_mask
    
    def update_action_masks(self, step: int):
        masks = self.envs.call("action_mask") if self.args.async_env else [env.unwrapped.action_mask for env in self.envs.envs]
        for agent in self.agents:
            for i in range(self.args.num_envs):
                #print(agent, step, i, masks[i][agent])
                self.agents[agent]["action_masks"][step][i] = self.mask_actions(masks[i][agent], self.agents[agent]["action_masks"][step][i])


    def step_multiagent(self, step: int):
        self.global_step += self.args.num_envs
        self.dones[step] = self.next_done
        policy_action = {}

        for agent in self.agents:
            self.agents[agent]["obs"][step] = self.agents[agent]["next_obs"]
            with torch.no_grad():
                action, logprob, _, value = self.agents[agent]["policy"].get_action_and_value(self.agents[agent]["next_obs"], action_mask=self.agents[agent]["action_masks"][step])
                self.agents[agent]["values"][step] = value.flatten()
                self.agents[agent]["actions"][step] = action
                self.agents[agent]["logprobs"][step] = logprob
            action = action.cpu().numpy()

            # Execute the selected action in the environment to collect experience for training.
            policy_action[agent] = action
        #print(policy_action)

        obs, reward, done, _, info = self.envs.step(policy_action)

        for agent in self.agents:
            self.agents[agent]["next_obs"] = torch.Tensor(obs[agent]).to(self.device)
            # Get agent-specific reward from info dict
            if isinstance(info, dict) and f"{agent}_reward" in info:
                agent_reward = info[f"{agent}_reward"]
            elif isinstance(info, list) and len(info) > 0 and isinstance(info[0], dict):
                # Vectorized environment - get reward from first env's info
                agent_reward = info[0].get(f"{agent}_reward", 0.0)
            else:
                # Fallback: use shared reward or 0
                if isinstance(reward, (int, float)):
                    agent_reward = reward
                elif isinstance(reward, dict):
                    agent_reward = reward.get(agent, 0.0)
                else:
                    agent_reward = 0.0
            self.agents[agent]["rewards"][step] = torch.tensor(agent_reward).to(self.device).view(-1)

        self.next_done = torch.Tensor(done).to(self.device)
    
    def log_stuff(self, writer, episodic_runtime, episodic_processing_time):
        output_str = f"global_step={self.global_step}"
        for agent in self.agents:
            mean_rew = self.agents[agent]["rewards"].sum(axis=0).mean()
            output_str += f", {agent}_episodic_return={mean_rew}"
            writer.add_scalar(f"charts/{agent}_episodic_return", mean_rew, self.global_step)
        print(output_str)
        writer.add_scalar(f"charts/episodic_runtime", episodic_runtime, self.global_step)
        writer.add_scalar(f"charts/episodic_process_time", episodic_processing_time, self.global_step)

    def compute_gae(self):
        for agent in self.agents:
            with torch.no_grad():
                next_value = self.agents[agent]["policy"].get_value(self.agents[agent]["next_obs"]).reshape(1, -1)
                self.agents[agent]["advantages"] = torch.zeros_like(self.agents[agent]["rewards"]).to(self.device)
                lastgaelam = 0
                for t in reversed(range(self.args.num_steps)):
                    if t == self.args.num_steps - 1:
                        nextnonterminal = 1.0 - self.next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - self.dones[t + 1]
                        nextvalues = self.agents[agent]["values"][t + 1]
                    #print(t, self.agents[agent]["rewards"], self.agents[agent]["values"])
                    delta = self.agents[agent]["rewards"][t] + self.args.gamma * nextvalues * nextnonterminal - self.agents[agent]["values"][t]
                    self.agents[agent]["advantages"][t] = lastgaelam = delta + self.args.gamma * self.args.gae_lambda * nextnonterminal * lastgaelam
                self.agents[agent]["returns"] = self.agents[agent]["advantages"] + self.agents[agent]["values"]

    def flatten_batch(self):
        for agent in self.agents:
            self.agents[agent]["batched"] = {}
            self.agents[agent]["batched"]["obs"] = self.agents[agent]["obs"].reshape((-1,) + self.agents[agent]["shape"])
            self.agents[agent]["batched"]["logprobs"] = self.agents[agent]["logprobs"].reshape(-1)
            self.agents[agent]["batched"]["actions"] = self.agents[agent]["actions"].reshape(-1)
            self.agents[agent]["batched"]["advantages"] = self.agents[agent]["advantages"].reshape(-1)
            self.agents[agent]["batched"]["returns"] = self.agents[agent]["returns"].reshape(-1)
            self.agents[agent]["batched"]["values"] = self.agents[agent]["values"].reshape(-1)
            self.agents[agent]["batched"]["action_masks"] = self.agents[agent]["action_masks"].reshape(-1, self.agents[agent]["action_masks"].shape[-1])
            self.agents[agent]["clipfracs"] = []
    
    def update_policy(self, mb_inds):
        for agent in self.agents:
            _, newlogprob, self.agents[agent]["entropy"], self.agents[agent]["newvalue"] = self.agents[agent]["policy"].get_action_and_value(
                self.agents[agent]["batched"]["obs"][mb_inds],
                self.agents[agent]["batched"]["actions"].long()[mb_inds],
                action_mask=self.agents[agent]["batched"]["action_masks"][mb_inds],
            )
            logratio = newlogprob - self.agents[agent]["batched"]["logprobs"][mb_inds]
            self.agents[agent]["ratio"] = logratio.exp()

            # Calculate the difference between the old policy and the new policy to limit the size of the update using args.clip_coef.
            with torch.no_grad():
                # calculate approx_kl http://joschu.net/blog/kl-approx.html
                self.agents[agent]["old_approx_kl"] = (-logratio).mean()
                self.agents[agent]["approx_kl"] = ((self.agents[agent]["ratio"] - 1) - logratio).mean()
                self.agents[agent]["clipfracs"] += [
                    ((self.agents[agent]["ratio"] - 1.0).abs() > self.args.clip_coef).float().mean().item()
                ]

            self.agents[agent]["mb_advantages"] = self.agents[agent]["batched"]["advantages"][mb_inds]
            if self.args.norm_adv:
                self.agents[agent]["mb_advantages"] = (self.agents[agent]["mb_advantages"] - self.agents[agent]["mb_advantages"].mean()) / (self.agents[agent]["mb_advantages"].std() + 1e-8)
    
    def calculate_loss(self, mb_inds):
        for agent in self.agents:
            # Policy loss using PPO's ration clipping
            pg_loss1 = -self.agents[agent]["mb_advantages"] * self.agents[agent]["ratio"]
            pg_loss2 = -self.agents[agent]["mb_advantages"] * torch.clamp(
                self.agents[agent]["ratio"], 1 - self.args.clip_coef, 1 + self.args.clip_coef
            )
            self.agents[agent]["policy_loss"] = torch.max(pg_loss1, pg_loss2).mean()

            # Value loss
            newvalue = self.agents[agent]["newvalue"].view(-1)
            # Calculate the MSE loss between the returns and the value predictions of the critic
            # Clipping V loss is often not necessary and arguably worse in practice
            if self.args.clip_vloss:
                v_loss_unclipped = (newvalue - self.agents[agent]["batched"]["returns"][mb_inds]) ** 2
                v_clipped = self.agents[agent]["batched"]["values"][mb_inds] + torch.clamp(
                    newvalue - self.agents[agent]["batched"]["values"][mb_inds],
                    -self.args.clip_coef,
                    self.args.clip_coef,
                )
                v_loss_clipped = (v_clipped - self.agents[agent]["batched"]["returns"][mb_inds]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                self.agents[agent]["value_loss"] = 0.5 * v_loss_max.mean()
            else:
                self.agents[agent]["value_loss"] = 0.5 * ((newvalue - self.agents[agent]["batched"]["returns"][mb_inds]) ** 2).mean()

            # Add an entropy bonus to the loss
            self.agents[agent]["entropy_loss"] = self.agents[agent]["entropy"].mean()
            self.agents[agent]["loss"] = self.agents[agent]["policy_loss"] - self.args.ent_coef * self.agents[agent]["entropy_loss"] + self.agents[agent]["value_loss"] * self.args.vf_coef
    
    def backpropagate(self, update):
        for agent in self.agents:
            # Backpropagation
            self.agents[agent]["optimizer"].zero_grad()
            self.agents[agent]["loss"].backward()
            nn.utils.clip_grad_norm_(self.agents[agent]["policy"].parameters(), self.args.max_grad_norm)
            self.agents[agent]["optimizer"].step()

            if self.args.anneal_lr == 'cosine_restarts': # cosine lr annealing, resetting at each evaluation/checkpoint
                self.agents[agent]["scheduler"].step(update)
            else: # linear lr annealing
                frac = 1.0 - (update - 1.0) / self.args.num_updates
                lrnow = frac * self.args.learning_rate
                # Update both actor and critic lr for this agent's optimizer
                self.agents[agent]["optimizer"].param_groups[0]["lr"] = lrnow
                self.agents[agent]["optimizer"].param_groups[1]["lr"] = lrnow


    
    def calculate_explained_variance(self):
        for agent in self.agents:
            pred, true = self.agents[agent]["batched"]["values"].cpu().numpy(), self.agents[agent]["batched"]["returns"].cpu().numpy()
            var = np.var(true)
            self.agents[agent]["explained_variance"] = np.nan if var == 0 else 1 - np.var(true - pred) / var
    
    def save_models(self):
        run_path = files("cyberwheel.data.models").joinpath(self.args.experiment_name)
        agent_paths = {}
        if not os.path.exists(run_path):
            os.makedirs(run_path)
        for agent in self.agents:
            agent_path = run_path.joinpath(f"{agent}_agent.pt")
            globalstep_path = run_path.joinpath(f"{agent}_{self.global_step}.pt")
            torch.save(self.agents[agent]["policy"].state_dict(), agent_path)
            torch.save(self.agents[agent]["policy"].state_dict(), globalstep_path)
            if self.args.track:
                import wandb
                wandb.save(agent_path, base_path=run_path, policy="now")
                wandb.save(globalstep_path, base_path=run_path, policy="now")
            agent_paths[agent] = agent_path
        return agent_paths
    
    def log_training_metrics(self, writer):
        for agent in self.agents:
            writer.add_scalar(f"charts/{agent}_actor_lr", self.agents[agent]["optimizer"].param_groups[0]["lr"], self.global_step)
            writer.add_scalar(f"charts/{agent}_critic_lr", self.agents[agent]["optimizer"].param_groups[1]["lr"], self.global_step)
            writer.add_scalar(f"losses/{agent}_value_loss", self.agents[agent]["value_loss"].item(), self.global_step)
            writer.add_scalar(f"losses/{agent}_policy_loss", self.agents[agent]["policy_loss"].item(), self.global_step)
            writer.add_scalar(f"losses/{agent}_entropy", self.agents[agent]["entropy_loss"].item(), self.global_step)
            writer.add_scalar(f"losses/{agent}_old_approx_kl", self.agents[agent]["old_approx_kl"].item(), self.global_step)
            writer.add_scalar(f"losses/{agent}_approx_kl", self.agents[agent]["approx_kl"].item(), self.global_step)
            writer.add_scalar(f"losses/{agent}_clipfrac", np.mean(self.agents[agent]["clipfracs"]), self.global_step)
            writer.add_scalar(f"losses/{agent}_explained_variance", self.agents[agent]["explained_variance"], self.global_step)

    def reset(self):
        for agent in self.agents:
            reset = self.envs.reset()[0]
            self.agents[agent]["resets"] = np.array(reset[agent])
            self.agents[agent]["next_obs"] = torch.Tensor(self.agents[agent]["resets"]).to(self.device)