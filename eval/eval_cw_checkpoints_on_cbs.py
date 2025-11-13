import os
import re
import glob
import json
import sys
from typing import List, Tuple, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
from stable_baselines3.ppo import PPO
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from adapters.unified_env import UnifiedSecEnv
from config.env_builders import make_cbs_env

# Add cyberwheel to path for loading native checkpoints
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'cyberwheel'))

def _prepare_obs(obs, expected_size: int) -> np.ndarray:
    if isinstance(obs, dict):
        obs_vec = obs.get("obs", obs)
    else:
        obs_vec = obs
    obs_array = np.asarray(obs_vec, dtype=np.float32).flatten()
    if np.any(np.isnan(obs_array)) or np.any(np.isinf(obs_array)):
        obs_array = np.nan_to_num(obs_array, nan=0.0, posinf=0.0, neginf=0.0)
    if len(obs_array) < expected_size:
        obs_array = np.pad(obs_array, (0, expected_size - len(obs_array)), mode='constant', constant_values=0.0)
    elif len(obs_array) > expected_size:
        obs_array = obs_array[:expected_size]
    obs_array = np.nan_to_num(obs_array, nan=0.0, posinf=0.0, neginf=0.0)
    return obs_array


def finetune_adapter_on_cbs(env: UnifiedSecEnv, feature_extractor: torch.nn.Module, adapter_head: torch.nn.Module,
                            obs_space_shape: tuple, device: torch.device) -> None:
    """Quick on-policy finetune of the 7-action adapter head using REINFORCE with unified action masking.
    If FINETUNE_FEATURES=1, also update the feature_extractor weights."""
    try:
        episodes = int(os.environ.get("FINETUNE_EPISODES", "3") or 3)
        max_steps = int(os.environ.get("FINETUNE_MAX_STEPS", "200") or 200)
        lr = float(os.environ.get("ADAPTER_LR", "0.003") or 0.003)
        ent_coef = float(os.environ.get("ADAPTER_ENTROPY", "0.01") or 0.01)
        finetune_features = os.environ.get("FINETUNE_FEATURES", "0") == "1"
    except Exception:
        episodes, max_steps, lr, ent_coef, finetune_features = 3, 200, 0.003, 0.01, False

    if finetune_features:
        feature_extractor.train()
        for p in feature_extractor.parameters():
            p.requires_grad = True
        params = list(adapter_head.parameters()) + list(feature_extractor.parameters())
    else:
        feature_extractor.eval()
        for p in feature_extractor.parameters():
            p.requires_grad = False
        params = list(adapter_head.parameters())
    adapter_head.train()
    optimizer = torch.optim.Adam(params, lr=lr)

    cbs_action_space_size = len(env.act_t.unified_actions)
    expected_size = int(np.prod(obs_space_shape))

    for _ in range(episodes):
        obs, _ = env.reset()
        step = 0
        while step < max_steps:
            step += 1
            obs_np = _prepare_obs(obs, expected_size)
            obs_t = torch.from_numpy(obs_np).float().to(device)
            with torch.no_grad():
                feats = feature_extractor(obs_t.unsqueeze(0))
            logits = adapter_head(feats)
            # Compute 7-D unified mask and apply
            try:
                mask_np = env._compute_unified_mask().astype(np.float32)
            except Exception:
                mask_np = np.ones((cbs_action_space_size,), dtype=np.float32)
            mask_t = torch.from_numpy(mask_np).to(device=device, dtype=torch.bool).unsqueeze(0)
            masked_logits = logits.masked_fill(~mask_t, float('-inf'))
            dist = torch.distributions.Categorical(logits=masked_logits)
            action = dist.sample()
            logprob = dist.log_prob(action)
            act_int = int(action.item())
            obs, r, done, truncated, _info = env.step(act_int)
            reward_t = torch.tensor([float(r)], device=device)
            loss = -(logprob * reward_t)
            # entropy bonus
            loss = loss - ent_coef * dist.entropy()
            optimizer.zero_grad(set_to_none=True)
            loss.mean().backward()
            torch.nn.utils.clip_grad_norm_(adapter_head.parameters(), 1.0)
            optimizer.step()
            if done or truncated:
                break



def load_cyberwheel_policy(ckpt_path: str, action_space_size: int, obs_space_shape: tuple) -> torch.nn.Module:
    """Load Cyberwheel native .pt checkpoint."""
    from cyberwheel.utils import RLPolicy
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = RLPolicy(action_space_shape=action_space_size, obs_space_shape=obs_space_shape).to(device)
    state_dict = torch.load(ckpt_path, map_location=device)
    policy.load_state_dict(state_dict)
    policy.eval()
    return policy


def infer_cyberwheel_config(ckpt_path: str) -> Tuple[int, tuple]:
    """Infer action space size and obs space shape from checkpoint."""
    state_dict = torch.load(ckpt_path, map_location='cpu')
    # RLPolicy actor has: input(0), hidden(2), output(4) layers
    if 'actor.4.weight' in state_dict and 'actor.0.weight' in state_dict:
        action_space_size = state_dict['actor.4.weight'].shape[0]  # Output dimension
        obs_size = state_dict['actor.0.weight'].shape[1]  # Input dimension
        return action_space_size, (obs_size,)
    return 20, (100,)  # Fallback defaults


def evaluate_on_cbs(model_path: str, episodes: int = 10, use_cyberwheel_native: bool = False, 
                    action_space_size: Optional[int] = None, obs_space_shape: Optional[tuple] = None) -> dict:
    """
    Load a Cyberwheel-trained policy and evaluate its jumpstart performance on CBS via transfer project.
    
    For Cyberwheel native checkpoints, uses Cyberwheel's RLPolicy evaluation interface
    through the transfer project's UnifiedSecEnv.
    """
    os.environ.setdefault("CBS_MULTI_INPUT", "0")
    os.environ.setdefault("CBS_REPAIR", "1")
    os.environ.setdefault("BOOST_STEPS", "0")
    os.environ.setdefault("EVAL_MASK_POLICY", "0")

    env = UnifiedSecEnv("cbs", cbs_factory=make_cbs_env)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if use_cyberwheel_native:
        # Load Cyberwheel native .pt checkpoint using Cyberwheel's evaluation approach
        if action_space_size is None or obs_space_shape is None:
            action_space_size, obs_space_shape = infer_cyberwheel_config(model_path)
        policy = load_cyberwheel_policy(model_path, action_space_size, obs_space_shape)
        cbs_action_space_size = len(env.act_t.unified_actions)
        # Don't use action mask for CBS evaluation - pass None to allow all actions
        action_mask = None
        # Optional: use a compact 7-action adapter head over CW features (actor without last layer)
        use_adapter = os.environ.get("USE_CBS_ADAPTER_HEAD", "0") == "1"
        adapter_head = None
        feature_extractor = None
        if use_adapter:
            # actor[:-1] yields [Linear, ReLU, Linear, ReLU] -> 64-dim features
            feature_extractor = torch.nn.Sequential(*list(policy.actor.children())[:-1]).to(device)
            adapter_head = torch.nn.Linear(64, cbs_action_space_size).to(device)
            adapter_head.eval()
            # Optional quick finetune of adapter head on CBS
            if os.environ.get("FINETUNE_ADAPTER", "0") == "1":
                finetune_adapter_on_cbs(env, feature_extractor, adapter_head, obs_space_shape, device)
                adapter_head.eval()
    else:
        # Load SB3 .zip checkpoint
        model = PPO.load(model_path)
        policy = None
        action_mask = None

    returns: List[float] = []
    steps_to_goal: List[int] = []
    successes: int = 0
    for episode_idx in range(episodes):
        obs, info = env.reset()
        
        if use_cyberwheel_native:
            # Prepare observation for Cyberwheel policy (matching Cyberwheel's evaluation format)
            expected_size = int(np.prod(obs_space_shape))
            obs_array = _prepare_obs(obs, expected_size)
            obs_tensor = torch.from_numpy(obs_array).float().to(device)
        
        done = False
        truncated = False
        ep_ret = 0.0
        step = 0
        max_steps = int(os.environ.get("CBS_MAX_STEPS", "1000"))
        
        while not (done or truncated) and step < max_steps:
            if use_cyberwheel_native:
                # Use Cyberwheel's policy interface (matching rl_evaluator.py pattern)
                with torch.no_grad():
                    if adapter_head is not None and feature_extractor is not None:
                        # Extract 64-dim features and map to 7 CBS actions directly (mask invalid)
                        feats = feature_extractor(obs_tensor.unsqueeze(0))
                        logits = adapter_head(feats)
                        try:
                            mask_np = env._compute_unified_mask().astype(np.float32)
                        except Exception:
                            mask_np = np.ones((cbs_action_space_size,), dtype=np.float32)
                        mask_t = torch.from_numpy(mask_np).to(device=device, dtype=torch.bool).unsqueeze(0)
                        logits = logits.masked_fill(~mask_t, float('-inf'))
                        # Optional stochastic action selection to avoid deterministic flat behavior
                        do_sample = os.environ.get("EVAL_SAMPLE", "0") == "1"
                        try:
                            tau = float(os.environ.get("EVAL_TAU", "1.0") or 1.0)
                        except Exception:
                            tau = 1.0
                        try:
                            eps = float(os.environ.get("EVAL_EPSILON", "0.0") or 0.0)
                        except Exception:
                            eps = 0.0
                        if do_sample:
                            logits_t = logits / max(tau, 1e-6)
                            dist = torch.distributions.Categorical(logits=logits_t)
                            act = dist.sample()
                            action_int = int(act.item())
                        else:
                            action_int = int(torch.argmax(logits, dim=-1).item())
                        # Epsilon-greedy override among valid actions
                        if eps > 0.0 and np.random.rand() < eps:
                            valid_idx = np.where(mask_np > 0.5)[0]
                            if valid_idx.size > 0:
                                action_int = int(np.random.choice(valid_idx))
                    else:
                        # Get action from policy (optionally with action mask)
                        # For CBS, we'll map Cyberwheel's large action space to CBS's unified actions
                        action, _, _, _ = policy.get_action_and_value(obs_tensor.unsqueeze(0), action_mask=action_mask)
                        action_int = int(action.item())
                        # Map Cyberwheel action space to CBS action space (CBS has 7 unified actions)
                        # Clamp first, then map using modulo
                        action_int = max(0, min(action_int, action_space_size - 1))
                        action_int = action_int % cbs_action_space_size
            else:
                # SB3 model prediction
                deterministic = not (os.environ.get("EVAL_SAMPLE", "0") == "1")
                action, _ = model.predict(obs, deterministic=deterministic)
                action_int = int(action)
                if os.environ.get("EVAL_SAMPLE", "0") == "1":
                    try:
                        eps = float(os.environ.get("EVAL_EPSILON", "0.0") or 0.0)
                    except Exception:
                        eps = 0.0
                    if eps > 0.0 and np.random.rand() < eps:
                        action_int = int(np.random.randint(0, env.action_space.n))
            
            obs, r, done, truncated, info = env.step(action_int)
            ep_ret += float(r)
            step += 1
            
            if use_cyberwheel_native:
                # Update observation tensor for next step
                if isinstance(obs, dict):
                    obs_vec = obs.get("obs", obs)
                else:
                    obs_vec = obs
                obs_array = np.asarray(obs_vec, dtype=np.float32).flatten()
                
                # Check for NaN/inf and replace with zeros
                if np.any(np.isnan(obs_array)) or np.any(np.isinf(obs_array)):
                    obs_array = np.nan_to_num(obs_array, nan=0.0, posinf=0.0, neginf=0.0)
                
                if len(obs_array) < expected_size:
                    obs_array = np.pad(obs_array, (0, expected_size - len(obs_array)), mode='constant', constant_values=0.0)
                elif len(obs_array) > expected_size:
                    obs_array = obs_array[:expected_size]
                
                # Ensure no NaN/inf after processing
                obs_array = np.nan_to_num(obs_array, nan=0.0, posinf=0.0, neginf=0.0)
                
                obs_tensor = torch.from_numpy(obs_array).float().to(device)
        
        returns.append(ep_ret)
        # Track success/steps
        try:
            if bool(done) and float(ep_ret) > 0.0:
                successes += 1
                steps_to_goal.append(step)
            else:
                steps_to_goal.append(step)
        except Exception:
            steps_to_goal.append(step)
    
    avg_return = float(np.mean(returns)) if returns else 0.0
    avg_steps = float(np.mean(steps_to_goal)) if steps_to_goal else 0.0
    success_rate = float(successes) / float(episodes) if episodes > 0 else 0.0
    return {
        "avg_return": avg_return,
        "avg_steps_to_goal": avg_steps,
        "success_rate": success_rate,
        "episode_returns": returns,
        "episode_steps_to_goal": steps_to_goal,
    }


def evaluate_random_baseline_on_cbs(episodes: int = 10) -> dict:
    """Evaluate a simple random masked policy on CBS for comparison."""
    os.environ.setdefault("CBS_MULTI_INPUT", "0")
    env = UnifiedSecEnv("cbs", cbs_factory=make_cbs_env)
    cbs_action_space_size = len(env.act_t.unified_actions)
    returns: List[float] = []
    steps_to_goal: List[int] = []
    successes: int = 0
    max_steps = int(os.environ.get("CBS_MAX_STEPS", "1000"))
    for _ in range(episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        ep_ret = 0.0
        step = 0
        while not (done or truncated) and step < max_steps:
            step += 1
            try:
                mask_np = env._compute_unified_mask().astype(np.float32)
            except Exception:
                mask_np = np.ones((cbs_action_space_size,), dtype=np.float32)
            valid_indices = np.where(mask_np > 0.5)[0]
            if valid_indices.size == 0:
                action_int = int(np.random.randint(0, cbs_action_space_size))
            else:
                action_int = int(np.random.choice(valid_indices))
            obs, r, done, truncated, info = env.step(action_int)
            ep_ret += float(r)
        returns.append(ep_ret)
        # Success proxy: positive terminal return
        if bool(done) and float(ep_ret) > 0.0:
            successes += 1
            steps_to_goal.append(step)
        else:
            steps_to_goal.append(step)
    return {
        "avg_return": float(np.mean(returns)) if returns else 0.0,
        "avg_steps_to_goal": float(np.mean(steps_to_goal)) if steps_to_goal else 0.0,
        "success_rate": float(successes) / float(episodes) if episodes > 0 else 0.0,
        "episode_returns": returns,
        "episode_steps_to_goal": steps_to_goal,
    }


def find_checkpoints(ckpt_dir: str) -> List[Tuple[int, str, bool]]:
    """Return list of (steps, path, is_cyberwheel_native) for all checkpoints sorted by steps.
    
    Supports both:
    - SB3 .zip files: pattern cw_ppo_ckpt_{steps}_steps.zip
    - Cyberwheel .pt files: pattern red_{steps}.pt
    """
    items: List[Tuple[int, str, bool]] = []
    
    # Find SB3 .zip checkpoints
    zip_paths = glob.glob(os.path.join(ckpt_dir, "*.zip"))
    for p in zip_paths:
        m = re.search(r"_(\d+)_steps\.zip$", os.path.basename(p))
        if m:
            steps = int(m.group(1))
            items.append((steps, p, False))
    
    # Find Cyberwheel native .pt checkpoints
    pt_paths = glob.glob(os.path.join(ckpt_dir, "red_*.pt"))
    for p in pt_paths:
        basename = os.path.basename(p)
        if basename == "red_agent.pt":
            continue  # Skip the latest checkpoint, prefer numbered ones
        m = re.search(r"red_(\d+)\.pt$", basename)
        if m:
            steps = int(m.group(1))
            items.append((steps, p, True))
    
    items.sort(key=lambda x: x[0])
    return items


def latest_tb_run_dir(tb_root: str) -> str:
    """Pick the latest TB run dir under runs/cw-ppo-training."""
    if not os.path.isdir(tb_root):
        return ""
    subdirs = [os.path.join(tb_root, d) for d in os.listdir(tb_root) if os.path.isdir(os.path.join(tb_root, d))]
    if not subdirs:
        return ""
    return max(subdirs, key=lambda d: os.path.getmtime(d))


def plot_side_by_side(tb_dir: str,
                      jump_steps: List[int],
                      jump_scores: List[float],
                      out_file: str,
                      cw_limit_steps: Optional[List[int]] = None,
                      random_scores: Optional[List[float]] = None,
                      episode_returns: Optional[List[List[float]]] = None,
                      random_episode_returns: Optional[List[List[float]]] = None) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    # Left: training curves from TB
    if tb_dir and os.path.isdir(tb_dir):
        try:
            ea = EventAccumulator(tb_dir)
            ea.Reload()
            # Training reward (Cyberwheel uses charts/red_episodic_return)
            if 'charts/red_episodic_return' in ea.Tags().get('scalars', []):
                s = ea.Scalars('charts/red_episodic_return')
                s_steps = [e.step for e in s]
                s_values = [e.value for e in s]
                if cw_limit_steps:
                    # Plot only the values at the requested checkpoint steps (nearest match)
                    cw_x = []
                    cw_y = []
                    for st in cw_limit_steps:
                        # find nearest step index
                        if len(s_steps) == 0:
                            continue
                        idx = int(np.argmin(np.abs(np.asarray(s_steps) - st)))
                        cw_x.append(s_steps[idx])
                        cw_y.append(s_values[idx])
                    if cw_x and cw_y:
                        axes[0].plot(cw_x, cw_y, marker='o', linewidth=2, color='blue', markersize=6, label='CW checkpoints')
                else:
                    axes[0].plot(s_steps, s_values, label='CW Training Return', linewidth=2, color='blue')
            elif 'rollout/ep_rew_mean' in ea.Tags().get('scalars', []):
                s = ea.Scalars('rollout/ep_rew_mean')
                axes[0].plot([e.step for e in s], [e.value for e in s], label='Train Reward', linewidth=2)
            # Eval reward
            if 'eval/mean_reward' in ea.Tags().get('scalars', []):
                s = ea.Scalars('eval/mean_reward')
                axes[0].plot([e.step for e in s], [e.value for e in s], label='Eval Reward', linewidth=2)
            axes[0].set_title('Cyberwheel Training Performance', fontsize=14, fontweight='bold')
            axes[0].set_xlabel('Timesteps')
            axes[0].set_ylabel('Reward')
            axes[0].grid(True, alpha=0.3)
            axes[0].legend()
            axes[0].axhline(y=0, color='black', linestyle='--', alpha=0.4)
        except Exception as e:
            axes[0].text(0.5, 0.5, f"TB load failed: {e}", ha='center', va='center')
            axes[0].set_axis_off()
    else:
        axes[0].text(0.5, 0.5, "No TB data found", ha='center', va='center')
        axes[0].set_axis_off()

    # Right: transfer performance on CBS
    if jump_steps and jump_scores:
        axes[1].plot(jump_steps, jump_scores, marker='o', linewidth=2, color='red', markersize=6, label='CBS Avg Return')
        # CBS episode scatter removed for clarity
        if random_scores and len(random_scores) == len(jump_steps) and random_episode_returns is not None:
            if any(rs is not None for rs in random_scores):
                axes[1].plot(
                    jump_steps,
                    [rs if rs is not None else float('nan') for rs in random_scores],
                    marker='x',
                    linewidth=1.5,
                    color='gray',
                    linestyle='--',
                    label='Random Avg Return',
                )
                for idx, step in enumerate(jump_steps):
                    eps = random_episode_returns[idx]
                    if eps:
                        axes[1].scatter([step] * len(eps), eps, color='gray', alpha=0.6, s=30, marker='s', label='Random Episodes' if idx == 0 else "")
        axes[1].set_title('Transfer Performance on CyberBattleSim', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Training Steps (Checkpoint)', fontsize=12)
        axes[1].set_ylabel('Avg Return on CBS', fontsize=12)
        axes[1].grid(True, alpha=0.3)
        axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.4)
        axes[1].legend(loc='upper left')
    else:
        axes[1].text(0.5, 0.5, "No checkpoints found", ha='center', va='center')
        axes[1].set_axis_off()

    plt.tight_layout()
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    print(f"Saved side-by-side plot: {out_file}")


def main():
    # Support both SB3 and Cyberwheel native checkpoints
    ckpt_dir = os.environ.get("CW_CKPT_DIR", "/home/ssaika/rl-transfer-sec-clean/cyberwheel/cyberwheel/data/models/CWRun_Long")
    episodes = int(os.environ.get("CBS_EVAL_EPISODES", "5"))
    tb_dir = os.environ.get("CW_TB_DIR", "/home/ssaika/rl-transfer-sec-clean/cyberwheel/cyberwheel/data/runs/CWRun_Long")
    out_dir = os.environ.get("OUT_DIR", "/home/ssaika/rl-transfer-sec-clean/artifacts/plots/transfer_eval")
    os.makedirs(out_dir, exist_ok=True)

    checkpoints = find_checkpoints(ckpt_dir)
    if not checkpoints:
        print(f"No checkpoint models found in: {ckpt_dir}")
    else:
        print(f"Found {len(checkpoints)} checkpoints")

    # Optionally limit to N evenly spaced checkpoints (default 5)
    try:
        num_ckpts = int(os.environ.get("NUM_CHECKPOINTS", "5") or 5)
    except Exception:
        num_ckpts = 5
    if checkpoints and num_ckpts > 0 and len(checkpoints) > num_ckpts:
        idxs = np.linspace(0, len(checkpoints) - 1, num=num_ckpts, dtype=int)
        checkpoints = [checkpoints[i] for i in idxs]
        print(f"Selected {len(checkpoints)} checkpoints (evenly spaced): {[s for s,_,_ in checkpoints]}")

    steps_list: List[int] = []
    scores_list: List[float] = []
    steps_to_goal_list: List[float] = []
    success_rates: List[float] = []
    random_scores_list: List[Optional[float]] = []
    random_steps_to_goal_list: List[Optional[float]] = []
    random_success_rates: List[Optional[float]] = []
    episode_returns_list: List[List[float]] = []
    random_episode_returns_list: List[List[float]] = []
    include_random = os.environ.get("EVAL_INCLUDE_RANDOM", "1") == "1"
    for steps, path, is_cw_native in checkpoints:
        print(f"Evaluating checkpoint: steps={steps} path={os.path.basename(path)} (native={'CW .pt' if is_cw_native else 'SB3 .zip'})")
        try:
            if is_cw_native:
                # Infer config from checkpoint
                action_space_size, obs_space_shape = infer_cyberwheel_config(path)
                stats = evaluate_on_cbs(path, episodes=episodes, use_cyberwheel_native=True,
                                    action_space_size=action_space_size, obs_space_shape=obs_space_shape)
            else:
                stats = evaluate_on_cbs(path, episodes=episodes, use_cyberwheel_native=False)
            print(f"  -> Avg CBS return over {episodes} eps: {stats['avg_return']:.3f} | avg_steps_to_goal={stats['avg_steps_to_goal']:.1f} | success_rate={stats['success_rate']:.2f}")
            # Random baseline per checkpoint (same settings/environment)
            rnd = None
            if include_random:
                rnd = evaluate_random_baseline_on_cbs(episodes=episodes)
                print(f"     Random baseline: avg_return={rnd['avg_return']:.3f} | avg_steps_to_goal={rnd['avg_steps_to_goal']:.1f} | success_rate={rnd['success_rate']:.2f}")
            steps_list.append(steps)
            scores_list.append(float(stats["avg_return"]))
            steps_to_goal_list.append(float(stats["avg_steps_to_goal"]))
            success_rates.append(float(stats["success_rate"]))
            episode_returns_list.append(list(stats.get("episode_returns", [])))
            if include_random and rnd is not None:
                random_scores_list.append(float(rnd["avg_return"]))
                random_steps_to_goal_list.append(float(rnd["avg_steps_to_goal"]))
                random_success_rates.append(float(rnd["success_rate"]))
                random_episode_returns_list.append(list(rnd.get("episode_returns", [])))
            else:
                random_scores_list.append(None)
                random_steps_to_goal_list.append(None)
                random_success_rates.append(None)
                random_episode_returns_list.append([])
        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            continue

    if not steps_list:
        print("❌ No checkpoints evaluated successfully!")
        return
    
    # Persist results
    results_path = os.path.join(out_dir, "cw_to_cbs_transfer_results.json")
    with open(results_path, "w") as f:
        json.dump({
            "steps": steps_list, 
            "scores": scores_list,
            "avg_steps_to_goal": steps_to_goal_list,
            "success_rate": success_rates,
            "episode_returns": episode_returns_list,
            "episode_returns_random": random_episode_returns_list,
            "scores_random": random_scores_list,
            "avg_steps_to_goal_random": random_steps_to_goal_list,
            "success_rate_random": random_success_rates,
            "episodes_per_checkpoint": episodes,
            "description": "Cyberwheel checkpoints evaluated on CBS; includes steps-to-goal and success rate"
        }, f, indent=2)
    print(f"✓ Saved results: {results_path}")

    # Side-by-side plot
    if not tb_dir or not os.path.isdir(tb_dir):
        # Try to find latest if path doesn't exist
        tb_root = os.path.dirname(tb_dir) if tb_dir else "runs/cw-ppo-training"
        tb_dir = latest_tb_run_dir(tb_root)
    side_by_side_path = os.path.join(out_dir, "cw_training_and_cbs_transfer_side_by_side.png")
    plot_side_by_side(tb_dir,
                      steps_list,
                      scores_list,
                      side_by_side_path,
                      cw_limit_steps=steps_list,
                      random_scores=random_scores_list,
                      episode_returns=episode_returns_list,
                      random_episode_returns=random_episode_returns_list)


if __name__ == "__main__":
    main()


