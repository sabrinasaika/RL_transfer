import os
from typing import Any, Dict

import numpy as np
from stable_baselines3 import PPO

from adapters.unified_env import UnifiedSecEnv
from config.env_builders import make_cbs_env


def main():
    # Force single-input obs for CW policy compatibility and disable adapters that change actions
    os.environ.setdefault("CBS_MULTI_INPUT", "0")
    os.environ.setdefault("CBS_REPAIR", "1")
    os.environ.setdefault("BOOST_STEPS", "0")
    os.environ.setdefault("EVAL_MASK_POLICY", "0")

    model_path = os.environ.get("CW_MODEL_PATH", "artifacts/policies/cw_ppo_minimal")
    max_steps = int(os.environ.get("ROLLOUT_STEPS", "1000"))
    deterministic = os.environ.get("ROLLOUT_DET", "1") == "1"
    num_episodes = int(os.environ.get("NUM_EPISODES", "3"))

    # Build CBS env
    env = UnifiedSecEnv("cbs", cbs_factory=make_cbs_env)

    # Load CW-trained policy (MlpPolicy on 8D vector)
    model = PPO.load(model_path)

    total_returns = []
    
    for episode in range(1, num_episodes + 1):
        obs, info = env.reset()
        if isinstance(obs, dict):
            obs_vec = obs.get("obs") if "obs" in obs else obs
        else:
            obs_vec = obs

        ret = 0.0
        print(f"Starting rollout with CW policy on CBS (episode {episode}/{num_episodes})\n")
        
        for t in range(1, max_steps + 1):
            action, _ = model.predict(obs_vec, deterministic=deterministic)
            action_int = int(action)
            action_name = env.act_t.unified_actions[action_int]

            # Derive backend action for logging (no side effects)
            try:
                backend_action = env._ensure_valid_backend_action(env._xlate_action(action_int))  # type: ignore[attr-defined]
            except Exception:
                backend_action = {"unknown": None}

            obs, r, terminated, truncated, step_info = env.step(action_int)
            if isinstance(obs, dict):
                obs_vec = obs.get("obs") if "obs" in obs else obs
            else:
                obs_vec = obs

            ret += float(r)

            # Extract simple CBS telemetry
            disc = 0
            owned = 0
            try:
                raw = getattr(env, "_last_raw_obs", {}) or {}
                disc = int(raw.get("discovered_node_count", 0) or 0)
                priv = raw.get("nodes_privilegelevel")
                if priv is not None:
                    arr = np.asarray(priv)
                    owned = int((arr >= 1).sum())
            except Exception:
                pass

            # Print step data
            print(f"episode={episode} step={t} obs={np.asarray(obs_vec).round(3).tolist()} action=({action_int}, {action_name}) backend={backend_action} r={float(r):.3f} ret={ret:.3f} discovered={disc} owned={owned}")

            if bool(terminated) or bool(truncated):
                break

        print(f"\nEpisode {episode} done in {t} steps. Return={ret:.3f}")
        total_returns.append(ret)
    
    # Print summary
    print(f"\n=== SUMMARY ===")
    print(f"Completed {num_episodes} episodes")
    print(f"Episode returns: {[f'{r:.3f}' for r in total_returns]}")
    print(f"Average return: {np.mean(total_returns):.3f}")
    print(f"Std return: {np.std(total_returns):.3f}")


def rollout_epsilon_greedy():
    # Keep original defaults; only add epsilon-greedy selection on unified actions
    os.environ.setdefault("CBS_MULTI_INPUT", "0")
    os.environ.setdefault("CBS_REPAIR", "0")
    os.environ.setdefault("BOOST_STEPS", "0")
    os.environ.setdefault("EVAL_MASK_POLICY", "0")

    model_path = os.environ.get("CW_MODEL_PATH", "artifacts/policies/cw_ppo_minimal")
    max_steps = int(os.environ.get("ROLLOUT_STEPS", "1000"))
    deterministic = os.environ.get("ROLLOUT_DET", "1") == "1"
    epsilon = float(os.environ.get("EPSILON", "0.9"))
    num_episodes = int(os.environ.get("NUM_EPISODES", "3"))

    env = UnifiedSecEnv("cbs", cbs_factory=make_cbs_env)
    model = PPO.load(model_path)

    total_returns = []
    
    for episode in range(1, num_episodes + 1):
        obs, info = env.reset()
        if isinstance(obs, dict):
            obs_vec = obs.get("obs") if "obs" in obs else obs
        else:
            obs_vec = obs

        ret = 0.0
        print(f"Starting epsilon-greedy rollout with CW policy on CBS (episode {episode}/{num_episodes})\n")
        
        for t in range(1, max_steps + 1):
            # Try to get unified valid action mask (7-D). Fallback to no mask.
            try:
                mask = env._compute_unified_mask()  # type: ignore[attr-defined]
            except Exception:
                mask = None

            if np.random.rand() < epsilon:
                if mask is not None and np.sum(mask) > 0:
                    valid_idxs = np.where(np.asarray(mask) > 0.0)[0]
                    action_int = int(np.random.choice(valid_idxs))
                else:
                    action_int = int(env.action_space.sample())
            else:
                action, _ = model.predict(obs_vec, deterministic=deterministic)
                action_int = int(action)

            action_name = env.act_t.unified_actions[action_int]

            # Derive backend action for logging (no side effects)
            try:
                backend_action = env._ensure_valid_backend_action(env._xlate_action(action_int))  # type: ignore[attr-defined]
            except Exception:
                backend_action = {"unknown": None}

            obs, r, terminated, truncated, step_info = env.step(action_int)
            if isinstance(obs, dict):
                obs_vec = obs.get("obs") if "obs" in obs else obs
            else:
                obs_vec = obs

            ret += float(r)

            # Extract simple CBS telemetry
            disc = 0
            owned = 0
            try:
                raw = getattr(env, "_last_raw_obs", {}) or {}
                disc = int(raw.get("discovered_node_count", 0) or 0)
                priv = raw.get("nodes_privilegelevel")
                if priv is not None:
                    arr = np.asarray(priv)
                    owned = int((arr >= 1).sum())
            except Exception:
                pass

            print(f"episode={episode} step={t} obs={np.asarray(obs_vec).round(3).tolist()} action=({action_int}, {action_name}) backend={backend_action} r={float(r):.3f} ret={ret:.3f} discovered={disc} owned={owned}")

            if bool(terminated) or bool(truncated):
                break

        print(f"\nEpisode {episode} done in {t} steps. Return={ret:.3f}")
        total_returns.append(ret)
    
    # Print summary
    print(f"\n=== SUMMARY ===")
    print(f"Completed {num_episodes} episodes")
    print(f"Episode returns: {[f'{r:.3f}' for r in total_returns]}")
    print(f"Average return: {np.mean(total_returns):.3f}")
    print(f"Std return: {np.std(total_returns):.3f}")


if __name__ == "__main__":
    # Preserve original behavior; opt-in to epsilon with ROLLOUT_MODE=eps
    mode = os.environ.get("ROLLOUT_MODE", "policy").lower()
    if mode == "eps":
        rollout_epsilon_greedy()
    else:
        main()


