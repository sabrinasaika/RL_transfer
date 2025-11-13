import os
import time
import numpy as np

from stable_baselines3.ppo import PPO
from adapters.unified_env import UnifiedSecEnv
from config.env_builders import make_cbs_env


def evaluate(model_path: str, episodes: int = 5, render: bool = False) -> float:
    if not os.path.exists(model_path + ".zip"):
        raise FileNotFoundError(f"Model not found: {model_path}.zip")

    env = UnifiedSecEnv("cbs", cbs_factory=make_cbs_env)
    model = PPO.load(model_path)

    returns = []
    max_steps = int(os.environ.get("EVAL_MAX_STEPS", "0"))
    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        ret = 0.0
        steps = 0
        verbose = os.environ.get("EVAL_VERBOSE", "0") == "1"
        log_every = 50
        try:
            log_every = int(os.environ.get("EVAL_LOG_EVERY", "50"))
            if log_every < 1:
                log_every = 1
        except Exception:
            log_every = 50
        while not (done or truncated):
            det = os.environ.get("EVAL_DETERMINISTIC", "0") == "1"
            action, _ = model.predict(obs, deterministic=det)
            obs, r, done, truncated, info = env.step(int(action))
            ret += float(r)
            steps += 1
            if max_steps > 0 and steps >= max_steps:
                break
            if render:
                time.sleep(0.01)
            if verbose and (steps % log_every == 0 or done or truncated):
                print(f"  step={steps} r={r:.3f} ret={ret:.3f} term={done} trunc={truncated}", flush=True)
        returns.append(ret)
        print(f"Episode {ep+1}/{episodes}: return={ret:.3f}, steps={steps}")

    avg = float(np.mean(returns)) if returns else 0.0
    print(f"Average return over {episodes} episodes: {avg:.3f}")
    return avg


if __name__ == "__main__":
    model_path = os.environ.get(
        "CW_MODEL_PATH", "artifacts/policies/cw_ppo_minimal"
    )
    # Fallback to CBS model if CW model is not present
    if not os.path.exists(model_path + ".zip"):
        alt = "artifacts/policies/cbs_ppo_minimal"
        if os.path.exists(alt + ".zip"):
            model_path = alt
    episodes = int(os.environ.get("EVAL_EPISODES", "5"))
    evaluate(model_path, episodes=episodes)


