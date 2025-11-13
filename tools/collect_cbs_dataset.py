import argparse
from collections import deque
from typing import List, Dict, Callable, Any

import numpy as np
import gymnasium as gym
import CyberBattleSim.cyberbattle._env.cyberbattle_env as cbs_env


def make_cbs_env(size: int = 6, own_atleast_percent: float = 0.5, reward: float = 50.0, seed: int | None = None):
    env = gym.make(
        "CyberBattleChain-v0",
        size=size,
        attacker_goal=cbs_env.AttackerGoal(own_atleast_percent=own_atleast_percent, reward=reward),
    )
    if seed is not None:
        try:
            env.reset(seed=seed)
        except Exception:
            pass
    return env


def collect(
    shared_keys: List[str],
    y_specs: List[Dict[str, Any]],
    episodes: int = 50,
    K: int = 8,
    max_steps: int | None = None,
    seed: int = 0,
    env_size: int = 6,
) -> tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Collect a dataset from CBS where inputs are windows of shared features X and labels are CBS-only Y.

    y_specs: list of dicts with entries like
      {"name": "target_feature", "from": "obs" | "info" | "compute", "fn": Optional[Callable]}
    """
    rng = np.random.default_rng(seed)
    X_rows: List[np.ndarray] = []
    Y_cols: Dict[str, List[float]] = {ys["name"]: [] for ys in y_specs}

    env = make_cbs_env(size=env_size)

    for _ in range(episodes):
        obs, info = env.reset(seed=int(rng.integers(0, 1_000_000)))
        buf: deque[list[float]] = deque(maxlen=K)
        for _ in range(K - 1):
            buf.append([float(obs[k]) for k in shared_keys])

        done = False
        t = 0
        while not done and (max_steps is None or t < max_steps):
            buf.append([float(obs[k]) for k in shared_keys])
            X_rows.append(np.array(buf, dtype=float).reshape(-1))

            for ys in y_specs:
                source = ys.get("from", "obs")
                if source == "obs":
                    Y_cols[ys["name"]].append(float(obs[ys["name"]]))
                elif source == "info":
                    Y_cols[ys["name"]].append(float((info or {}).get(ys["name"], 0.0)))
                elif source == "compute":
                    fn: Callable = ys["fn"]
                    Y_cols[ys["name"]].append(float(fn(obs, info, env)))
                else:
                    raise ValueError(f"unknown y source: {source}")

            action = env.action_space.sample()
            step_out = env.step(action)
            if len(step_out) == 5:
                obs, _, term, trunc, info = step_out
                done = bool(term) or bool(trunc)
            else:
                obs, _, done, info = step_out
                done = bool(done)
            t += 1

    env.close()
    X = np.asarray(X_rows)
    Y = {k: np.asarray(v) for k, v in Y_cols.items()}
    return X, Y


def main():
    parser = argparse.ArgumentParser(description="Collect CBS dataset of X (shared) and Y (CBS-only)")
    parser.add_argument("--out", type=str, default="cbs_dataset.npz")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--history", type=int, default=8)
    parser.add_argument("--env_size", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    # For simplicity, let user pass comma-separated keys
    parser.add_argument("--shared_keys", type=str, required=True, help="Comma-separated shared feature keys for X")
    parser.add_argument(
        "--y_keys",
        type=str,
        required=True,
        help="Comma-separated CBS-only feature keys for Y (obs dict keys)",
    )
    args = parser.parse_args()

    shared_keys = [k.strip() for k in args.shared_keys.split(",") if k.strip()]
    y_specs = [{"name": k.strip(), "from": "obs"} for k in args.y_keys.split(",") if k.strip()]

    X, Y = collect(
        shared_keys=shared_keys,
        y_specs=y_specs,
        episodes=args.episodes,
        K=args.history,
        env_size=args.env_size,
        seed=args.seed,
    )

    np.savez(
        args.out,
        X=X,
        shared_keys=np.array(shared_keys, dtype=object),
        K=np.array([args.history]),
        **{k: v for k, v in Y.items()},
    )
    print(f"Saved dataset to {args.out} with X shape {X.shape} and Y keys {list(Y.keys())}")


if __name__ == "__main__":
    main()


