import os
from stable_baselines3.ppo import PPO
from adapters.unified_env import UnifiedSecEnv
from config.env_builders import make_cbs_env


def main():
    print("ZERO-SHOT: Evaluate CW-trained policy on CBS")
    print("=" * 60)

    model_path = "artifacts/policies/cw_ppo_minimal.zip"
    if not os.path.exists(model_path):
        print(" Model not found. Train first: python -m train.train_cw_ppo_minimal")
        return

    model = PPO.load(model_path)
    env = UnifiedSecEnv("cbs", cbs_factory=make_cbs_env)

    obs, info = env.reset()
    total_reward = 0.0
    steps = 0
    for step in range(200):
        action, _ = model.predict(obs, deterministic=False)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        steps += 1
        if terminated or truncated:
            break

    print(f"Episode steps: {steps}")
    print(f"Episode return: {total_reward:.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()


