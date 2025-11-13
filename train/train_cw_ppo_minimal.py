import os
from stable_baselines3.ppo import PPO
from adapters.unified_env import UnifiedSecEnv
from config.env_builders import make_cw_env


def main():
    print(" Starting CW training...")

    # Allow overriding total steps via env var for quick iterations
    total_steps = int(os.environ.get("CW_TRAIN_STEPS", "50000"))

    env = UnifiedSecEnv("cw", cw_factory=make_cw_env)
    print("CW environment created")

    model = PPO(
        "MlpPolicy", env,
        n_steps=512, batch_size=256,
        learning_rate=3e-4, gamma=0.995,
        ent_coef=0.01, vf_coef=0.5, clip_range=0.2,
        verbose=1
    )

    print(f" Training for {total_steps} steps...")
    model.learn(total_timesteps=total_steps)

    # Ensure artifacts directory exists
    os.makedirs("artifacts/policies", exist_ok=True)
    model.save("artifacts/policies/cw_ppo_minimal")
    print("Model saved to artifacts/policies/cw_ppo_minimal")

    print(" Training completed successfully!")


if __name__ == "__main__":
    main()


