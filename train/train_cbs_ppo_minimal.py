from stable_baselines3.ppo import PPO
from adapters.unified_env import UnifiedSecEnv
from config.env_builders import make_cbs_env


def main():
    print("🚀 Starting CBS training...")
    
    # Create environment
    env = UnifiedSecEnv("cbs", cbs_factory=make_cbs_env)
    print("✅ Environment created")
    
    # Create model with tuned hyperparameters
    model = PPO(
        "MlpPolicy", env,
        n_steps=512, batch_size=256,
        learning_rate=3e-4, gamma=0.995,
        ent_coef=0.01, vf_coef=0.5, clip_range=0.2,
        verbose=1
    )
    print("PPO model created")
    
    # Longer training
    print(" Training for 50000 steps...")
    model.learn(total_timesteps=50000)
    
    # Save model
    model.save("artifacts/policies/cbs_ppo_minimal")
    print("Model saved to artifacts/policies/cbs_ppo_minimal")
    
    print(" Training completed successfully!")


if __name__ == "__main__":
    main()
