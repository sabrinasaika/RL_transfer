import os
from stable_baselines3.ppo import PPO
from adapters.unified_env import UnifiedSecEnv
from adapters.dapn_env_wrapper import DAPNEnvWrapper
from config.env_builders import make_cw_env


def main():
    print("Starting CW training WITH DAPN (for transfer learning)...")

    # Get training parameters
    total_steps = int(os.environ.get("CW_TRAIN_STEPS", "1000"))
    dapn_encoder_path = os.environ.get("DAPN_ENCODER_PATH", "artifacts/transfer_models/dapn_encoder.pt")
    
    # Check if DAPN encoder exists
    if not os.path.exists(dapn_encoder_path):
        print(f"DAPN encoder not found: {dapn_encoder_path}")
        print("  Training DAPN encoder first...")
        import subprocess
        subprocess.run([
            "python", "train_dapn_encoder.py",
            "--num-samples", "1000",
            "--epochs", "50",
            "--save-encoder", dapn_encoder_path
        ])
        print(" DAPN encoder trained")
    
    # Create base environment
    os.environ["CW_ENV_YAML"] = "credential_preference_scenario.yaml"
    base_env = UnifiedSecEnv("cw", cw_factory=make_cw_env)
    print("Base environment created")
    
    # Wrap with DAPN
    dapn_env = DAPNEnvWrapper(
        base_env,
        encoder_path=dapn_encoder_path,
        use_dapn=True
    )
    print("Environment wrapped with DAPN")
    print(f"   Observation space: {dapn_env.observation_space}")

    # Create model - use MultiInputPolicy for Dict observations
    model = PPO(
        "MultiInputPolicy",  # Use MultiInputPolicy for Dict observations
        dapn_env,
        n_steps=64,
        batch_size=32,
        learning_rate=3e-4, gamma=0.995,
        ent_coef=0.01, vf_coef=0.5, clip_range=0.2,
        verbose=1
    )

    print(f"Training for {total_steps} steps with DAPN...")
    model.learn(total_timesteps=total_steps)

    # Ensure artifacts directory exists
    os.makedirs("artifacts/policies", exist_ok=True)
    model.save("artifacts/policies/cw_ppo_dapn")
    print("Model saved to artifacts/policies/cw_ppo_dapn.zip")
    print("Done! This model can be transferred to CyberBattleSim.")


if __name__ == "__main__":
    main()

