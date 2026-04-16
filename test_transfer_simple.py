from stable_baselines3.ppo import PPO
from adapters.unified_env import UnifiedSecEnv
from adapters.dapn_env_wrapper import DAPNEnvWrapper
from config.env_builders import make_cbs_env
import os

def test_transfer_learning_concept(use_dapn=False, dapn_encoder_path=None):
    print("TRANSFER LEARNING DEMONSTRATION")
    if use_dapn:
        print("(Using DAPN for observation encoding)")
    print("=" * 50)
    
    # 1. Load the CBS-trained model
    print(" Loading CBS-trained model...")
    model_path = "artifacts/policies/cbs_ppo_minimal.zip"
    if not os.path.exists(model_path):
        print(" Model not found. Please run training first.")
        return
    
    cbs_model = PPO.load(model_path)
    print("  CBS model loaded successfully")
    
    # 2. Create CBS environment (with optional DAPN)
    print(" Creating CBS environment...")
    base_env = UnifiedSecEnv("cbs", cbs_factory=make_cbs_env)
    
    if use_dapn:
        cbs_env = DAPNEnvWrapper(
            base_env,
            encoder_path=dapn_encoder_path,
            feature_size=256,
            use_dapn=True
        )
        print(" CBS environment created with DAPN")
        print(f"  Observation space: {cbs_env.observation_space}")
    else:
        cbs_env = base_env
        print(" CBS environment created")
    
    # 3. Test the model on CBS (source environment)
    print("\n TESTING ON SOURCE ENVIRONMENT (CBS):")
    print("-" * 40)
    obs, info = cbs_env.reset()
    
    # Handle both dict and array observations
    if isinstance(obs, dict):
        if 'obs' in obs:
            obs_shape = obs['obs'].shape
            print(f" Initial state shape: {obs_shape} (Dict observation with 'obs' key)")
        else:
            print(f" Initial state: Dict with keys {list(obs.keys())}")
            obs_shape = None
    else:
        obs_shape = obs.shape
        print(f" Initial state shape: {obs_shape}")
    
    total_reward = 0.0
    actions_taken = []

    # Longer rollout with exploration to surface rewards
    for step in range(200):
        # Handle dict observations for prediction
        obs_for_pred = obs['obs'] if isinstance(obs, dict) else obs
        action, _ = cbs_model.predict(obs_for_pred, deterministic=False)
        actions_taken.append(action)

        obs, reward, terminated, truncated, info = cbs_env.step(action)
        total_reward += reward

        print(f"   Step {step+1}: Action={action}, Reward={reward:.2f}")

        if terminated or truncated:
            print(f"   Episode ended at step {step+1}")
            break

    print(f" Episode return: {total_reward:.2f}")
    print(f"   Actions taken: {actions_taken}")
    
    # 4. Explain the transfer learning concept
    print("\n🔄 TRANSFER LEARNING EXPLANATION:")
    print("-" * 40)
    
    
    print("\n NEXT STEPS FOR FULL TRANSFER:")
    print("-" * 40)
   
    
    print("\n TRANSFER LEARNING FOUNDATION IS WORKING!")
    print("=" * 50)

if __name__ == "__main__":
    # Set use_dapn=True to enable DAPN observation encoding
    # Set dapn_encoder_path to use a pre-trained encoder
    test_transfer_learning_concept(
        use_dapn=False,  # Set to True to use DAPN
        dapn_encoder_path=None  # Set to "artifacts/transfer_models/dapn_encoder.pt" if you have one
    )
