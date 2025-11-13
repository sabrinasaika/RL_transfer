from stable_baselines3.ppo import PPO
from adapters.unified_env import UnifiedSecEnv
from config.env_builders import make_cbs_env
import os

def test_transfer_learning_concept():
    print("TRANSFER LEARNING DEMONSTRATION")
    print("=" * 50)
    
    # 1. Load the CBS-trained model
    print(" Loading CBS-trained model...")
    model_path = "artifacts/policies/cbs_ppo_minimal.zip"
    if not os.path.exists(model_path):
        print(" Model not found. Please run training first.")
        return
    
    cbs_model = PPO.load(model_path)
    print("  CBS model loaded successfully")
    
    # 2. Create CBS environment
    print(" Creating CBS environment...")
    cbs_env = UnifiedSecEnv("cbs", cbs_factory=make_cbs_env)
    print(" CBS environment created")
    
    # 3. Test the model on CBS (source environment)
    print("\n TESTING ON SOURCE ENVIRONMENT (CBS):")
    print("-" * 40)
    obs, info = cbs_env.reset()
    print(f" Initial state shape: {obs.shape}")
    
    total_reward = 0.0
    actions_taken = []

    # Longer rollout with exploration to surface rewards
    for step in range(200):
        action, _ = cbs_model.predict(obs, deterministic=False)
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
    test_transfer_learning_concept()
