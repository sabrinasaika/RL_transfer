from stable_baselines3.ppo import PPO
from adapters.unified_env import UnifiedSecEnv
from adapters.dapn_env_wrapper import DAPNEnvWrapper
from config.env_builders import make_cbs_env
import os

def test_transfer_learning_with_dapn():
    print("TRANSFER LEARNING DEMONSTRATION WITH DAPN")
    print("=" * 50)
    
    # 1. Create CBS environment with DAPN
    print(" Creating CBS environment with DAPN...")
    base_env = UnifiedSecEnv("cbs", cbs_factory=make_cbs_env)
    
    # Wrap with DAPN (using random initialization for now)
    # To use a pre-trained encoder, set encoder_path="artifacts/transfer_models/dapn_encoder.pt"
    cbs_env = DAPNEnvWrapper(
        base_env,
        encoder_path=None,  # Use random initialization
        feature_size=256,
        use_dapn=True
    )
    print(" CBS environment with DAPN created")
    print(f" Observation space: {cbs_env.observation_space}")
    
    # 2. Test environment with DAPN
    print("\n TESTING ENVIRONMENT WITH DAPN:")
    print("-" * 40)
    obs, info = cbs_env.reset()
    
    # Handle dict observation space
    if isinstance(obs, dict):
        obs_shape = obs['obs'].shape
        print(f" Initial state shape: {obs_shape} (DAPN encoded)")
        print(f" Observation type: Dict with 'obs' and 'mask' keys")
    else:
        obs_shape = obs.shape
        print(f" Initial state shape: {obs_shape} (DAPN encoded)")
    
    total_reward = 0.0
    actions_taken = []

    # Test rollout
    for step in range(50):
        action = cbs_env.action_space.sample()
        actions_taken.append(action)

        obs, reward, terminated, truncated, info = cbs_env.step(action)
        total_reward += reward

        if step < 5 or step % 10 == 0:
            obs_val = obs['obs'] if isinstance(obs, dict) else obs
            print(f"   Step {step+1}: Action={action}, Reward={reward:.2f}, Obs shape={obs_val.shape}")

        if terminated or truncated:
            print(f"   Episode ended at step {step+1}")
            break

    print(f"\n Episode return: {total_reward:.2f}")
    print(f" Total steps: {len(actions_taken)}")
    
    # 3. Compare with and without DAPN
    print("\n COMPARISON: With vs Without DAPN")
    print("-" * 40)
    
    # Without DAPN
    env_no_dapn = UnifiedSecEnv("cbs", cbs_factory=make_cbs_env)
    obs_no_dapn, _ = env_no_dapn.reset()
    obs_no_dapn_val = obs_no_dapn['obs'] if isinstance(obs_no_dapn, dict) else obs_no_dapn
    print(f" Without DAPN: shape={obs_no_dapn_val.shape}, dtype={obs_no_dapn_val.dtype}")
    
    # With DAPN
    obs_dapn_val = obs['obs'] if isinstance(obs, dict) else obs
    print(f" With DAPN: shape={obs_dapn_val.shape}, dtype={obs_dapn_val.dtype}")
    print(f" Feature expansion: {obs_no_dapn_val.shape[0]} -> {obs_dapn_val.shape[0]} dimensions")
    
    print("\n DAPN INTEGRATION SUCCESSFUL!")
    print("=" * 50)
    print("\n To use a pre-trained DAPN encoder:")
    print(" 1. Train encoder: python train_dapn_encoder.py --num-samples 1000 --epochs 50")
    print(" 2. Use encoder: Set encoder_path='artifacts/transfer_models/dapn_encoder.pt'")
    print("=" * 50)

if __name__ == "__main__":
    test_transfer_learning_with_dapn()

