import os
import sys
from pathlib import Path
import gymnasium as gym
from stable_baselines3 import PPO
from adapters.unified_env import UnifiedSecEnv
from adapters.dapn_env_wrapper import DAPNEnvWrapper
from config.env_builders import make_cw_env, make_cbs_env

# Add paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "cyberwheel"))

def evaluate_transfer_cw_to_cbs(
    source_model_path="artifacts/policies/cw_ppo_very_short.zip",
    dapn_encoder_path="artifacts/transfer_models/dapn_encoder.pt",
    num_episodes=5,
    use_dapn=True
):
    """
    Transfer learning: Train on Cyberwheel (Scenario 1), Test on CyberBattleSim (Scenario 2)
    """
    print("TRANSFER LEARNING EVALUATION")
    print("Train on Cyberwheel → Test on CyberBattleSim")
    
    # Check if source model exists
    if not os.path.exists(source_model_path):
        print(f"✗ Source model not found: {source_model_path}")
        print("  Train on Cyberwheel first: python train/train_cw_ppo_very_short.py")
        return None
    
    # Load source model (trained on Cyberwheel)
    print(f"\n1. Loading source model (trained on Cyberwheel)...")
    print(f"   Model: {source_model_path}")
    model = PPO.load(source_model_path)
    print("   ✓ Model loaded")
    
    # Create target environment (CyberBattleSim)
    print(f"\n2. Creating target environment (CyberBattleSim)...")
    os.environ["CBS_ENV"] = "CyberBattleFlat-v0"
    os.environ["CBS_FLAT_NODES"] = "20"
    os.environ["CBS_CRED_REUSE_PROB"] = "0.6"
    os.environ["CBS_EXPLOIT_PROB"] = "0.3"
    
    base_env = UnifiedSecEnv("cbs", cbs_factory=make_cbs_env)
    
    # Use DAPN if encoder exists
    if use_dapn and os.path.exists(dapn_encoder_path):
        print(f"   Using DAPN encoder: {dapn_encoder_path}")
        target_env = DAPNEnvWrapper(
            base_env,
            encoder_path=dapn_encoder_path,
            use_dapn=True
        )
        print("   ✓ Environment created with DAPN adaptation")
    else:
        if use_dapn:
            print(f"   ⚠ DAPN encoder not found: {dapn_encoder_path}")
            print("   Training without DAPN (may not work well)")
        target_env = base_env
        print("   ✓ Environment created (no DAPN)")
    
    # Check observation space compatibility
    print(f"\n3. Checking model compatibility...")
    print(f"   Model observation space: {model.observation_space}")
    print(f"   Target env observation space: {target_env.observation_space}")
    
    # Check if spaces are compatible
    model_obs_space = model.observation_space
    env_obs_space = target_env.observation_space
    
    # Extract actual observation shape from Dict if needed
    if isinstance(env_obs_space, gym.spaces.Dict):
        env_obs_shape = env_obs_space['obs'].shape
    else:
        env_obs_shape = env_obs_space.shape
    
    if isinstance(model_obs_space, gym.spaces.Dict):
        model_obs_shape = model_obs_space['obs'].shape
    else:
        model_obs_shape = model_obs_space.shape
    
    if model_obs_shape != env_obs_shape:
        print(f"\n⚠ WARNING: Observation space mismatch!")
        print(f"   Model expects: {model_obs_shape}")
        print(f"   Environment provides: {env_obs_shape}")
        print(f"\n   This model was likely trained WITHOUT DAPN.")
        print(f"   For proper transfer learning, train with DAPN:")
        print(f"     python train/train_cw_ppo_with_dapn.py")
        print(f"\n   Attempting evaluation anyway (may fail)...")
    
    # Don't use set_env - it will fail if spaces don't match
    # We'll handle observations manually in the loop
    
    # Run evaluation episodes
    print(f"\n4. Running {num_episodes} evaluation episodes on target scenario...")
    print("-" * 60)
    
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(num_episodes):
        obs, info = target_env.reset()
        total_reward = 0
        steps = 0
        done = False
        truncated = False
        
        # Handle dict observations from DAPN wrapper
        while not (done or truncated) and steps < 100:
            # Extract observation based on what model expects
            if isinstance(obs, dict):
                # DAPN wrapper returns Dict with 'obs' key
                if isinstance(model_obs_space, gym.spaces.Dict):
                    # Model expects Dict - pass as-is
                    obs_for_pred = obs
                else:
                    # Model expects Box - extract 'obs' from Dict
                    obs_for_pred = obs['obs']
            else:
                # Environment returns Box directly
                obs_for_pred = obs
            
            try:
                action, _ = model.predict(obs_for_pred, deterministic=True)
                obs, reward, done, truncated, info = target_env.step(action)
                total_reward += reward
                steps += 1
            except Exception as e:
                print(f"\n   ✗ Error during prediction: {e}")
                print(f"   Observation shape: {obs_for_pred.shape if hasattr(obs_for_pred, 'shape') else type(obs_for_pred)}")
                print(f"   Model expects: {model_obs_space}")
                break
        
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        print(f"   Episode {episode+1}: Reward={total_reward:.2f}, Steps={steps}")
    
    # Statistics
    avg_reward = sum(episode_rewards) / len(episode_rewards)
    avg_length = sum(episode_lengths) / len(episode_lengths)
    max_reward = max(episode_rewards)
    
    print("-" * 60)
    print("TRANSFER LEARNING RESULTS:")
    print(f"   Average reward: {avg_reward:.2f}")
    print(f"   Max reward: {max_reward:.2f}")
    print(f"   Average episode length: {avg_length:.1f} steps")
    print("=" * 60)
    
    if avg_reward > 0:
        print("✓ Transfer learning successful! Model works on target scenario.")
    else:
        print("⚠ Transfer learning may need more training or DAPN encoder.")
    
    return {
        "avg_reward": avg_reward,
        "max_reward": max_reward,
        "avg_length": avg_length,
        "episode_rewards": episode_rewards
    }


def main():
    """Run transfer learning evaluation"""
    print("\n" + "=" * 60)
    print("TRANSFER LEARNING: Cyberwheel → CyberBattleSim")
    print("=" * 60)
    print("\nThis tests if a model trained on Cyberwheel can work on CyberBattleSim")
    print("using DAPN for domain adaptation.\n")
    
    # Check for models - prefer DAPN-trained model
    cw_model = "artifacts/policies/cw_ppo_dapn.zip"
    use_dapn_trained = True
    if not os.path.exists(cw_model):
        print("DAPN-trained model not found. Looking for regular model...")
        cw_model = "artifacts/policies/cw_ppo_very_short.zip"
        if not os.path.exists(cw_model):
            cw_model = "artifacts/policies/cw_ppo_minimal.zip"
        use_dapn_trained = False
        print("  Using model trained WITHOUT DAPN")
        print("  This model expects 8D observations, but DAPN produces 256D")
        print("  Transfer learning will likely fail!")
        print("\n  To fix: Train model WITH DAPN first:")
        print("    1. Train DAPN encoder: python train_dapn_encoder.py --num-samples 1000 --epochs 50")
        print("    2. Train model with DAPN: python train/train_cw_ppo_with_dapn.py")
    
    dapn_encoder = "artifacts/transfer_models/dapn_encoder.pt"
    
    if not os.path.exists(cw_model):
        print("Cyberwheel model not found. Train it first:")
        print("  python train/train_cw_ppo_very_short.py")
        return
    
    if not os.path.exists(dapn_encoder):
        if use_dapn_trained:
            print("DAPN encoder not found. Train it first:")
            print("  python train_dapn_encoder.py --num-samples 1000 --epochs 50")
            print("\n  Or run without DAPN (may not work as well):")
            use_dapn = False
        else:
            print("DAPN encoder not found, but model wasn't trained with DAPN anyway.")
            print("  Transfer learning requires BOTH:")
            print("    1. Model trained WITH DAPN")
            print("    2. DAPN encoder for adaptation")
            use_dapn = False
    else:
        use_dapn = True
    
    # Run transfer evaluation
    evaluate_transfer_cw_to_cbs(
        source_model_path=cw_model,
        dapn_encoder_path=dapn_encoder,
        num_episodes=5,
        use_dapn=use_dapn
    )


if __name__ == "__main__":
    main()

