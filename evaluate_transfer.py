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
        
        # Track cumulative reward at 10-step intervals
        interval_rewards = []
        
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
                # Use stochastic actions to see if agent learned anything useful
                # Deterministic=True causes the agent to always pick the same (often invalid) action
                action, _ = model.predict(obs_for_pred, deterministic=False)
                obs, reward, done, truncated, info = target_env.step(action)
                total_reward += reward
                steps += 1
                
                # Record cumulative reward at 10-step intervals
                if steps % 10 == 0 or done or truncated:
                    interval_rewards.append((steps, total_reward))
            except Exception as e:
                print(f"\n   ✗ Error during prediction: {e}")
                print(f"   Observation shape: {obs_for_pred.shape if hasattr(obs_for_pred, 'shape') else type(obs_for_pred)}")
                print(f"   Model expects: {model_obs_space}")
                break
        
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        
        # Print episode results with 10-step intervals
        print(f"\n   Episode {episode+1}:")
        print(f"      Total reward: {total_reward:.2f}, Steps: {steps}")
        print(f"      Reward per 10 steps:")
        for step, cum_reward in interval_rewards:
            print(f"         Step {step:3d}: cumulative reward = {cum_reward:6.2f}")
        if not interval_rewards:
            print(f"         (Episode ended before first 10-step interval)")
    
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


def run_cw_no_encoder_baseline_cbs(
    model_path: str,
    num_episodes: int = 5,
    max_steps: int = 100,
    seed: int = None,
):
    """
    Run a CW policy trained WITHOUT the DAPN encoder on CBS (no encoder at test time).
    Uses UnifiedSecEnv for CBS so the policy gets 8D unified obs, matching what it saw during training on CW.
    Same env config as transfer eval. Returns same stats shape as evaluate_transfer_cw_to_cbs.
    """
    if seed is not None:
        import numpy as np
        np.random.seed(seed)
    os.environ["CBS_ENV"] = "CyberBattleFlat-v0"
    os.environ["CBS_FLAT_NODES"] = "20"
    os.environ["CBS_CRED_REUSE_PROB"] = "0.6"
    os.environ["CBS_EXPLOIT_PROB"] = "0.3"
    base_env = UnifiedSecEnv("cbs", cbs_factory=make_cbs_env)
    model = PPO.load(model_path)
    episode_rewards = []
    episode_lengths = []
    for episode in range(num_episodes):
        obs, _ = base_env.reset()
        total_reward = 0
        steps = 0
        done = False
        truncated = False
        while not (done or truncated) and steps < max_steps:
            obs_for_pred = obs["obs"] if isinstance(obs, dict) else obs
            action, _ = model.predict(obs_for_pred, deterministic=False)
            obs, reward, done, truncated, info = base_env.step(action)
            total_reward += reward
            steps += 1
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        print(f"   CW (no encoder) episode {episode + 1}/{num_episodes}: reward={total_reward:.0f}, steps={steps}")
    avg_reward = sum(episode_rewards) / len(episode_rewards)
    avg_length = sum(episode_lengths) / len(episode_lengths)
    max_reward = max(episode_rewards)
    return {
        "avg_reward": avg_reward,
        "max_reward": max_reward,
        "avg_length": avg_length,
        "episode_rewards": episode_rewards,
    }


def run_random_baseline_cbs(num_episodes=5, max_steps=100, seed=None):
    """
    Run a random policy on CBS (no encoder, no trained policy).
    Same env config as transfer eval so comparison is fair.
    Returns same stats as evaluate_transfer_cw_to_cbs.
    """
    if seed is not None:
        import numpy as np
        np.random.seed(seed)
    os.environ["CBS_ENV"] = "CyberBattleFlat-v0"
    os.environ["CBS_FLAT_NODES"] = "20"
    os.environ["CBS_CRED_REUSE_PROB"] = "0.6"
    os.environ["CBS_EXPLOIT_PROB"] = "0.3"
    base_env = UnifiedSecEnv("cbs", cbs_factory=make_cbs_env)
    episode_rewards = []
    episode_lengths = []
    for episode in range(num_episodes):
        obs, _ = base_env.reset()
        total_reward = 0
        steps = 0
        done = False
        truncated = False
        while not (done or truncated) and steps < max_steps:
            action = base_env.action_space.sample()
            obs, reward, done, truncated, info = base_env.step(action)
            total_reward += reward
            steps += 1
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        print(f"   Baseline episode {episode + 1}/{num_episodes}: reward={total_reward:.0f}, steps={steps}")
    avg_reward = sum(episode_rewards) / len(episode_rewards)
    avg_length = sum(episode_lengths) / len(episode_lengths)
    max_reward = max(episode_rewards)
    return {
        "avg_reward": avg_reward,
        "max_reward": max_reward,
        "avg_length": avg_length,
        "episode_rewards": episode_rewards,
    }


def main():
    """Run transfer learning evaluation (reward/success on CBS = real transfer metric)."""
    import argparse
    parser = argparse.ArgumentParser(
        description="Evaluate CW→CBS transfer: run a Cyberwheel policy on CyberBattleSim with DAPN encoder. Reports average reward (real transfer metric)."
    )
    parser.add_argument("--encoder", type=str, default=None,
                        help="Path to DAPN encoder .pt (e.g. artifacts/transfer_models/dapn_encoder_episodic.pt)")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to CW policy .zip (must be trained with DAPN if using encoder)")
    parser.add_argument("--num-episodes", type=int, default=5, help="Number of evaluation episodes")
    parser.add_argument("--no-baseline", action="store_true", help="Skip random-policy baseline (faster)")
    parser.add_argument("--baseline-model", type=str, default=None,
                        help="Path to CW policy trained WITHOUT encoder (e.g. cw_ppo_very_short.zip). If set, run on CBS without encoder and show episode-by-episode comparison.")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("TRANSFER LEARNING: Cyberwheel → CyberBattleSim")
    print("=" * 60)
    print("Metric: average reward on CBS (real transfer), not synthetic-label accuracy.\n")

    # Encoder path: CLI > episodic default > legacy default
    dapn_encoder = args.encoder
    if dapn_encoder is None:
        dapn_encoder = "artifacts/transfer_models/dapn_encoder_episodic.pt"
        if not os.path.exists(dapn_encoder):
            dapn_encoder = "artifacts/transfer_models/dapn_encoder.pt"

    # Policy: must be trained with DAPN to match encoder observation space (e.g. 256-d)
    cw_model = args.model
    use_dapn_trained = True
    if cw_model is None:
        cw_model = "artifacts/policies/cw_ppo_dapn.zip"
        if not os.path.exists(cw_model):
            cw_model = "artifacts/policies/cw_ppo_very_short.zip"
            if not os.path.exists(cw_model):
                cw_model = "artifacts/policies/cw_ppo_minimal.zip"
            use_dapn_trained = False
            print("DAPN-trained policy not found. Using policy trained without DAPN.")
            print("  Observation space may mismatch (8D vs 256D). For proper evaluation:")
            print("  Train CW policy with DAPN: python train/train_cw_ppo_with_dapn.py")
            print("  Then: python evaluate_transfer.py --encoder <your_encoder.pt> --model <cw_ppo_dapn.zip>")

    if not os.path.exists(cw_model):
        print("Policy not found:", cw_model)
        print("  Train a CW policy first, e.g. python train/train_cw_ppo_very_short.py")
        print("  For transfer, train with DAPN: python train/train_cw_ppo_with_dapn.py")
        return

    if not os.path.exists(dapn_encoder):
        print("Encoder not found:", dapn_encoder)
        print("  Train encoder first: python train_dapn_encoder_episodic.py")
        print("  Then: python evaluate_transfer.py --encoder artifacts/transfer_models/dapn_encoder_episodic.pt")
        use_dapn = False
    else:
        use_dapn = True
        print("Encoder:", dapn_encoder)
    print("Policy:", cw_model)
    print("Episodes:", args.num_episodes)

    # 1) Transfer: CW policy + DAPN encoder on CBS
    transfer_results = evaluate_transfer_cw_to_cbs(
        source_model_path=cw_model,
        dapn_encoder_path=dapn_encoder if use_dapn else "",
        num_episodes=args.num_episodes,
        use_dapn=use_dapn
    )

    # 2) Baseline: random policy on CBS (same env, no encoder)
    baseline_results = None
    if not args.no_baseline and transfer_results is not None:
        import numpy as np
        print("\n" + "=" * 60)
        print("BASELINE: Random policy on CBS (same env, no encoder)")
        print("=" * 60)
        try:
            baseline_results = run_random_baseline_cbs(
                num_episodes=args.num_episodes,
                max_steps=100,
                seed=42,
            )
            print(f"   Average reward: {baseline_results['avg_reward']:.2f}")
            print(f"   Max reward: {baseline_results['max_reward']:.2f}")
            print(f"   Average episode length: {baseline_results['avg_length']:.1f} steps")
        except Exception as e:
            print(f"   Baseline failed: {e}")
            import traceback
            traceback.print_exc()
        print("=" * 60)

    # 2b) Baseline: CW policy trained WITHOUT encoder, run on CBS (no encoder)
    cw_no_encoder_results = None
    baseline_model = args.baseline_model
    if baseline_model is None:
        baseline_model = "artifacts/policies/cw_ppo_very_short.zip"
        if not os.path.exists(baseline_model):
            baseline_model = "artifacts/policies/cw_ppo_minimal.zip"
    if transfer_results is not None and os.path.exists(baseline_model):
        print("\n" + "=" * 60)
        print("BASELINE: CW policy (trained WITHOUT encoder) on CBS (no encoder)")
        print("=" * 60)
        print(f"   Model: {baseline_model}")
        try:
            cw_no_encoder_results = run_cw_no_encoder_baseline_cbs(
                model_path=baseline_model,
                num_episodes=args.num_episodes,
                max_steps=100,
                seed=42,
            )
            print(f"   Average reward: {cw_no_encoder_results['avg_reward']:.2f}")
            print(f"   Max reward: {cw_no_encoder_results['max_reward']:.2f}")
        except Exception as e:
            print(f"   CW no-encoder baseline failed: {e}")
            import traceback
            traceback.print_exc()
        print("=" * 60)

    # 3) Episode-by-episode comparison and summary
    n_ep = args.num_episodes
    if transfer_results is not None:
        t_rew = transfer_results["episode_rewards"]
        print("\n" + "=" * 60)
        print("EPISODE-BY-EPISODE COMPARISON (reward per episode)")
        print("=" * 60)
        print(f"   {'Episode':<10} {'Transfer (CW+DAPN on CBS)':<28} {'CW no encoder (on CBS)':<24} {'Random (CBS)':<14}")
        print("   " + "-" * 76)
        for ep in range(n_ep):
            tr = t_rew[ep] if ep < len(t_rew) else float("nan")
            cw_no = cw_no_encoder_results["episode_rewards"][ep] if cw_no_encoder_results and ep < len(cw_no_encoder_results["episode_rewards"]) else None
            rnd = baseline_results["episode_rewards"][ep] if baseline_results and ep < len(baseline_results["episode_rewards"]) else None
            cw_str = f"{cw_no:.2f}" if cw_no is not None else "—"
            rnd_str = f"{rnd:.2f}" if rnd is not None else "—"
            print(f"   {ep + 1:<10} {tr:<28.2f} {cw_str:<24} {rnd_str:<14}")
        print("   " + "-" * 76)
        t_avg = transfer_results["avg_reward"]
        cw_no_avg = cw_no_encoder_results["avg_reward"] if cw_no_encoder_results else None
        b_avg = baseline_results["avg_reward"] if baseline_results else None
        cw_avg_str = f"{cw_no_avg:.2f}" if cw_no_avg is not None else "—"
        b_avg_str = f"{b_avg:.2f}" if b_avg is not None else "—"
        print(f"   {'Average':<10} {t_avg:<28.2f} {cw_avg_str:<24} {b_avg_str:<14}")
        print("=" * 60)

    if transfer_results is not None and baseline_results is not None:
        t_avg = transfer_results["avg_reward"]
        b_avg = baseline_results["avg_reward"]
        diff = t_avg - b_avg
        pct = (100.0 * (t_avg - b_avg) / b_avg) if b_avg != 0 else (100.0 if t_avg > 0 else 0.0)
        print("\nCOMPARISON (Transfer vs Random baseline):")
        print(f"   Transfer avg reward:  {t_avg:.2f}")
        print(f"   Random avg reward:   {b_avg:.2f}")
        print(f"   Difference:          {diff:+.2f} ({pct:+.1f}%)")
        if cw_no_encoder_results is not None:
            c_avg = cw_no_encoder_results["avg_reward"]
            print(f"   CW no-encoder avg:   {c_avg:.2f}")
            print(f"   Transfer vs CW no-enc: {t_avg - c_avg:+.2f}")
        if diff > 0:
            print("   => Transfer policy outperforms random baseline.")
        else:
            print("   => Transfer policy does not outperform random baseline on this run.")
        print("=" * 60)


if __name__ == "__main__":
    main()

