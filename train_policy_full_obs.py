#!/usr/bin/env python3
"""
Train a policy using full observation encoders.
Uses the encoded features as input to train a PPO policy.
"""

import os
import sys
import argparse
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy

try:
    import tqdm  # noqa: F401
    import rich  # noqa: F401
    _SB3_PROGRESS_BAR = True
except ImportError:
    _SB3_PROGRESS_BAR = False

from adapters.unified_env import UnifiedSecEnv
from adapters.full_obs_translator import FullObservationTranslator
from config.env_builders import make_cbs_env


class FullObsWrapper(gym.Wrapper):
    """Wrapper to use full observation encoder with SB3"""
    def __init__(self, env, encoder_path: str):
        super().__init__(env)
        self.encoder_path = encoder_path
        
        # Replace observation translator with full obs translator
        self.env.obs_t = FullObservationTranslator(
            use_transfer=True,
            encoder_path=encoder_path
        )
        
        # Update observation space to match encoder output (64-dim)
        # The encoder outputs 64-dim features, so update the 'obs' key in Dict space
        from gymnasium import spaces
        if isinstance(self.observation_space, gym.spaces.Dict):
            # Keep mask as is, update obs to 64-dim
            self.observation_space = spaces.Dict({
                'mask': self.observation_space['mask'],
                'obs': spaces.Box(low=0.0, high=1.0, shape=(64,), dtype=np.float32)
            })
        else:
            # If it's a Box, convert to Dict with mask
            self.observation_space = spaces.Dict({
                'obs': spaces.Box(low=0.0, high=1.0, shape=(64,), dtype=np.float32),
                'mask': spaces.Box(low=0.0, high=1.0, shape=(7,), dtype=np.float32)
            })
    
    def reset(self, **kwargs):
        """Reset and encode observation"""
        obs, info = self.env.reset(**kwargs)
        # Store raw observation before encoding
        self._last_raw_obs = obs
        return self._encode_obs(obs), info
    
    def step(self, action):
        """Step and encode observation"""
        obs, reward, done, truncated, info = self.env.step(action)
        # Store raw observation before encoding
        self._last_raw_obs = obs
        return self._encode_obs(obs), reward, done, truncated, info
    
    def _encode_obs(self, obs):
        """Encode observation using full obs translator"""
        # Get raw CBS observation - prefer stored raw obs, fallback to current obs
        raw_obs = getattr(self, '_last_raw_obs', None)
        if raw_obs is None:
            # Try to get from underlying env
            raw_obs = getattr(self.env, '_last_raw_cbs_obs', None) or getattr(self.env, '_last_raw_obs', None)
        if raw_obs is None:
            # Fallback: use obs as-is if it's already a dict
            raw_obs = obs if isinstance(obs, dict) else {}
        
        # Encode using full obs translator
        try:
            if isinstance(raw_obs, dict):
                encoded = self.env.obs_t.from_cbs(raw_obs)
            else:
                # If raw_obs is not a dict, wrap it
                encoded = self.env.obs_t.from_cbs({'raw': raw_obs}) if hasattr(self.env.obs_t, 'from_cbs') else np.zeros(64, dtype=np.float32)
        except Exception as e:
            # If encoding fails, use a zero vector as fallback
            print(f"Warning: Encoding failed: {e}, using zero vector")
            encoded = np.zeros(64, dtype=np.float32)
        
        # Ensure encoded is numpy array
        if not isinstance(encoded, np.ndarray):
            encoded = np.array(encoded, dtype=np.float32)
        if encoded.shape != (64,):
            if encoded.size == 64:
                encoded = encoded.reshape(64)
            else:
                print(f"Warning: Encoded shape is {encoded.shape}, expected (64,), using zero vector")
                encoded = np.zeros(64, dtype=np.float32)
        
        # Return as Dict with mask
        if isinstance(obs, dict) and 'mask' in obs:
            return {'obs': encoded, 'mask': obs['mask']}
        else:
            # Create mask (all actions valid by default)
            mask = np.ones(7, dtype=np.float32)
            try:
                if hasattr(self.env, '_compute_unified_mask'):
                    mask = self.env._compute_unified_mask()
            except:
                pass
            return {'obs': encoded, 'mask': mask}


def make_full_obs_env(encoder_path: str):
    """Create environment with full observation encoder"""
    def _init():
        env = UnifiedSecEnv("cbs", cbs_factory=make_cbs_env)
        return FullObsWrapper(env, encoder_path)
    return _init


def train_policy(
    encoder_path: str,
    total_timesteps: int = 50000,
    output_dir: str = "artifacts/policies/full_obs_policy",
    eval_freq: int = 5000,
    n_eval_episodes: int = 5
):
    """Train PPO policy using full observation encoder"""
    print("=" * 60)
    print("TRAINING POLICY WITH FULL OBSERVATION ENCODER")
    print("=" * 60)
    
    if not os.path.exists(encoder_path):
        print(f"Error: Encoder not found: {encoder_path}")
        print("Train encoder first:")
        print("  python train_full_observation_transfer.py")
        return
    
    print(f"\nEncoder: {encoder_path}")
    print(f"Total timesteps: {total_timesteps}")
    print(f"Output directory: {output_dir}")
    
    # Create environment
    print("\n1. Creating environment...")
    env = make_vec_env(
        make_full_obs_env(encoder_path),
        n_envs=1
    )
    
    # Check observation space
    obs_space = env.observation_space
    print(f"   Observation space: {obs_space}")
    
    # Create policy
    print("\n2. Creating PPO policy...")
    # Use MultiInputPolicy if dict observation space
    if isinstance(obs_space, gym.spaces.Dict):
        policy = "MultiInputPolicy"
    else:
        policy = "MlpPolicy"
    
    model = PPO(
        policy,
        env,
        verbose=1,
        tensorboard_log=os.path.join(output_dir, "tensorboard"),
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
    )
    
    print(f"   Policy: {policy}")
    print(f"   Model parameters: {sum(p.numel() for p in model.policy.parameters())}")
    
    # Create callbacks
    print("\n3. Setting up callbacks...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Evaluation callback
    eval_env = make_vec_env(
        make_full_obs_env(encoder_path),
        n_envs=1
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(output_dir, "best_model"),
        log_path=os.path.join(output_dir, "eval_logs"),
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False
    )
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=eval_freq,
        save_path=os.path.join(output_dir, "checkpoints"),
        name_prefix="policy"
    )
    
    # Train
    print("\n4. Training policy...")
    print("   (This may take a while...)")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=_SB3_PROGRESS_BAR,
    )
    
    # Save final model
    final_model_path = os.path.join(output_dir, "final_model.zip")
    model.save(final_model_path)
    print(f"\n✓ Saved final model to {final_model_path}")
    
    # Evaluate
    print("\n5. Final evaluation...")
    eval_env = make_vec_env(
        make_full_obs_env(encoder_path),
        n_envs=1
    )
    mean_reward, std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=10,
        deterministic=True
    )
    
    print(f"\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\nFinal Performance:")
    print(f"  Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"\nModel saved to: {final_model_path}")
    print(f"Best model: {os.path.join(output_dir, 'best_model', 'best_model.zip')}")
    print(f"\nTo evaluate:")
    print(f"  python eval_trained_policy_full_obs.py --model_path {final_model_path}")




def main():
    parser = argparse.ArgumentParser(description="Train policy with full observation encoder")
    parser.add_argument("--encoder_path", type=str,
                       default="artifacts/transfer_models/full_obs_encoder.pt",
                       help="Path to full observation encoder")
    parser.add_argument("--total_timesteps", type=int, default=50000,
                       help="Total training timesteps")
    parser.add_argument("--output_dir", type=str,
                       default="artifacts/policies/full_obs_policy",
                       help="Output directory for policy")
    parser.add_argument("--eval_freq", type=int, default=5000,
                       help="Evaluation frequency")
    parser.add_argument("--n_eval_episodes", type=int, default=5,
                       help="Number of evaluation episodes")
    
    args = parser.parse_args()
    
    train_policy(
        encoder_path=args.encoder_path,
        total_timesteps=args.total_timesteps,
        output_dir=args.output_dir,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes
    )


if __name__ == "__main__":
    main()

