"""
Gym wrapper to use DAPN observation encoder with environments.
"""

import os
import numpy as np
import gymnasium as gym
from typing import Optional
from adapters.dapn_observation_encoder import DAPNObservationTranslator


class DAPNEnvWrapper(gym.Wrapper):
    """
    Wrapper that uses DAPN for observation encoding.
    Can be used with UnifiedSecEnv to enable DAPN-based observation handling.
    """
    
    def __init__(
        self,
        env,
        encoder_path: Optional[str] = None,
        feature_size: int = 256,
        use_dapn: bool = True,
        device: Optional[str] = None
    ):
        """
        Initialize DAPN environment wrapper.
        
        Args:
            env: Environment to wrap (should be UnifiedSecEnv)
            encoder_path: Path to saved DAPN encoder checkpoint
            feature_size: Size of feature space
            use_dapn: Whether to use DAPN encoding
            device: Device to run encoder on ('cuda' or 'cpu')
        """
        super().__init__(env)
        self.use_dapn = use_dapn
        self.feature_size = feature_size
        
        # Create DAPN translator
        import torch
        torch_device = torch.device(device) if device else None
        
        # Decide full-obs vs 8D: env var override, or infer from checkpoint (episodic training uses 512D)
        use_full_obs = os.environ.get("DAPN_USE_FULL_OBS", "0") == "1"
        checkpoint_unified_dim = None
        if encoder_path and os.path.isfile(encoder_path):
            try:
                ckpt = torch.load(encoder_path, map_location="cpu", weights_only=False)
                checkpoint_unified_dim = ckpt.get("input_dim", None)
                if checkpoint_unified_dim is not None and checkpoint_unified_dim != 8:
                    use_full_obs = True
            except Exception:
                pass
        
        if use_full_obs:
            # Use full observations with SINGLE unified encoder (episodic-trained or DAPN_USE_FULL_OBS=1)
            from adapters.dapn_unified_full_obs_translator import DAPNUnifiedFullObsTranslator
            unified_dim = checkpoint_unified_dim if checkpoint_unified_dim is not None else 512
            self.dapn_translator = DAPNUnifiedFullObsTranslator(
                use_dapn=use_dapn,
                encoder_path=encoder_path,
                feature_size=feature_size,
                unified_dim=unified_dim,
                device=torch_device
            )
        else:
            # Use 8D unified format with DAPN (default for legacy checkpoints)
            self.dapn_translator = DAPNObservationTranslator(
                use_dapn=use_dapn,
                encoder_path=encoder_path,
                feature_size=feature_size,
                device=torch_device
            )
        
        # Replace the environment's observation translator
        self.env.obs_t = self.dapn_translator
        
        # Update observation space to match DAPN feature size
        # For transfer learning consistency, always use Dict format when DAPN is enabled
        from gymnasium import spaces
        if isinstance(self.observation_space, gym.spaces.Dict):
            # Keep mask, update obs dimension
            self.observation_space = spaces.Dict({
                'mask': self.observation_space['mask'],
                'obs': spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(feature_size,),
                    dtype=np.float32
                )
            })
        else:
            # Convert Box to Dict format for consistency with transfer learning
            # This ensures models trained on Cyberwheel with DAPN can transfer to CBS
            # Create a dummy mask (all ones) for environments without action masking
            from adapters.action_translator import ActionTranslator
            num_actions = len(ActionTranslator().unified_actions)
            self.observation_space = spaces.Dict({
                'obs': spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(feature_size,),
                    dtype=np.float32
                ),
                'mask': spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(num_actions,),
                    dtype=np.float32
                )
            })
    
    def reset(self, **kwargs):
        """Reset environment and encode observation with DAPN."""
        obs, info = self.env.reset(**kwargs)
        return self._encode_obs(obs), info
    
    def step(self, action):
        """Step environment and encode observation with DAPN."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._encode_obs(obs), reward, terminated, truncated, info
    
    def _encode_obs(self, obs):
        """Encode observation using DAPN translator."""
        # Get raw observation from environment
        raw_obs = getattr(self.env, '_last_raw_obs', None)
        if raw_obs is None:
            raw_obs = getattr(self.env, '_last_raw_cbs_obs', None)
        if raw_obs is None:
            raw_obs = obs
        
        # Encode based on backend
        if self.env.backend == "cbs":
            if isinstance(raw_obs, dict):
                encoded = self.dapn_translator.from_cbs(raw_obs)
            else:
                # Fallback: try to use obs as-is
                encoded = self.dapn_translator.from_cbs(obs if isinstance(obs, dict) else {})
        else:  # cw
            if isinstance(raw_obs, np.ndarray):
                encoded = self.dapn_translator.from_cw(raw_obs)
            else:
                encoded = self.dapn_translator.from_cw(obs if isinstance(obs, np.ndarray) else np.array([]))
        
        # Ensure correct shape
        if not isinstance(encoded, np.ndarray):
            encoded = np.array(encoded, dtype=np.float32)
        
        # Always return Dict format when DAPN is enabled (for transfer learning consistency)
        if isinstance(obs, dict) and 'mask' in obs:
            # Keep existing mask from environment
            return {
                'obs': encoded,
                'mask': obs['mask']
            }
        else:
            # Compute a semantic kill-chain mask for CW (or fall back to all-valid).
            # UnifiedSecEnv._compute_unified_mask() now handles both backends:
            #   CW  → derived from raw obs (which actions are productive at this stage)
            #   CBS → derived from CBS action_mask dict
            # This means the policy is trained with meaningful masks on CW, not just
            # constant all-ones, so it can condition on them correctly during CBS eval.
            try:
                mask = self.env._compute_unified_mask()
            except Exception:
                from adapters.action_translator import ActionTranslator
                num_actions = len(ActionTranslator().unified_actions)
                mask = np.ones(num_actions, dtype=np.float32)
            return {
                'obs': encoded,
                'mask': mask
            }

