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
        
        # Check if we should use full observations (set via env var)
        use_full_obs = os.environ.get("DAPN_USE_FULL_OBS", "0") == "1"
        use_unified_full_obs = os.environ.get("DAPN_USE_UNIFIED_FULL_OBS", "0") == "1"
        
        if use_full_obs:
            if use_unified_full_obs:
                # Use full observations with SINGLE unified encoder (follows DAPN master)
                from adapters.dapn_unified_full_obs_translator import DAPNUnifiedFullObsTranslator
                self.dapn_translator = DAPNUnifiedFullObsTranslator(
                    use_dapn=use_dapn,
                    encoder_path=encoder_path,
                    feature_size=feature_size,
                    device=torch_device
                )
            else:
                # Use full raw observations with separate encoders
                from adapters.dapn_full_obs_translator import DAPNFullObservationTranslator
                self.dapn_translator = DAPNFullObservationTranslator(
                    use_dapn=use_dapn,
                    encoder_path=encoder_path,
                    feature_size=feature_size,
                    device=torch_device
                )
        else:
            # Use 8D unified format with DAPN (default)
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
                    low=0.0, 
                    high=1.0, 
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
            # Create a dummy mask (all ones) for environments without action masking
            # This ensures consistency between Cyberwheel and CBS when using DAPN
            from adapters.action_translator import ActionTranslator
            num_actions = len(ActionTranslator().unified_actions)
            mask = np.ones(num_actions, dtype=np.float32)
            return {
                'obs': encoded,
                'mask': mask
            }

