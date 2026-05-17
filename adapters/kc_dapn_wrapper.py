"""
KCDAPNWrapper — applies the frozen DAPN encoder on top of UnifiedSecEnv.

  Input  obs : 512-D  (UnifiedFullObsPreprocessor output from UnifiedSecEnv)
  Output obs : 256-D  (domain-invariant embedding from frozen DAPN encoder)
  Actions    : Discrete(9)  kill-chain intent — unchanged, passed straight through

Usage:
    env   = UnifiedSecEnv("cw", cw_factory=make_cw_env)
    env   = KCDAPNWrapper(env, encoder_path="artifacts/transfer_models/dapn_encoder.pt")
    model = PPO("MlpPolicy", env, ...)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional


ENCODER_OUTPUT_DIM = 256


class KCDAPNWrapper(gym.Wrapper):
    """
    Thin Gym wrapper: 512-D obs  →  frozen DAPN encoder  →  256-D obs.
    Action space (Discrete(9) kill-chain intent) is passed through unchanged.
    """

    def __init__(self, env: gym.Env, encoder_path: str,
                 feature_size: int = ENCODER_OUTPUT_DIM,
                 device: Optional[str] = None):
        super().__init__(env)

        import torch
        self.device = torch.device(
            device if device else ("mps" if torch.backends.mps.is_available() else "cpu")
        )

        # ── Load checkpoint ────────────────────────────────────────────────────
        ckpt = torch.load(encoder_path, map_location="cpu", weights_only=False)

        input_dim  = int(ckpt.get("input_dim",   512))
        feature_sz = int(ckpt.get("feature_size", 256))

        # ── Reconstruct encoder architecture ──────────────────────────────────
        from adapters.dapn_observation_encoder import DAPNObservationEncoder
        self._encoder = DAPNObservationEncoder(
            input_dim=input_dim,
            feature_size=feature_sz,
            use_bottleneck=True,
            bottleneck_dim=feature_sz,
            device=self.device,
        )
        self._encoder.load_state_dict(ckpt["shared_encoder_state_dict"])
        self._encoder.to(self.device)
        self._encoder.eval()
        for p in self._encoder.parameters():
            p.requires_grad_(False)   # frozen

        # ── Normalisation stats ────────────────────────────────────────────────
        self._norm_mean = torch.tensor(ckpt["norm_mean"], dtype=torch.float32,
                                       device=self.device)
        self._norm_std  = torch.tensor(
            np.clip(ckpt["norm_std"], 1e-8, None), dtype=torch.float32,
            device=self.device
        )
        self._clip_z = float(ckpt.get("clip_z", 5.0))

        # ── Override observation space ─────────────────────────────────────────
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(feature_sz,), dtype=np.float32
        )
        # Action space stays as-is (Discrete(9))

    # ── Gym interface ──────────────────────────────────────────────────────────

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._encode(obs), info

    def step(self, action):
        obs, r, terminated, truncated, info = self.env.step(action)
        return self._encode(obs), r, terminated, truncated, info

    # ── Encoding ───────────────────────────────────────────────────────────────

    def _encode(self, obs_512: np.ndarray) -> np.ndarray:
        """Normalise → clip → run through frozen encoder → return 256-D numpy."""
        import torch
        x = torch.tensor(obs_512, dtype=torch.float32, device=self.device).unsqueeze(0)

        # Standardise
        x = (x - self._norm_mean) / self._norm_std

        # Clip to prevent exploding inputs
        x = torch.clamp(x, -self._clip_z, self._clip_z)

        with torch.no_grad():
            z = self._encoder(x)           # returns 256-D tensor
            if isinstance(z, (tuple, list)):
                z = z[0]                   # encoder returns (features, ...) if return_all

        return z.squeeze(0).cpu().numpy().astype(np.float32)
