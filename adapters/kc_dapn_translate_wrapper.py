"""
KCDAPNTranslateWrapper — domain-translation wrapper for zero-shot transfer.

Instead of training a separate policy on DAPN-encoded 256-D features (which is
slow due to the frozen-encoder bottleneck), this wrapper uses the DAPN
encoder+decoder pair to *translate* CBS observations into the CW observation
space.  The raw 512-D CW policy can then be applied unchanged.

Pipeline:
  CBS obs (512-D)
       ↓  normalise + clip
  DAPN encoder  (512 → 256-D  domain-invariant latent)
       ↓
  DAPN decoder  (256 → 512-D  reconstructed in CW obs space)
       ↓
  raw CW policy  (Discrete(9) kill-chain intent)

The decoder was trained with a reconstruction loss, so the 256-D latent
"decoded back" from a CBS obs should fall in the same 512-D region as a
CW obs at the same kill-chain stage, bridging the domain gap.

Usage:
    env    = UnifiedSecEnv("cbs", cbs_factory=make_cbs_env)
    env    = KCDAPNTranslateWrapper(env, encoder_path="artifacts/transfer_models/dapn_encoder.pt")
    policy = PPO.load("artifacts/policies/best_kc_raw/best_model.zip")
    obs, _ = env.reset()
    action, _ = policy.predict(obs, deterministic=True)   # 512-D → raw policy
"""

import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from gymnasium import spaces
from typing import Optional


TRANSLATED_DIM = 512   # decoder output == raw obs dimension


class KCDAPNTranslateWrapper(gym.Wrapper):
    """
    CBS obs (512-D) → DAPN encoder (256-D) → DAPN decoder (512-D) → raw CW policy.
    Observation space is 512-D (same as the raw CW policy expects).
    Action space Discrete(9) is passed through unchanged.
    """

    def __init__(self, env: gym.Env, encoder_path: str,
                 device: Optional[str] = None):
        super().__init__(env)

        self.device = torch.device(
            device if device else ("mps" if torch.backends.mps.is_available() else "cpu")
        )

        # ── Load checkpoint ────────────────────────────────────────────────────
        ckpt = torch.load(encoder_path, map_location="cpu", weights_only=False)

        input_dim  = int(ckpt.get("input_dim",   512))
        feature_sz = int(ckpt.get("feature_size", 256))

        # ── Reconstruct encoder+decoder ────────────────────────────────────────
        from adapters.dapn_observation_encoder import DAPNObservationEncoder
        full_model = DAPNObservationEncoder(
            input_dim=input_dim,
            feature_size=feature_sz,
            use_bottleneck=True,
            bottleneck_dim=feature_sz,
            device=self.device,
        )
        full_model.load_state_dict(ckpt["shared_encoder_state_dict"])
        # Restore trained CW-specific decoder if checkpoint contains it
        if "decoder_cw_state_dict" in ckpt and hasattr(full_model, "decoder_cw"):
            full_model.decoder_cw.load_state_dict(ckpt["decoder_cw_state_dict"])
        full_model.to(self.device).eval()
        for p in full_model.parameters():
            p.requires_grad_(False)

        # Expose encoder (512→256) and decoder (256→512) separately
        # The DAPNObservationEncoder forward() returns the bottleneck output.
        # We access the internal decoder submodule directly.
        self._full_model = full_model

        # ── Normalisation stats (from training data statistics) ────────────────
        self._norm_mean = torch.tensor(ckpt["norm_mean"], dtype=torch.float32,
                                       device=self.device)
        self._norm_std  = torch.tensor(
            np.clip(ckpt["norm_std"], 1e-8, None), dtype=torch.float32,
            device=self.device
        )
        self._clip_z = float(ckpt.get("clip_z", 5.0))

        # ── Override observation space → 512-D (matches raw CW policy) ─────────
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(TRANSLATED_DIM,), dtype=np.float32
        )

    # ── Gym interface ──────────────────────────────────────────────────────────

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._translate(obs), info

    def step(self, action):
        obs, r, terminated, truncated, info = self.env.step(action)
        return self._translate(obs), r, terminated, truncated, info

    # ── Domain translation ─────────────────────────────────────────────────────

    def _translate(self, obs_512: np.ndarray) -> np.ndarray:
        """CBS 512-D obs → DAPN encode (256-D) → decode (512-D CW-like)."""
        x = torch.tensor(obs_512, dtype=torch.float32, device=self.device).unsqueeze(0)

        # Normalise using training statistics
        x = (x - self._norm_mean) / self._norm_std
        x = torch.clamp(x, -self._clip_z, self._clip_z)

        with torch.no_grad():
            # Encode: 512 → 128 → 256 (feature_layers) → 256 (bottleneck)
            z = self._full_model(x)           # returns bottleneck 256-D
            if isinstance(z, (tuple, list)):
                z = z[0]

            # Decode: 256 → 512 using CW-specific decoder if available.
            # decoder_cw was trained ONLY on CW observations, so its output is in
            # CW observation space — the raw CW policy can interpret it correctly.
            # Falls back to the shared decoder for backward compatibility.
            if hasattr(self._full_model, "decoder_cw"):
                decoded = self._full_model.decoder_cw(z)  # CW-space output
            else:
                decoded = self._full_model.decoder(z)     # legacy fallback

        return decoded.squeeze(0).cpu().numpy().astype(np.float32)
