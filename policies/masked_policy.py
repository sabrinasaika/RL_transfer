from typing import Optional, Tuple, Dict

import torch
from torch import nn

from stable_baselines3.ppo.policies import MultiInputPolicy as PPOMultiInputPolicy
from stable_baselines3.common.distributions import Distribution


class MaskedMultiInputPolicy(PPOMultiInputPolicy):
    """
    Actor-critic policy for Multi-Input observations where obs is a dict:
    { 'obs': float32[N], 'mask': float32[K] }.

    The action mask is applied to the actor logits as log(mask + eps), forcing
    invalid actions to probability ~0 during training and inference.
    """

    def _masked_logits(self, latent_pi: torch.Tensor, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        logits = self.action_net(latent_pi)
        mask: Optional[torch.Tensor] = None
        if isinstance(obs, dict):
            mask = obs.get("mask")
        if mask is not None:
            # Ensure same dtype/device and numeric stability
            mask = mask.to(logits.device).clamp(min=0.0, max=1.0)
            logits = logits + torch.log(mask + 1e-8)
        return logits

    def forward(
        self,
        obs: Dict[str, torch.Tensor],
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Let SB3 handle dict preprocessing (MultiInput)
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        distribution: Distribution = self._get_action_dist_from_latent(latent_pi)

        # Apply mask to logits before sampling
        logits = self._masked_logits(latent_pi, obs)
        distribution.proba_distribution(action_logits=logits)

        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        return actions, values, log_prob

    def _predict(self, observation: Dict[str, torch.Tensor], deterministic: bool = False) -> torch.Tensor:
        # Predict path with masked logits as well (dict passthrough)
        features = self.extract_features(observation)
        latent_pi, _ = self.mlp_extractor(features)
        distribution: Distribution = self._get_action_dist_from_latent(latent_pi)
        logits = self._masked_logits(latent_pi, observation)
        distribution.proba_distribution(action_logits=logits)
        return distribution.get_actions(deterministic=deterministic)


