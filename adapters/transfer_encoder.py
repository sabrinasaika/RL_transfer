"""
Observation Transfer Encoder and Dynamics Model
Based on transfer_across_obs approach for model-based regularization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


class L2Norm(nn.Module):
    """L2 normalization layer for feature vectors"""
    def forward(self, x):
        return x / (x.norm(p=2, dim=-1, keepdim=True) + 1e-8)


class ObservationEncoder(nn.Module):
    """
    Encodes observations from both Cyberwheel and CBS into a shared feature space.
    Input: Unified observation vector (OBS_DIM = 8)
    Output: L2-normalized feature vector (feature_size dimensions)
    """
    def __init__(self, input_dim: int = 8, feature_size: int = 64, hidden_size: int = 128):
        super().__init__()
        self.input_dim = input_dim
        self.feature_size = feature_size
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, feature_size),
            L2Norm()  # Normalize features
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: Observation tensor of shape (batch_size, input_dim) or (input_dim,)
        Returns:
            features: Normalized feature vector of shape (batch_size, feature_size) or (feature_size,)
        """
        was_1d = obs.dim() == 1
        if was_1d:
            obs = obs.unsqueeze(0)
        features = self.encoder(obs)
        if was_1d:
            features = features.squeeze(0)
        return features


class DynamicsModel(nn.Module):
    """
    Predicts next-state features and rewards in the shared feature space.
    Used for model-based regularization during transfer learning.
    """
    def __init__(self, feature_size: int = 64, num_actions: int = 7, hidden_size: int = 64):
        super().__init__()
        self.feature_size = feature_size
        self.num_actions = num_actions
        
        # Predict next state features for each action
        # Output: [batch, feature_size * num_actions]
        self.transition_net = nn.Sequential(
            nn.Linear(feature_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, feature_size * num_actions),
        )
        
        # Predict reward for each action
        # Output: [batch, num_actions]
        self.reward_net = nn.Sequential(
            nn.Linear(feature_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_actions),
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, state_features: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predicts next state features and rewards given current state features and actions.
        
        Args:
            state_features: Current state features [batch, feature_size]
            actions: Action indices [batch] or [batch, 1] (discrete actions)
        
        Returns:
            next_features: Predicted next state features [batch, feature_size]
            predicted_rewards: Predicted rewards [batch]
        """
        batch_size = state_features.size(0)
        
        # Ensure actions are 1D
        if actions.dim() > 1:
            actions = actions.squeeze(-1)
        actions = actions.long()
        
        # Predict next features for all actions
        next_features_all = self.transition_net(state_features)  # [batch, feature_size * num_actions]
        # Reshape to [batch, num_actions, feature_size]
        next_features_all = next_features_all.view(batch_size, self.num_actions, self.feature_size)
        
        # Select features for chosen actions
        # actions: [batch] -> [batch, 1] -> [batch, 1, feature_size]
        action_indices = actions.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, self.feature_size)
        next_features = next_features_all.gather(1, action_indices).squeeze(1)  # [batch, feature_size]
        
        # Predict rewards for all actions
        rewards_all = self.reward_net(state_features)  # [batch, num_actions]
        # Select reward for chosen action
        predicted_rewards = rewards_all.gather(1, actions.unsqueeze(-1)).squeeze(1)  # [batch]
        
        return next_features, predicted_rewards
    
    def compute_loss(self, state_features: torch.Tensor, next_state_features: torch.Tensor,
                     actions: torch.Tensor, rewards: torch.Tensor) -> torch.Tensor:
        """
        Compute dynamics model loss (MSE for both next state and reward prediction).
        
        Args:
            state_features: Current state features [batch, feature_size]
            next_state_features: Actual next state features [batch, feature_size]
            actions: Action indices [batch]
            rewards: Actual rewards [batch]
        
        Returns:
            loss: Combined MSE loss
        """
        pred_next_features, pred_rewards = self.forward(state_features, actions)
        
        # MSE loss for next state prediction
        next_state_loss = F.mse_loss(pred_next_features, next_state_features)
        
        # MSE loss for reward prediction
        reward_loss = F.mse_loss(pred_rewards, rewards.float())
        
        return next_state_loss + reward_loss


def load_transfer_models(checkpoint_path: str, device: torch.device = None) -> Tuple[ObservationEncoder, DynamicsModel]:
    """
    Load encoder and dynamics model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load models on
    
    Returns:
        encoder: Loaded ObservationEncoder
        dynamics_model: Loaded DynamicsModel
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get dimensions from checkpoint or use defaults
    input_dim = checkpoint.get('input_dim', 8)
    feature_size = checkpoint.get('feature_size', 64)
    num_actions = checkpoint.get('num_actions', 7)
    
    encoder = ObservationEncoder(input_dim=input_dim, feature_size=feature_size)
    dynamics_model = DynamicsModel(feature_size=feature_size, num_actions=num_actions)
    
    encoder.load_state_dict(checkpoint['encoder'])
    dynamics_model.load_state_dict(checkpoint['dynamics_model'])
    
    encoder.to(device)
    dynamics_model.to(device)
    
    return encoder, dynamics_model


def save_transfer_models(encoder: ObservationEncoder, dynamics_model: DynamicsModel,
                        checkpoint_path: str, input_dim: int = 8, feature_size: int = 64,
                        num_actions: int = 7):
    """
    Save encoder and dynamics model to checkpoint.
    
    Args:
        encoder: ObservationEncoder to save
        dynamics_model: DynamicsModel to save
        checkpoint_path: Path to save checkpoint
        input_dim: Input dimension (for loading later)
        feature_size: Feature dimension (for loading later)
        num_actions: Number of actions (for loading later)
    """
    checkpoint = {
        'encoder': encoder.state_dict(),
        'dynamics_model': dynamics_model.state_dict(),
        'input_dim': input_dim,
        'feature_size': feature_size,
        'num_actions': num_actions,
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved transfer models to {checkpoint_path}")

