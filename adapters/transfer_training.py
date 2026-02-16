"""
Training utilities for observation transfer learning.
Includes functions for training dynamics model and using it for regularization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional
from collections import deque, namedtuple

from adapters.transfer_encoder import ObservationEncoder, DynamicsModel, save_transfer_models


Transition = namedtuple('Transition', ('obs', 'action', 'next_obs', 'reward', 'done'))


class ReplayBuffer:
    """Simple replay buffer for storing transitions"""
    def __init__(self, capacity: int = 10000):
        self.memory = deque([], maxlen=capacity)
    
    def push(self, obs, action, next_obs, reward, done):
        self.memory.append(Transition(obs, action, next_obs, reward, done))
    
    def sample(self, batch_size: int):
        indices = np.random.choice(len(self.memory), batch_size, replace=False)
        return [self.memory[i] for i in indices]
    
    def __len__(self):
        return len(self.memory)


def train_dynamics_model(
    encoder: ObservationEncoder,
    dynamics_model: DynamicsModel,
    replay_buffer: ReplayBuffer,
    batch_size: int = 128,
    num_epochs: int = 10,
    learning_rate: float = 1e-3,
    device: torch.device = None
) -> List[float]:
    """
    Train dynamics model on collected transitions.
    
    Args:
        encoder: ObservationEncoder (frozen or trainable)
        dynamics_model: DynamicsModel to train
        replay_buffer: Buffer with transitions
        batch_size: Batch size for training
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        device: Device to train on
    
    Returns:
        losses: List of losses per epoch
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if len(replay_buffer) < batch_size:
        return []
    
    dynamics_model.train()
    if encoder.training:
        encoder.eval()  # Freeze encoder during dynamics training
    
    optimizer = torch.optim.Adam(dynamics_model.parameters(), lr=learning_rate)
    losses = []
    
    for epoch in range(num_epochs):
        batch = replay_buffer.sample(batch_size)
        
        # Prepare batch
        obs_batch = torch.FloatTensor(np.array([t.obs for t in batch])).to(device)
        next_obs_batch = torch.FloatTensor(np.array([t.next_obs for t in batch])).to(device)
        action_batch = torch.LongTensor([t.action for t in batch]).to(device)
        reward_batch = torch.FloatTensor([t.reward for t in batch]).to(device)
        
        # Encode observations
        with torch.no_grad():
            state_features = encoder(obs_batch)
            next_state_features = encoder(next_obs_batch)
        
        # Compute loss
        loss = dynamics_model.compute_loss(
            state_features, next_state_features, action_batch, reward_batch
        )
        
        # Optimize
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(dynamics_model.parameters(), 1.0)
        optimizer.step()
        
        losses.append(loss.item())
    
    return losses


def compute_regularization_loss(
    encoder: ObservationEncoder,
    dynamics_model: DynamicsModel,
    obs: np.ndarray,
    next_obs: np.ndarray,
    action: int,
    device: torch.device = None
) -> torch.Tensor:
    """
    Compute regularization loss using dynamics model.
    Used during target task training to preserve source task knowledge.
    
    Args:
        encoder: ObservationEncoder
        dynamics_model: Pre-trained DynamicsModel
        obs: Current observation
        next_obs: Next observation
        action: Action taken
        device: Device to compute on
    
    Returns:
        reg_loss: Regularization loss tensor
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    encoder.eval()
    dynamics_model.eval()
    
    with torch.no_grad():
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
        next_obs_tensor = torch.FloatTensor(next_obs).unsqueeze(0).to(device)
        action_tensor = torch.LongTensor([action]).to(device)
        
        # Encode observations
        state_features = encoder(obs_tensor)
        next_state_features = encoder(next_obs_tensor)
        
        # Predict next features
        pred_next_features, _ = dynamics_model(state_features, action_tensor)
        
        # Regularization loss: predicted features should match actual
        reg_loss = F.mse_loss(pred_next_features, next_state_features)
    
    return reg_loss


def collect_transitions_for_dynamics_training(
    env,
    policy,
    num_steps: int = 1000,
    device: torch.device = None
) -> ReplayBuffer:
    """
    Collect transitions from environment using policy for dynamics model training.
    
    Args:
        env: Environment to collect from
        policy: Policy to use for action selection
        num_steps: Number of steps to collect
        device: Device for policy
    
    Returns:
        replay_buffer: Buffer with collected transitions
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    replay_buffer = ReplayBuffer()
    obs, _ = env.reset()
    
    for step in range(num_steps):
        # Get action from policy
        if hasattr(policy, 'predict'):
            action, _ = policy.predict(obs, deterministic=False)
        elif hasattr(policy, '__call__'):
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            with torch.no_grad():
                action_logits = policy(obs_tensor)
                action = torch.argmax(action_logits, dim=-1).item()
        else:
            action = env.action_space.sample()
        
        # Step environment
        next_obs, reward, done, truncated, info = env.step(action)
        
        # Store transition
        replay_buffer.push(obs, action, next_obs, reward, done or truncated)
        
        if done or truncated:
            obs, _ = env.reset()
        else:
            obs = next_obs
    
    return replay_buffer

