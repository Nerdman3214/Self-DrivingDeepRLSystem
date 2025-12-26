"""
MLP Policy Networks for State-Based RL

For environments with vector observations (not images).
Suitable for lane-keeping, vehicle control, etc.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
import numpy as np


class MLPFeatureExtractor(nn.Module):
    """
    Multi-Layer Perceptron feature extractor for vector observations.
    
    Architecture:
        Input → FC(64) → ReLU → FC(64) → ReLU → Output
    """
    
    def __init__(self, input_dim: int, hidden_dims: Tuple[int, ...] = (64, 64)):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        self.network = nn.Sequential(*layers)
        self.output_dim = prev_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, input_dim]
        
        Returns:
            features: [batch_size, output_dim]
        """
        return self.network(x)


class MLPActorCritic(nn.Module):
    """
    Actor-Critic network for continuous control with vector observations.
    
    Used for:
        - Lane keeping
        - Vehicle control
        - Robot manipulation
        - Any state-based RL task
    
    Architecture:
        Input → Shared MLP → Actor Head (policy)
                           → Critic Head (value)
    """
    
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dims: Tuple[int, ...] = (256, 256),
        activation: str = 'relu',
        log_std_init: float = 0.0,
    ):
        """
        Args:
            observation_dim: Dimension of observation vector
            action_dim: Dimension of action vector
            hidden_dims: Sizes of hidden layers
            activation: Activation function ('relu', 'tanh')
            log_std_init: Initial value for log standard deviation
        """
        super().__init__()
        
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        
        # Shared feature extractor
        self.features = MLPFeatureExtractor(observation_dim, hidden_dims)
        feature_dim = self.features.output_dim
        
        # Actor head (policy): outputs mean of action distribution
        self.actor_mean = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh()  # Actions in [-1, 1]
        )
        
        # Learnable log standard deviation (state-independent)
        self.actor_log_std = nn.Parameter(
            torch.ones(action_dim) * log_std_init
        )
        
        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Orthogonal initialization for better training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        
        # Last layer of actor and critic with smaller gain
        nn.init.orthogonal_(self.actor_mean[-2].weight, gain=0.01)
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through actor-critic.
        
        Args:
            obs: [batch_size, observation_dim]
        
        Returns:
            action_mean: [batch_size, action_dim]
            value: [batch_size, 1]
        """
        features = self.features(obs)
        action_mean = self.actor_mean(features)
        value = self.critic(features)
        return action_mean, value
    
    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action, log probability, entropy, and value.
        
        Args:
            obs: [batch_size, observation_dim]
            action: Optional pre-selected action for log prob calculation
            deterministic: If True, return mean action (no noise)
        
        Returns:
            action: [batch_size, action_dim]
            log_prob: [batch_size]
            entropy: [batch_size]
            value: [batch_size, 1]
        """
        action_mean, value = self.forward(obs)
        action_std = torch.exp(self.actor_log_std)
        
        # Create Gaussian distribution
        dist = torch.distributions.Normal(action_mean, action_std)
        
        if action is None:
            if deterministic:
                action = action_mean
            else:
                action = dist.sample()
        
        # Clip action to [-1, 1]
        action = torch.clamp(action, -1.0, 1.0)
        
        # Compute log probability
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        # Compute entropy
        entropy = dist.entropy().sum(dim=-1)
        
        return action, log_prob, entropy, value
    
    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Get value estimate only.
        
        Args:
            obs: [batch_size, observation_dim]
        
        Returns:
            value: [batch_size, 1]
        """
        features = self.features(obs)
        return self.critic(features)
    
    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions (for PPO update).
        
        Args:
            obs: [batch_size, observation_dim]
            actions: [batch_size, action_dim]
        
        Returns:
            values: [batch_size, 1]
            log_probs: [batch_size]
            entropies: [batch_size]
        """
        action_mean, value = self.forward(obs)
        action_std = torch.exp(self.actor_log_std)
        
        dist = torch.distributions.Normal(action_mean, action_std)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        return value, log_prob, entropy


class MLPActorCriticForExport(nn.Module):
    """
    Simplified version for ONNX export (actor only).
    """
    
    def __init__(self, policy: MLPActorCritic):
        super().__init__()
        self.features = policy.features
        self.actor_mean = policy.actor_mean
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass (deterministic policy for inference).
        
        Args:
            obs: [batch_size, observation_dim]
        
        Returns:
            action: [batch_size, action_dim]
        """
        features = self.features(obs)
        action = self.actor_mean(features)
        return action
