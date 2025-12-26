"""
Policy Network implementations for different action space types.

Provides:
- GaussianPolicy: For continuous action spaces (e.g., steering, throttle)
- CategoricalPolicy: For discrete action spaces (e.g., turn left/right/straight)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import numpy as np


class GaussianPolicy(nn.Module):
    """
    Gaussian policy for continuous action spaces.
    
    Outputs mean and log_std of a Gaussian distribution from which
    actions are sampled. Supports both state-dependent and fixed std.
    """
    
    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        hidden_dims: Tuple[int, ...] = (256, 256),
        log_std_init: float = 0.0,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
        state_dependent_std: bool = False
    ):
        super().__init__()
        
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.state_dependent_std = state_dependent_std
        
        # Build MLP for mean
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        
        self.shared_net = nn.Sequential(*layers)
        
        # Mean head
        self.mean_head = nn.Linear(prev_dim, action_dim)
        
        # Standard deviation
        if state_dependent_std:
            self.log_std_head = nn.Linear(prev_dim, action_dim)
        else:
            self.log_std = nn.Parameter(torch.ones(action_dim) * log_std_init)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for stable training."""
        for module in self.shared_net.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.zeros_(module.bias)
        
        # Small initialization for output heads
        nn.init.orthogonal_(self.mean_head.weight, gain=0.01)
        nn.init.zeros_(self.mean_head.bias)
        
        if self.state_dependent_std:
            nn.init.orthogonal_(self.log_std_head.weight, gain=0.01)
            nn.init.zeros_(self.log_std_head.bias)
    
    def forward(
        self,
        features: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the Gaussian policy.
        
        Args:
            features: Feature tensor from encoder (batch, feature_dim)
            deterministic: If True, return mean action without sampling
        
        Returns:
            actions: Sampled or deterministic actions (batch, action_dim)
            log_probs: Log probability of actions (batch,)
            entropy: Policy entropy (batch,)
        """
        hidden = self.shared_net(features)
        mean = self.mean_head(hidden)
        
        if self.state_dependent_std:
            log_std = self.log_std_head(hidden)
        else:
            log_std = self.log_std.expand_as(mean)
        
        # Clamp for numerical stability
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        
        # Create distribution
        dist = torch.distributions.Normal(mean, std)
        
        if deterministic:
            actions = mean
        else:
            actions = dist.rsample()
        
        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        return actions, log_probs, entropy
    
    def evaluate(
        self,
        features: torch.Tensor,
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate log probability and entropy of given actions.
        
        Args:
            features: Feature tensor (batch, feature_dim)
            actions: Actions to evaluate (batch, action_dim)
        
        Returns:
            log_probs: Log probability of actions (batch,)
            entropy: Policy entropy (batch,)
        """
        hidden = self.shared_net(features)
        mean = self.mean_head(hidden)
        
        if self.state_dependent_std:
            log_std = self.log_std_head(hidden)
        else:
            log_std = self.log_std.expand_as(mean)
        
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        
        dist = torch.distributions.Normal(mean, std)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        return log_probs, entropy
    
    def get_mean(self, features: torch.Tensor) -> torch.Tensor:
        """Get deterministic mean action (for export)."""
        hidden = self.shared_net(features)
        return self.mean_head(hidden)


class CategoricalPolicy(nn.Module):
    """
    Categorical policy for discrete action spaces.
    
    Outputs logits for a categorical distribution over actions.
    """
    
    def __init__(
        self,
        input_dim: int,
        num_actions: int,
        hidden_dims: Tuple[int, ...] = (256, 256)
    ):
        super().__init__()
        
        self.num_actions = num_actions
        
        # Build MLP
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_actions))
        self.net = nn.Sequential(*layers)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for module in self.net.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.zeros_(module.bias)
        
        # Small initialization for output layer
        final_layer = list(self.net.modules())[-1]
        if isinstance(final_layer, nn.Linear):
            nn.init.orthogonal_(final_layer.weight, gain=0.01)
    
    def forward(
        self,
        features: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the categorical policy.
        
        Args:
            features: Feature tensor (batch, feature_dim)
            deterministic: If True, return argmax action
        
        Returns:
            actions: Sampled or deterministic actions (batch,)
            log_probs: Log probability of actions (batch,)
            entropy: Policy entropy (batch,)
        """
        logits = self.net(features)
        dist = torch.distributions.Categorical(logits=logits)
        
        if deterministic:
            actions = torch.argmax(logits, dim=-1)
        else:
            actions = dist.sample()
        
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return actions, log_probs, entropy
    
    def evaluate(
        self,
        features: torch.Tensor,
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate log probability and entropy of given actions.
        
        Args:
            features: Feature tensor (batch, feature_dim)
            actions: Actions to evaluate (batch,)
        
        Returns:
            log_probs: Log probability of actions (batch,)
            entropy: Policy entropy (batch,)
        """
        logits = self.net(features)
        dist = torch.distributions.Categorical(logits=logits)
        
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return log_probs, entropy
    
    def get_action_probs(self, features: torch.Tensor) -> torch.Tensor:
        """Get action probabilities (for visualization)."""
        logits = self.net(features)
        return F.softmax(logits, dim=-1)


class SquashedGaussianPolicy(nn.Module):
    """
    Squashed Gaussian policy for bounded continuous action spaces.
    
    Uses tanh squashing to ensure actions are within [-1, 1], with
    proper log probability correction. Commonly used in SAC.
    """
    
    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        hidden_dims: Tuple[int, ...] = (256, 256),
        log_std_min: float = -20.0,
        log_std_max: float = 2.0
    ):
        super().__init__()
        
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        # Build MLP
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        
        self.shared_net = nn.Sequential(*layers)
        self.mean_head = nn.Linear(prev_dim, action_dim)
        self.log_std_head = nn.Linear(prev_dim, action_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.zeros_(module.bias)
        
        # Smaller initialization for output heads
        nn.init.orthogonal_(self.mean_head.weight, gain=0.01)
        nn.init.orthogonal_(self.log_std_head.weight, gain=0.01)
    
    def forward(
        self,
        features: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with tanh squashing.
        
        Args:
            features: Feature tensor (batch, feature_dim)
            deterministic: If True, return deterministic action
        
        Returns:
            actions: Squashed actions in [-1, 1] (batch, action_dim)
            log_probs: Corrected log probabilities (batch,)
            entropy: Approximate entropy (batch,)
        """
        hidden = self.shared_net(features)
        mean = self.mean_head(hidden)
        log_std = self.log_std_head(hidden)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        
        dist = torch.distributions.Normal(mean, std)
        
        if deterministic:
            pre_tanh = mean
        else:
            pre_tanh = dist.rsample()
        
        # Squash with tanh
        actions = torch.tanh(pre_tanh)
        
        # Log probability with tanh correction
        log_probs = dist.log_prob(pre_tanh).sum(dim=-1)
        # Correction for tanh squashing
        log_probs -= torch.log(1 - actions.pow(2) + 1e-6).sum(dim=-1)
        
        # Approximate entropy (actual entropy of squashed Gaussian is complex)
        entropy = dist.entropy().sum(dim=-1)
        
        return actions, log_probs, entropy
    
    def evaluate(
        self,
        features: torch.Tensor,
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate squashed actions.
        
        Note: This requires the pre-tanh actions for exact computation.
        For PPO, use the standard GaussianPolicy instead.
        """
        hidden = self.shared_net(features)
        mean = self.mean_head(hidden)
        log_std = self.log_std_head(hidden)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        
        # Inverse tanh to get pre-tanh actions
        # Clamp to avoid numerical issues at boundaries
        actions_clamped = torch.clamp(actions, -0.999, 0.999)
        pre_tanh = 0.5 * (torch.log(1 + actions_clamped) - torch.log(1 - actions_clamped))
        
        dist = torch.distributions.Normal(mean, std)
        log_probs = dist.log_prob(pre_tanh).sum(dim=-1)
        log_probs -= torch.log(1 - actions.pow(2) + 1e-6).sum(dim=-1)
        
        entropy = dist.entropy().sum(dim=-1)
        
        return log_probs, entropy
