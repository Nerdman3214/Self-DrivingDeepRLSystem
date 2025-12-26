"""
Actor-Critic Neural Network Architectures for Self-Driving RL

This module implements CNN-based Actor-Critic networks optimized for
visual input from driving simulators like CarRacing-v2.
"""

import torch
import torch.nn as nn
from typing import Tuple
import numpy as np


class CNNFeatureExtractor(nn.Module):
    """
    Convolutional Neural Network for extracting features from visual observations.
    
    Designed for processing RGB images from driving simulators.
    Input: (batch, channels, height, width) - typically (batch, 3, 96, 96) or (batch, 4, 84, 84) for stacked frames
    Output: Flattened feature vector
    """
    
    def __init__(
        self,
        input_channels: int = 3,
        input_height: int = 96,
        input_width: int = 96,
        feature_dim: int = 512
    ):
        super().__init__()
        
        self.input_channels = input_channels
        self.input_height = input_height
        self.input_width = input_width
        self.feature_dim = feature_dim
        
        # CNN layers - Nature DQN style architecture
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
        )
        
        # Calculate the size of CNN output
        self._cnn_output_dim = self._get_conv_output_dim()
        
        # Fully connected layer to desired feature dimension
        self.fc = nn.Sequential(
            nn.Linear(self._cnn_output_dim, feature_dim),
            nn.ReLU()
        )
        
        # Initialize weights
        self._init_weights()
    
    def _get_conv_output_dim(self) -> int:
        """Calculate the output dimension of the CNN layers."""
        with torch.no_grad():
            dummy_input = torch.zeros(1, self.input_channels, self.input_height, self.input_width)
            output = self.conv(dummy_input)
            return int(np.prod(output.shape[1:]))
    
    def _init_weights(self):
        """Initialize network weights using orthogonal initialization."""
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the CNN feature extractor.
        
        Args:
            x: Input tensor of shape (batch, channels, height, width)
               Values should be normalized to [0, 1]
        
        Returns:
            Feature tensor of shape (batch, feature_dim)
        """
        # Ensure input is float and normalized
        if x.dtype != torch.float32:
            x = x.float()
        
        # If input is in [0, 255], normalize to [0, 1]
        if x.max() > 1.0:
            x = x / 255.0
        
        # CNN forward pass
        conv_out = self.conv(x)
        
        # Flatten and pass through FC
        flat = conv_out.view(conv_out.size(0), -1)
        features = self.fc(flat)
        
        return features


class ActorCritic(nn.Module):
    """
    Actor-Critic network for PPO on continuous action spaces.
    
    Combines:
    - Shared CNN feature extractor
    - Actor head: outputs action mean and log_std for Gaussian policy
    - Critic head: outputs state value estimate
    
    Designed for CarRacing-v2 with 3 continuous actions:
    - Steering: [-1, 1]
    - Throttle: [0, 1]
    - Brake: [0, 1]
    """
    
    def __init__(
        self,
        input_channels: int = 3,
        input_height: int = 96,
        input_width: int = 96,
        action_dim: int = 3,
        feature_dim: int = 512,
        log_std_init: float = 0.0,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0
    ):
        super().__init__()
        
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        # Shared feature extractor
        self.feature_extractor = CNNFeatureExtractor(
            input_channels=input_channels,
            input_height=input_height,
            input_width=input_width,
            feature_dim=feature_dim
        )
        
        # Actor head - outputs mean of Gaussian distribution
        self.actor_mean = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()  # Bound actions to [-1, 1] initially
        )
        
        # Learnable log standard deviation
        self.actor_log_std = nn.Parameter(
            torch.ones(action_dim) * log_std_init
        )
        
        # Critic head - outputs value estimate
        self.critic = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        # Initialize actor and critic heads
        self._init_heads()
    
    def _init_heads(self):
        """Initialize actor and critic head weights."""
        # Actor initialization - small weights for stable initial policy
        for module in self.actor_mean.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=0.01)
                nn.init.zeros_(module.bias)
        
        # Critic initialization
        for module in self.critic.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                nn.init.zeros_(module.bias)
    
    def forward(
        self,
        obs: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for training - returns actions, log_probs, entropy, and values.
        
        Args:
            obs: Observation tensor (batch, channels, height, width)
            deterministic: If True, return mean action without sampling
        
        Returns:
            actions: Sampled actions (batch, action_dim)
            log_probs: Log probability of actions (batch,)
            entropy: Policy entropy (batch,)
            values: Value estimates (batch,)
        """
        # Extract features
        features = self.feature_extractor(obs)
        
        # Actor forward
        action_mean = self.actor_mean(features)
        
        # Clamp log_std for numerical stability
        log_std = torch.clamp(self.actor_log_std, self.log_std_min, self.log_std_max)
        action_std = torch.exp(log_std)
        
        # Create distribution
        dist = torch.distributions.Normal(action_mean, action_std)
        
        if deterministic:
            actions = action_mean
        else:
            actions = dist.rsample()  # Reparameterization trick
        
        # Calculate log probability (sum over action dimensions)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        
        # Calculate entropy
        entropy = dist.entropy().sum(dim=-1)
        
        # Critic forward
        values = self.critic(features).squeeze(-1)
        
        return actions, log_probs, entropy, values
    
    def get_action(
        self,
        obs: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get action for inference (no gradient computation).
        
        Args:
            obs: Observation tensor (batch, channels, height, width)
            deterministic: If True, return mean action
        
        Returns:
            actions: Actions to take
            values: Value estimates (useful for GAE)
        """
        with torch.no_grad():
            features = self.feature_extractor(obs)
            action_mean = self.actor_mean(features)
            
            if deterministic:
                actions = action_mean
            else:
                log_std = torch.clamp(self.actor_log_std, self.log_std_min, self.log_std_max)
                action_std = torch.exp(log_std)
                dist = torch.distributions.Normal(action_mean, action_std)
                actions = dist.sample()
            
            values = self.critic(features).squeeze(-1)
        
        return actions, values
    
    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for PPO update (used during learning).
        
        Args:
            obs: Observation tensor
            actions: Actions to evaluate
        
        Returns:
            log_probs: Log probabilities of the given actions
            entropy: Policy entropy
            values: Value estimates
        """
        features = self.feature_extractor(obs)
        
        # Actor
        action_mean = self.actor_mean(features)
        log_std = torch.clamp(self.actor_log_std, self.log_std_min, self.log_std_max)
        action_std = torch.exp(log_std)
        
        dist = torch.distributions.Normal(action_mean, action_std)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        # Critic
        values = self.critic(features).squeeze(-1)
        
        return log_probs, entropy, values
    
    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Get value estimate only (for bootstrapping)."""
        with torch.no_grad():
            features = self.feature_extractor(obs)
            values = self.critic(features).squeeze(-1)
        return values


class ActorCriticForExport(nn.Module):
    """
    Simplified Actor-Critic for ONNX export (inference only).
    
    This version removes training-specific components and outputs
    only the deterministic action, suitable for deployment.
    """
    
    def __init__(self, actor_critic: ActorCritic):
        super().__init__()
        
        # Copy the feature extractor and actor components
        self.feature_extractor = actor_critic.feature_extractor
        self.actor_mean = actor_critic.actor_mean
        
        # Store action bounds for post-processing
        # CarRacing: steering [-1,1], gas [0,1], brake [0,1]
        self.register_buffer('action_low', torch.tensor([-1.0, 0.0, 0.0]))
        self.register_buffer('action_high', torch.tensor([1.0, 1.0, 1.0]))
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for inference - returns deterministic action.
        
        Args:
            obs: Observation tensor (batch, channels, height, width)
                 Should be normalized to [0, 1]
        
        Returns:
            actions: Deterministic actions (batch, action_dim)
                     Properly bounded for CarRacing environment
        """
        features = self.feature_extractor(obs)
        action_mean = self.actor_mean(features)
        
        # Post-process actions for CarRacing
        # actor_mean outputs in [-1, 1] due to Tanh
        # Need to rescale gas and brake to [0, 1]
        actions = action_mean.clone()
        actions[:, 1] = (actions[:, 1] + 1.0) / 2.0  # Gas: [-1,1] -> [0,1]
        actions[:, 2] = (actions[:, 2] + 1.0) / 2.0  # Brake: [-1,1] -> [0,1]
        
        return actions
