"""
Vision-Based Policy Networks

CNN encoder for image-based observations in autonomous driving.
Compatible with existing PPO training infrastructure.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
import numpy as np


class CNNFeatureExtractor(nn.Module):
    """
    Convolutional Neural Network for visual feature extraction.
    
    Architecture (inspired by Nature DQN):
        Input (84x84x1 grayscale or 84x84x3 RGB)
        → Conv1 (32 filters, 8x8, stride 4) → ReLU
        → Conv2 (64 filters, 4x4, stride 2) → ReLU
        → Conv3 (64 filters, 3x3, stride 1) → ReLU
        → Flatten
        → FC (512) → ReLU
        → Output features
    
    Suitable for:
        - PyBullet camera images
        - Simulated sensor data
        - First-person driving views
    """
    
    def __init__(
        self,
        input_channels: int = 1,
        image_size: int = 84,
        hidden_dim: int = 512
    ):
        """
        Args:
            input_channels: Number of image channels (1=grayscale, 3=RGB)
            image_size: Height/width of square input image
            hidden_dim: Dimension of output features
        """
        super().__init__()
        
        self.input_channels = input_channels
        self.image_size = image_size
        
        # Convolutional layers
        self.conv = nn.Sequential(
            # Layer 1: 84x84 → 20x20
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            
            # Layer 2: 20x20 → 9x9
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            
            # Layer 3: 9x9 → 7x7
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        # Calculate flattened size
        conv_output_size = self._get_conv_output_size(input_channels, image_size)
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(conv_output_size, hidden_dim),
            nn.ReLU()
        )
        
        self.output_dim = hidden_dim
        
        # Initialize weights
        self._initialize_weights()
    
    def _get_conv_output_size(self, channels: int, size: int) -> int:
        """Calculate output size after convolutions"""
        with torch.no_grad():
            dummy_input = torch.zeros(1, channels, size, size)
            dummy_output = self.conv(dummy_input)
            return int(np.prod(dummy_output.shape[1:]))
    
    def _initialize_weights(self):
        """Orthogonal initialization for better training"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, channels, height, width]
        
        Returns:
            features: [batch_size, hidden_dim]
        """
        # Normalize pixel values to [0, 1]
        x = x.float() / 255.0
        
        # Convolutional features
        conv_out = self.conv(x)
        
        # Flatten
        flat = conv_out.reshape(conv_out.size(0), -1)
        
        # Fully connected
        features = self.fc(flat)
        
        return features


class VisionActorCritic(nn.Module):
    """
    Actor-Critic network for visual observations.
    
    Architecture:
        Image → CNN → Actor Head (policy)
                   → Critic Head (value)
    
    Compatible with PPO, A2C, and other actor-critic algorithms.
    """
    
    def __init__(
        self,
        input_channels: int,
        action_dim: int,
        image_size: int = 84,
        hidden_dim: int = 512,
        log_std_init: float = 0.0
    ):
        """
        Args:
            input_channels: Number of image channels (1 or 3)
            action_dim: Dimension of continuous action space
            image_size: Height/width of input images
            hidden_dim: Hidden layer size
            log_std_init: Initial log standard deviation for actions
        """
        super().__init__()
        
        self.input_channels = input_channels
        self.action_dim = action_dim
        
        # CNN feature extractor
        self.features = CNNFeatureExtractor(
            input_channels=input_channels,
            image_size=image_size,
            hidden_dim=hidden_dim
        )
        
        # Actor head (policy)
        self.actor_mean = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()  # Actions in [-1, 1]
        )
        
        # Learnable log standard deviation
        self.actor_log_std = nn.Parameter(
            torch.ones(action_dim) * log_std_init
        )
        
        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        # Initialize heads
        self._initialize_heads()
    
    def _initialize_heads(self):
        """Initialize actor and critic with small weights"""
        nn.init.orthogonal_(self.actor_mean[-2].weight, gain=0.01)
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through actor-critic.
        
        Args:
            obs: [batch_size, channels, height, width]
        
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
        Sample action and compute log probability, entropy, value.
        
        Args:
            obs: [batch_size, channels, height, width]
            action: Optional pre-selected action
            deterministic: If True, return mean (no noise)
        
        Returns:
            action: [batch_size, action_dim]
            log_prob: [batch_size]
            entropy: [batch_size]
            value: [batch_size, 1]
        """
        action_mean, value = self.forward(obs)
        action_std = torch.exp(self.actor_log_std)
        
        dist = torch.distributions.Normal(action_mean, action_std)
        
        if action is None:
            if deterministic:
                action = action_mean
            else:
                action = dist.sample()
        
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        return action, log_prob, entropy, value
    
    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for PPO update.
        
        Args:
            obs: [batch_size, channels, height, width]
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


class HybridActorCritic(nn.Module):
    """
    Hybrid Actor-Critic that combines vision AND state observations.
    
    Architecture:
        Image → CNN → \
                       → Concat → Actor/Critic
        State → MLP → /
    
    Best of both worlds:
        - Vision: Spatial understanding
        - State: Precise numerical data (speed, distances)
    
    This is how real autonomous systems work (camera + sensors).
    """
    
    def __init__(
        self,
        input_channels: int,
        state_dim: int,
        action_dim: int,
        image_size: int = 84,
        vision_hidden: int = 256,
        state_hidden: int = 128,
        log_std_init: float = 0.0
    ):
        """
        Args:
            input_channels: Image channels
            state_dim: State vector dimension
            action_dim: Action vector dimension
            image_size: Image height/width
            vision_hidden: Vision features dimension
            state_hidden: State features dimension
            log_std_init: Initial log std for actions
        """
        super().__init__()
        
        # Vision encoder
        self.vision_encoder = CNNFeatureExtractor(
            input_channels=input_channels,
            image_size=image_size,
            hidden_dim=vision_hidden
        )
        
        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, state_hidden),
            nn.ReLU(),
            nn.Linear(state_hidden, state_hidden),
            nn.ReLU()
        )
        
        # Combined feature dimension
        combined_dim = vision_hidden + state_hidden
        
        # Actor head
        self.actor_mean = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )
        
        self.actor_log_std = nn.Parameter(
            torch.ones(action_dim) * log_std_init
        )
        
        # Critic head
        self.critic = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    
    def forward(
        self,
        image: torch.Tensor,
        state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with both vision and state.
        
        Args:
            image: [batch_size, channels, height, width]
            state: [batch_size, state_dim]
        
        Returns:
            action_mean: [batch_size, action_dim]
            value: [batch_size, 1]
        """
        # Extract features
        vision_features = self.vision_encoder(image)
        state_features = self.state_encoder(state)
        
        # Concatenate
        combined = torch.cat([vision_features, state_features], dim=-1)
        
        # Actor and critic
        action_mean = self.actor_mean(combined)
        value = self.critic(combined)
        
        return action_mean, value
    
    def get_action_and_value(
        self,
        image: torch.Tensor,
        state: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get action with hybrid observations"""
        action_mean, value = self.forward(image, state)
        action_std = torch.exp(self.actor_log_std)
        
        dist = torch.distributions.Normal(action_mean, action_std)
        
        if action is None:
            if deterministic:
                action = action_mean
            else:
                action = dist.sample()
        
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        return action, log_prob, entropy, value
