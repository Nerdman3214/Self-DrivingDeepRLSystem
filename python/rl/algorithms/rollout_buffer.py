"""
Rollout Buffer for PPO Training

Stores trajectories collected during environment interaction and provides
mini-batches for training. Implements Generalized Advantage Estimation (GAE).
"""

import torch
import numpy as np
from typing import Generator, Tuple, Optional, NamedTuple


class RolloutBufferSamples(NamedTuple):
    """Named tuple for rollout buffer samples."""
    observations: torch.Tensor
    actions: torch.Tensor
    old_values: torch.Tensor
    old_log_probs: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor


class RolloutBuffer:
    """
    Rollout buffer for on-policy algorithms like PPO.
    
    Stores complete trajectories and computes advantages using GAE.
    Provides mini-batch sampling for multiple epochs of updates.
    """
    
    def __init__(
        self,
        buffer_size: int,
        observation_shape: Tuple[int, ...],
        action_dim: int,
        device: torch.device = torch.device("cpu"),
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        n_envs: int = 1
    ):
        """
        Initialize the rollout buffer.
        
        Args:
            buffer_size: Number of steps per environment to store
            observation_shape: Shape of observations (e.g., (3, 96, 96))
            action_dim: Dimension of action space
            device: Device to store tensors on
            gamma: Discount factor for rewards
            gae_lambda: Lambda for GAE computation
            n_envs: Number of parallel environments
        """
        self.buffer_size = buffer_size
        self.observation_shape = observation_shape
        self.action_dim = action_dim
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.n_envs = n_envs
        
        self.pos = 0
        self.full = False
        self.generator_ready = False
        
        # Initialize buffers
        self.reset()
    
    def reset(self):
        """Reset the buffer for a new rollout."""
        # Observations: (buffer_size, n_envs, *obs_shape)
        self.observations = torch.zeros(
            (self.buffer_size, self.n_envs) + self.observation_shape,
            dtype=torch.float32,
            device=self.device
        )
        
        # Actions: (buffer_size, n_envs, action_dim)
        self.actions = torch.zeros(
            (self.buffer_size, self.n_envs, self.action_dim),
            dtype=torch.float32,
            device=self.device
        )
        
        # Rewards: (buffer_size, n_envs)
        self.rewards = torch.zeros(
            (self.buffer_size, self.n_envs),
            dtype=torch.float32,
            device=self.device
        )
        
        # Episode dones: (buffer_size, n_envs)
        self.dones = torch.zeros(
            (self.buffer_size, self.n_envs),
            dtype=torch.float32,
            device=self.device
        )
        
        # Value estimates: (buffer_size, n_envs)
        self.values = torch.zeros(
            (self.buffer_size, self.n_envs),
            dtype=torch.float32,
            device=self.device
        )
        
        # Log probabilities: (buffer_size, n_envs)
        self.log_probs = torch.zeros(
            (self.buffer_size, self.n_envs),
            dtype=torch.float32,
            device=self.device
        )
        
        # Computed after rollout
        self.advantages = torch.zeros(
            (self.buffer_size, self.n_envs),
            dtype=torch.float32,
            device=self.device
        )
        
        self.returns = torch.zeros(
            (self.buffer_size, self.n_envs),
            dtype=torch.float32,
            device=self.device
        )
        
        self.pos = 0
        self.full = False
        self.generator_ready = False
    
    def add(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        value: torch.Tensor,
        log_prob: torch.Tensor
    ):
        """
        Add a step to the buffer.
        
        Args:
            obs: Observation (n_envs, *obs_shape)
            action: Action taken (n_envs, action_dim)
            reward: Reward received (n_envs,)
            done: Episode done flag (n_envs,)
            value: Value estimate (n_envs,)
            log_prob: Log probability of action (n_envs,)
        """
        if self.pos >= self.buffer_size:
            raise RuntimeError("Rollout buffer is full!")
        
        # Convert to tensors if needed
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs)
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action)
        if isinstance(reward, np.ndarray):
            reward = torch.from_numpy(reward)
        if isinstance(done, np.ndarray):
            done = torch.from_numpy(done)
        
        self.observations[self.pos] = obs.to(self.device)
        self.actions[self.pos] = action.to(self.device)
        self.rewards[self.pos] = reward.to(self.device)
        self.dones[self.pos] = done.float().to(self.device)
        self.values[self.pos] = value.to(self.device)
        self.log_probs[self.pos] = log_prob.to(self.device)
        
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
    
    def compute_returns_and_advantages(
        self,
        last_values: torch.Tensor,
        last_dones: torch.Tensor
    ):
        """
        Compute returns and advantages using Generalized Advantage Estimation.
        
        GAE-Lambda advantage estimation:
        A_t = delta_t + gamma * lambda * A_{t+1}
        where delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
        
        Args:
            last_values: Value estimates for the last observation (n_envs,)
            last_dones: Done flags for the last step (n_envs,)
        """
        if isinstance(last_values, np.ndarray):
            last_values = torch.from_numpy(last_values)
        if isinstance(last_dones, np.ndarray):
            last_dones = torch.from_numpy(last_dones)
        
        last_values = last_values.to(self.device)
        last_dones = last_dones.float().to(self.device)
        
        # Initialize last_gae_lam
        last_gae_lam = torch.zeros(self.n_envs, device=self.device)
        
        # Compute advantages in reverse order
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - last_dones
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.dones[step + 1]
                next_values = self.values[step + 1]
            
            # TD error: delta = r + gamma * V(s') - V(s)
            delta = (
                self.rewards[step]
                + self.gamma * next_values * next_non_terminal
                - self.values[step]
            )
            
            # GAE: A = delta + gamma * lambda * A'
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam
        
        # Returns = advantages + values
        returns = self.advantages + self.values
        self.returns = returns
        
        self.generator_ready = True
    
    def get(
        self,
        batch_size: Optional[int] = None
    ) -> Generator[RolloutBufferSamples, None, None]:
        """
        Generate mini-batches from the buffer.
        
        Args:
            batch_size: Size of mini-batches. If None, use entire buffer.
        
        Yields:
            RolloutBufferSamples containing mini-batch data
        """
        if not self.generator_ready:
            raise RuntimeError(
                "Call compute_returns_and_advantages() before generating batches!"
            )
        
        # Flatten the buffer
        total_size = self.buffer_size * self.n_envs
        
        # Reshape to (total_size, ...)
        observations = self.observations.reshape((total_size,) + self.observation_shape)
        actions = self.actions.reshape(total_size, self.action_dim)
        values = self.values.reshape(total_size)
        log_probs = self.log_probs.reshape(total_size)
        advantages = self.advantages.reshape(total_size)
        returns = self.returns.reshape(total_size)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        if batch_size is None:
            batch_size = total_size
        
        # Generate random indices
        indices = torch.randperm(total_size, device=self.device)
        
        # Yield mini-batches
        start_idx = 0
        while start_idx < total_size:
            batch_indices = indices[start_idx:start_idx + batch_size]
            
            yield RolloutBufferSamples(
                observations=observations[batch_indices],
                actions=actions[batch_indices],
                old_values=values[batch_indices],
                old_log_probs=log_probs[batch_indices],
                advantages=advantages[batch_indices],
                returns=returns[batch_indices]
            )
            
            start_idx += batch_size
    
    def get_all(self) -> RolloutBufferSamples:
        """Get all data as a single batch (for small datasets)."""
        if not self.generator_ready:
            raise RuntimeError(
                "Call compute_returns_and_advantages() before getting data!"
            )
        
        total_size = self.buffer_size * self.n_envs
        
        observations = self.observations.reshape((total_size,) + self.observation_shape)
        actions = self.actions.reshape(total_size, self.action_dim)
        values = self.values.reshape(total_size)
        log_probs = self.log_probs.reshape(total_size)
        advantages = self.advantages.reshape(total_size)
        returns = self.returns.reshape(total_size)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return RolloutBufferSamples(
            observations=observations,
            actions=actions,
            old_values=values,
            old_log_probs=log_probs,
            advantages=advantages,
            returns=returns
        )
