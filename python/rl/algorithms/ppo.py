"""
Proximal Policy Optimization (PPO) Algorithm

Implementation of PPO-Clip for training self-driving agents.
Reference: Schulman et al., "Proximal Policy Optimization Algorithms" (2017)
"""

import torch
import torch.optim as optim
import numpy as np
from typing import Dict, Optional
import time

from ..networks.actor_critic import ActorCritic
from .rollout_buffer import RolloutBuffer


class PPO:
    """
    Proximal Policy Optimization (PPO-Clip) algorithm.
    
    Key features:
    - Clipped surrogate objective for stable policy updates
    - Generalized Advantage Estimation (GAE) for variance reduction
    - Value function clipping (optional)
    - Entropy bonus for exploration
    """
    
    def __init__(
        self,
        policy: ActorCritic,
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        clip_range_vf: Optional[float] = None,
        normalize_advantage: bool = True,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        target_kl: Optional[float] = None,
        device: torch.device = torch.device("cpu"),
        n_envs: int = 1
    ):
        """
        Initialize PPO.
        
        Args:
            policy: ActorCritic policy network
            learning_rate: Learning rate for optimizer
            n_steps: Number of steps to collect per environment per update
            batch_size: Mini-batch size for training
            n_epochs: Number of epochs to train on collected data
            gamma: Discount factor
            gae_lambda: Lambda for GAE
            clip_range: Clipping parameter for PPO objective
            clip_range_vf: Clipping parameter for value function (None = no clipping)
            normalize_advantage: Whether to normalize advantages
            entropy_coef: Coefficient for entropy loss
            value_coef: Coefficient for value function loss
            max_grad_norm: Maximum gradient norm for clipping
            target_kl: Target KL divergence for early stopping (None = disabled)
            device: Device to train on
            n_envs: Number of parallel environments
        """
        self.policy = policy.to(device)
        self.device = device
        self.n_envs = n_envs
        
        # PPO hyperparameters
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.normalize_advantage = normalize_advantage
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        
        # Optimizer
        self.optimizer = optim.Adam(policy.parameters(), lr=learning_rate, eps=1e-5)
        
        # Get observation shape from policy
        self.observation_shape = (
            policy.feature_extractor.input_channels,
            policy.feature_extractor.input_height,
            policy.feature_extractor.input_width
        )
        self.action_dim = policy.action_dim
        
        # Rollout buffer
        self.rollout_buffer = RolloutBuffer(
            buffer_size=n_steps,
            observation_shape=self.observation_shape,
            action_dim=self.action_dim,
            device=device,
            gamma=gamma,
            gae_lambda=gae_lambda,
            n_envs=n_envs
        )
        
        # Training statistics
        self.num_timesteps = 0
        self._n_updates = 0
    
    def collect_rollouts(self, env, callback=None) -> bool:
        """
        Collect rollouts from the environment.
        
        Args:
            env: Vectorized environment
            callback: Optional callback function
        
        Returns:
            True if rollout collection was successful
        """
        self.policy.eval()
        self.rollout_buffer.reset()
        
        # Get initial observation
        if not hasattr(self, '_last_obs'):
            self._last_obs = env.reset()
            if isinstance(self._last_obs, tuple):
                self._last_obs = self._last_obs[0]  # Handle gym 0.26+ API
            self._last_dones = np.zeros(self.n_envs, dtype=bool)
        
        for _ in range(self.n_steps):
            # Convert observation to tensor
            obs_tensor = torch.from_numpy(self._last_obs).float().to(self.device)
            
            # Handle single env case
            if obs_tensor.dim() == 3:
                obs_tensor = obs_tensor.unsqueeze(0)
            
            # Ensure correct format (batch, channels, height, width)
            if obs_tensor.shape[-1] == 3 or obs_tensor.shape[-1] == 4:
                # HWC -> CHW format
                obs_tensor = obs_tensor.permute(0, 3, 1, 2)
            
            # Normalize to [0, 1]
            if obs_tensor.max() > 1.0:
                obs_tensor = obs_tensor / 255.0
            
            # Get action and value
            with torch.no_grad():
                actions, log_probs, _, values = self.policy(obs_tensor)
            
            # Convert to numpy for environment
            actions_np = actions.cpu().numpy()
            
            # Clip actions to valid range for CarRacing
            actions_np = np.clip(actions_np, -1.0, 1.0)
            # Rescale gas and brake to [0, 1]
            actions_np[:, 1] = (actions_np[:, 1] + 1.0) / 2.0  # gas
            actions_np[:, 2] = (actions_np[:, 2] + 1.0) / 2.0  # brake
            
            # Step environment
            if self.n_envs == 1:
                step_result = env.step(actions_np[0])
            else:
                step_result = env.step(actions_np)
            
            # Handle different gym API versions
            if len(step_result) == 5:
                new_obs, rewards, terminated, truncated, infos = step_result
                dones = np.logical_or(terminated, truncated)
            else:
                new_obs, rewards, dones, infos = step_result
            
            # Handle single env case
            if self.n_envs == 1:
                new_obs = np.expand_dims(new_obs, 0)
                rewards = np.array([rewards])
                dones = np.array([dones])
            
            self.num_timesteps += self.n_envs
            
            # Store in buffer
            # Convert obs back to CHW if needed for storage
            obs_store = self._last_obs
            if obs_store.ndim == 3:
                obs_store = np.expand_dims(obs_store, 0)
            if obs_store.shape[-1] == 3 or obs_store.shape[-1] == 4:
                obs_store = np.transpose(obs_store, (0, 3, 1, 2))
            if obs_store.max() > 1.0:
                obs_store = obs_store / 255.0
            
            self.rollout_buffer.add(
                obs=obs_store,
                action=actions.cpu(),
                reward=rewards,
                done=dones,
                value=values.cpu(),
                log_prob=log_probs.cpu()
            )
            
            self._last_obs = new_obs
            self._last_dones = dones
            
            # Handle episode resets
            for _, done in enumerate(dones):
                if done:
                    if isinstance(infos, dict) and 'final_observation' in infos:
                        pass  # VecEnv auto-resets
                    elif self.n_envs == 1:
                        reset_result = env.reset()
                        if isinstance(reset_result, tuple):
                            self._last_obs = np.expand_dims(reset_result[0], 0)
                        else:
                            self._last_obs = np.expand_dims(reset_result, 0)
            
            if callback is not None:
                callback(locals(), globals())
        
        # Compute returns and advantages
        with torch.no_grad():
            # Get value for last observation
            obs_tensor = torch.from_numpy(self._last_obs).float().to(self.device)
            if obs_tensor.dim() == 3:
                obs_tensor = obs_tensor.unsqueeze(0)
            if obs_tensor.shape[-1] == 3 or obs_tensor.shape[-1] == 4:
                obs_tensor = obs_tensor.permute(0, 3, 1, 2)
            if obs_tensor.max() > 1.0:
                obs_tensor = obs_tensor / 255.0
            
            last_values = self.policy.get_value(obs_tensor)
        
        self.rollout_buffer.compute_returns_and_advantages(
            last_values=last_values.cpu(),
            last_dones=self._last_dones
        )
        
        return True
    
    def train(self) -> Dict[str, float]:
        """
        Perform PPO update using collected rollouts.
        
        Returns:
            Dictionary of training statistics
        """
        self.policy.train()
        
        # Training statistics
        pg_losses = []
        value_losses = []
        entropy_losses = []
        clip_fractions = []
        approx_kl_divs = []
        
        continue_training = True
        
        for _ in range(self.n_epochs):
            if not continue_training:
                break
            
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                # Get current policy outputs
                log_probs, entropy, values = self.policy.evaluate_actions(
                    rollout_data.observations,
                    rollout_data.actions
                )
                
                # Normalize advantages if configured
                advantages = rollout_data.advantages
                if self.normalize_advantage:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
                # Ratio for PPO update: r(θ) = π(a|s) / π_old(a|s)
                ratio = torch.exp(log_probs - rollout_data.old_log_probs)
                
                # Clipped surrogate objective
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(
                    ratio, 1.0 - self.clip_range, 1.0 + self.clip_range
                )
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
                
                # Value function loss
                if self.clip_range_vf is not None:
                    # Clipped value loss
                    values_clipped = rollout_data.old_values + torch.clamp(
                        values - rollout_data.old_values,
                        -self.clip_range_vf, self.clip_range_vf
                    )
                    value_loss_1 = (values - rollout_data.returns).pow(2)
                    value_loss_2 = (values_clipped - rollout_data.returns).pow(2)
                    value_loss = 0.5 * torch.max(value_loss_1, value_loss_2).mean()
                else:
                    value_loss = 0.5 * (values - rollout_data.returns).pow(2).mean()
                
                # Entropy loss (negative because we want to maximize entropy)
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    + self.entropy_coef * entropy_loss
                )
                
                # Optimization step
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # Logging
                pg_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())
                
                # Calculate clip fraction
                with torch.no_grad():
                    clip_fraction = torch.mean(
                        (torch.abs(ratio - 1) > self.clip_range).float()
                    ).item()
                    clip_fractions.append(clip_fraction)
                    
                    # Approximate KL divergence
                    log_ratio = log_probs - rollout_data.old_log_probs
                    approx_kl = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).item()
                    approx_kl_divs.append(approx_kl)
                
                # Early stopping based on KL divergence
                if self.target_kl is not None and approx_kl > 1.5 * self.target_kl:
                    continue_training = False
                    break
        
        self._n_updates += self.n_epochs
        
        return {
            "policy_loss": np.mean(pg_losses),
            "value_loss": np.mean(value_losses),
            "entropy_loss": np.mean(entropy_losses),
            "clip_fraction": np.mean(clip_fractions),
            "approx_kl": np.mean(approx_kl_divs),
            "n_updates": self._n_updates
        }
    
    def learn(
        self,
        env,
        total_timesteps: int,
        callback=None,
        log_interval: int = 1,
        eval_env=None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5
    ) -> "PPO":
        """
        Train the agent.
        
        Args:
            env: Training environment
            total_timesteps: Total number of timesteps to train
            callback: Optional callback
            log_interval: Log every N updates
            eval_env: Optional evaluation environment
            eval_freq: Evaluate every N timesteps (-1 = disable)
            n_eval_episodes: Number of episodes for evaluation
        
        Returns:
            Self
        """
        iteration = 0
        
        while self.num_timesteps < total_timesteps:
            iteration += 1
            
            # Collect rollouts
            start_time = time.time()
            self.collect_rollouts(env, callback)
            rollout_time = time.time() - start_time
            
            # Train on collected data
            start_time = time.time()
            train_stats = self.train()
            train_time = time.time() - start_time
            
            # Logging
            if log_interval > 0 and iteration % log_interval == 0:
                fps = int(self.n_steps * self.n_envs / (rollout_time + train_time))
                print(f"\n{'='*60}")
                print(f"Iteration {iteration} | Timesteps: {self.num_timesteps}/{total_timesteps}")
                print(f"{'='*60}")
                print(f"  Policy Loss: {train_stats['policy_loss']:.4f}")
                print(f"  Value Loss: {train_stats['value_loss']:.4f}")
                print(f"  Entropy: {-train_stats['entropy_loss']:.4f}")
                print(f"  Clip Fraction: {train_stats['clip_fraction']:.3f}")
                print(f"  Approx KL: {train_stats['approx_kl']:.4f}")
                print(f"  FPS: {fps}")
            
            # Evaluation
            if eval_env is not None and eval_freq > 0:
                if self.num_timesteps % eval_freq < self.n_steps * self.n_envs:
                    eval_rewards = self.evaluate(eval_env, n_eval_episodes)
                    print(f"\n  Eval Reward: {np.mean(eval_rewards):.2f} +/- {np.std(eval_rewards):.2f}")
        
        return self
    
    def evaluate(self, env, n_episodes: int = 5) -> np.ndarray:
        """
        Evaluate the policy.
        
        Args:
            env: Evaluation environment
            n_episodes: Number of episodes to evaluate
        
        Returns:
            Array of episode rewards
        """
        self.policy.eval()
        episode_rewards = []
        
        for _ in range(n_episodes):
            obs = env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]
            
            done = False
            total_reward = 0.0
            
            while not done:
                obs_tensor = torch.from_numpy(obs).float().to(self.device)
                if obs_tensor.dim() == 3:
                    obs_tensor = obs_tensor.unsqueeze(0)
                if obs_tensor.shape[-1] == 3:
                    obs_tensor = obs_tensor.permute(0, 3, 1, 2)
                if obs_tensor.max() > 1.0:
                    obs_tensor = obs_tensor / 255.0
                
                with torch.no_grad():
                    action, _ = self.policy.get_action(obs_tensor, deterministic=True)
                
                action_np = action.cpu().numpy()[0]
                action_np = np.clip(action_np, -1.0, 1.0)
                action_np[1] = (action_np[1] + 1.0) / 2.0
                action_np[2] = (action_np[2] + 1.0) / 2.0
                
                step_result = env.step(action_np)
                if len(step_result) == 5:
                    obs, reward, terminated, truncated, _ = step_result
                    done = terminated or truncated
                else:
                    obs, reward, done, _ = step_result
                
                total_reward += reward
            
            episode_rewards.append(total_reward)
        
        return np.array(episode_rewards)
    
    def save(self, path: str):
        """Save the policy and optimizer state."""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'num_timesteps': self.num_timesteps,
            'n_updates': self._n_updates
        }, path)
        print(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load the policy and optimizer state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.num_timesteps = checkpoint.get('num_timesteps', 0)
        self._n_updates = checkpoint.get('n_updates', 0)
        print(f"Model loaded from {path}")
