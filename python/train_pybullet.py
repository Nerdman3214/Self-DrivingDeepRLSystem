"""
PyBullet 3D Training Script

PHASE 2: Reinforcement Learning Fundamentals
============================================

This script teaches RL concepts through a realistic 3D environment.

What You'll Learn:
    - How reward functions shape behavior
    - Episode structure and termination
    - Policy gradient training (PPO)
    - Learning curves and metrics
    - Visualization of learning progress

This is a tutorial-focused entry point to the system.
For production training, see: train_lane_keeping.py
"""

import torch
import numpy as np
from pathlib import Path
import argparse
from datetime import datetime
from typing import Dict, List, Tuple
import time

from rl.envs.pybullet_driving_env import PyBulletDrivingEnv
from rl.networks.mlp_policy import MLPActorCritic


# ==============================================================================
# CONFIGURATION (Tutorial: These are hyperparameters)
# ==============================================================================

class TrainingConfig:
    """
    Hyperparameters for PPO Training
    
    Tutorial Notes:
        - learning_rate: How fast the AI updates (too high = unstable)
        - gamma: How much to value future rewards (0.99 = patient)
        - gae_lambda: Advantage estimation smoothness
        - clip_epsilon: PPO's safety limit on policy changes
    """
    
    # Environment
    max_episode_steps = 1000
    render_during_training = False  # Set True to watch learning
    
    # PPO Algorithm
    learning_rate = 3e-4
    gamma = 0.99              # Discount factor
    gae_lambda = 0.95         # GAE parameter
    clip_epsilon = 0.2        # PPO clip range
    value_coef = 0.5          # Value loss coefficient
    entropy_coef = 0.01       # Exploration bonus
    
    # Training
    total_timesteps = 100000
    batch_size = 64
    num_epochs = 10
    
    # Logging
    log_interval = 10         # Episodes between logs
    eval_interval = 1000      # Steps between evaluations
    save_interval = 5000      # Steps between checkpoints


# ==============================================================================
# SIMPLE PPO TRAINER (Tutorial Implementation)
# ==============================================================================

class SimplePPOTrainer:
    """
    Simplified PPO Trainer for Learning
    
    Tutorial: This is a minimal but correct PPO implementation.
    For production use, see rl/ppo/ppo_trainer.py
    
    Key Concepts:
        1. Collect experience (rollout)
        2. Compute advantages (how good were actions?)
        3. Update policy (make good actions more likely)
        4. Update value function (predict rewards better)
    """
    
    def __init__(self, env, policy, config: TrainingConfig):
        self.env = env
        self.policy = policy
        self.config = config
        
        # Optimizer (Adam is standard for deep RL)
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=config.learning_rate
        )
        
        # Metrics tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.total_steps = 0
        
    def collect_rollout(self, num_steps: int) -> Dict:
        """
        TUTORIAL: Rollout Collection
        
        This is the "experience gathering" phase.
        The agent acts in the environment and remembers:
            - What it saw (observations)
            - What it did (actions)
            - What it got (rewards)
            - What happened next (next observations)
        
        This is the DATA that PPO learns from.
        """
        observations = []
        actions = []
        rewards = []
        values = []
        log_probs = []
        dones = []
        
        obs, _ = self.env.reset()
        episode_reward = 0
        episode_length = 0
        
        for step in range(num_steps):
            # Convert to tensor
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            
            # Get action from policy (STOCHASTIC - explores)
            with torch.no_grad():
                action, log_prob, _, value = self.policy.get_action_and_value(obs_tensor)
            
            # Execute action in environment
            next_obs, reward, terminated, truncated, info = self.env.step(action[0].numpy())
            done = terminated or truncated
            
            # Store experience
            observations.append(obs)
            actions.append(action[0].numpy())
            rewards.append(reward)
            values.append(value.item())
            log_probs.append(log_prob.item())
            dones.append(done)
            
            # Update tracking
            episode_reward += reward
            episode_length += 1
            self.total_steps += 1
            
            # Episode ended?
            if done:
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                
                obs, _ = self.env.reset()
                episode_reward = 0
                episode_length = 0
            else:
                obs = next_obs
        
        return {
            'observations': np.array(observations),
            'actions': np.array(actions),
            'rewards': np.array(rewards),
            'values': np.array(values),
            'log_probs': np.array(log_probs),
            'dones': np.array(dones)
        }
    
    def compute_advantages(self, rollout: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        TUTORIAL: Advantage Estimation (GAE)
        
        "Advantage" = How much better was this action than average?
        
        Math:
            A_t = reward + gamma * V(next_state) - V(current_state)
        
        If A_t > 0: Action was good (increase probability)
        If A_t < 0: Action was bad (decrease probability)
        
        GAE smooths this over multiple steps for stability.
        """
        rewards = rollout['rewards']
        values = rollout['values']
        dones = rollout['dones']
        
        advantages = np.zeros_like(rewards)
        returns = np.zeros_like(rewards)
        
        gae = 0
        next_value = 0
        
        # Compute backwards (from end of episode)
        for t in reversed(range(len(rewards))):
            if dones[t]:
                next_value = 0
                gae = 0
            
            # TD error (temporal difference)
            delta = rewards[t] + self.config.gamma * next_value - values[t]
            
            # GAE accumulation
            gae = delta + self.config.gamma * self.config.gae_lambda * gae
            
            advantages[t] = gae
            returns[t] = gae + values[t]
            
            next_value = values[t]
        
        # Normalize advantages (stabilizes training)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def update_policy(self, rollout: Dict, advantages: np.ndarray, returns: np.ndarray):
        """
        TUTORIAL: PPO Policy Update
        
        This is where learning happens!
        
        PPO's key insight: Don't change the policy too much at once.
        The "clipping" prevents catastrophic updates.
        
        Loss has three parts:
            1. Policy loss (make good actions more likely)
            2. Value loss (predict rewards better)
            3. Entropy loss (encourage exploration)
        """
        # Convert to tensors
        obs_tensor = torch.FloatTensor(rollout['observations'])
        actions_tensor = torch.FloatTensor(rollout['actions'])
        old_log_probs_tensor = torch.FloatTensor(rollout['log_probs'])
        advantages_tensor = torch.FloatTensor(advantages)
        returns_tensor = torch.FloatTensor(returns)
        
        # Multiple epochs over same data (PPO can reuse data!)
        for epoch in range(self.config.num_epochs):
            # Get new predictions with current policy
            values_pred, new_log_probs, entropy = self.policy.evaluate_actions(
                obs_tensor, actions_tensor
            )
            
            # Ratio of new to old probabilities
            ratio = torch.exp(new_log_probs - old_log_probs_tensor)
            
            # PPO clipped objective (THE MAGIC!)
            surr1 = ratio * advantages_tensor
            surr2 = torch.clamp(
                ratio,
                1.0 - self.config.clip_epsilon,
                1.0 + self.config.clip_epsilon
            ) * advantages_tensor
            
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value function loss (MSE between prediction and actual return)
            value_loss = ((values_pred - returns_tensor) ** 2).mean()
            
            # Entropy bonus (encourages exploration)
            entropy_loss = -entropy.mean()
            
            # Total loss
            loss = (
                policy_loss +
                self.config.value_coef * value_loss +
                self.config.entropy_coef * entropy_loss
            )
            
            # Gradient descent step
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()
    
    def train(self, save_dir: Path):
        """
        TUTORIAL: Main Training Loop
        
        Structure:
            1. Collect experience
            2. Compute advantages
            3. Update policy
            4. Log metrics
            5. Repeat
        
        This is the outer loop of RL training.
        """
        save_dir.mkdir(parents=True, exist_ok=True)
        
        print("\n" + "="*70)
        print("🚗 PHASE 2: PPO TRAINING ON PYBULLET 3D ENVIRONMENT")
        print("="*70)
        print(f"Total timesteps: {self.config.total_timesteps:,}")
        print(f"Learning rate: {self.config.learning_rate}")
        print(f"Batch size: {self.config.batch_size}")
        print("="*70 + "\n")
        
        start_time = time.time()
        
        while self.total_steps < self.config.total_timesteps:
            # === STEP 1: COLLECT EXPERIENCE ===
            rollout = self.collect_rollout(self.config.batch_size)
            
            # === STEP 2: COMPUTE ADVANTAGES ===
            advantages, returns = self.compute_advantages(rollout)
            
            # === STEP 3: UPDATE POLICY ===
            self.update_policy(rollout, advantages, returns)
            
            # === STEP 4: LOGGING ===
            if len(self.episode_rewards) > 0 and len(self.episode_rewards) % self.config.log_interval == 0:
                recent_rewards = self.episode_rewards[-self.config.log_interval:]
                recent_lengths = self.episode_lengths[-self.config.log_interval:]
                
                print(f"Steps: {self.total_steps:6d} | "
                      f"Episodes: {len(self.episode_rewards):4d} | "
                      f"Reward: {np.mean(recent_rewards):7.2f} ± {np.std(recent_rewards):6.2f} | "
                      f"Length: {np.mean(recent_lengths):6.1f}")
            
            # === STEP 5: SAVE CHECKPOINT ===
            if self.total_steps % self.config.save_interval == 0:
                checkpoint_path = save_dir / f"pybullet_model_{self.total_steps}.pt"
                torch.save({
                    'policy_state_dict': self.policy.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'total_steps': self.total_steps,
                    'episode_rewards': self.episode_rewards,
                }, checkpoint_path)
                print(f"💾 Checkpoint saved: {checkpoint_path}")
        
        # Final save
        final_path = save_dir / "pybullet_model_final.pt"
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'total_steps': self.total_steps,
            'episode_rewards': self.episode_rewards,
        }, final_path)
        
        elapsed_time = time.time() - start_time
        
        print("\n" + "="*70)
        print("✅ TRAINING COMPLETE!")
        print("="*70)
        print(f"Total episodes: {len(self.episode_rewards)}")
        print(f"Final avg reward: {np.mean(self.episode_rewards[-100:]):.2f}")
        print(f"Training time: {elapsed_time/60:.1f} minutes")
        print(f"Model saved: {final_path}")
        print("="*70 + "\n")


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Phase 2: Train PPO on PyBullet 3D Environment"
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=100000,
        help="Total training timesteps"
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render environment during training (slow but educational)"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="checkpoints/pybullet",
        help="Directory to save checkpoints"
    )
    
    args = parser.parse_args()
    
    # Update config from args
    config = TrainingConfig()
    config.total_timesteps = args.timesteps
    config.render_during_training = args.render
    
    # Create environment
    print("🏗️  Creating PyBullet 3D environment...")
    env = PyBulletDrivingEnv(
        render_mode="human" if args.render else None,
        max_episode_steps=config.max_episode_steps
    )
    
    # Create policy network
    print("🧠 Initializing PPO policy...")
    policy = MLPActorCritic(
        observation_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        hidden_dims=(64, 64)
    )
    
    print(f"📊 Observation space: {env.observation_space.shape}")
    print(f"🎮 Action space: {env.action_space.shape}")
    print(f"🔢 Policy parameters: {sum(p.numel() for p in policy.parameters()):,}")
    
    # Create trainer
    trainer = SimplePPOTrainer(env, policy, config)
    
    # Train!
    save_dir = Path(args.save_dir)
    trainer.train(save_dir)
    
    # Cleanup
    env.close()


if __name__ == "__main__":
    main()
