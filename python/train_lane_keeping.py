#!/usr/bin/env python3
"""
Training script for Lane-Keeping Self-Driving RL

This trains an agent to:
- Stay centered in the lane
- Maintain appropriate speed
- Align with road heading
- Handle curves smoothly

No camera. Pure state-based control.
"""

import os
import argparse
from pathlib import Path
from typing import Dict, Any
import time

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from rl.envs import LaneKeepingEnv, make_lane_keeping_env
from rl.networks import MLPActorCritic
from rl.algorithms.ppo import PPO
from rl.algorithms.rollout_buffer import RolloutBuffer
from rl.utils.logger import Logger
from rl.utils.scheduler import LinearSchedule


def parse_args():
    parser = argparse.ArgumentParser(description="Train Lane-Keeping Agent with PPO")
    
    # Environment
    parser.add_argument('--env', type=str, default='lane-keeping',
                        help='Environment name')
    parser.add_argument('--lane-width', type=float, default=3.5,
                        help='Lane width in meters')
    parser.add_argument('--max-speed', type=float, default=30.0,
                        help='Maximum speed in m/s')
    parser.add_argument('--episode-steps', type=int, default=1000,
                        help='Maximum steps per episode')
    
    # Training
    parser.add_argument('--total-timesteps', type=int, default=500000,
                        help='Total training timesteps')
    parser.add_argument('--n-envs', type=int, default=1,
                        help='Number of parallel environments')
    
    # PPO hyperparameters
    parser.add_argument('--learning-rate', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--n-steps', type=int, default=2048,
                        help='Steps per rollout')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Mini-batch size')
    parser.add_argument('--n-epochs', type=int, default=10,
                        help='Epochs per update')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor')
    parser.add_argument('--gae-lambda', type=float, default=0.95,
                        help='GAE lambda')
    parser.add_argument('--clip-range', type=float, default=0.2,
                        help='PPO clip range')
    parser.add_argument('--entropy-coef', type=float, default=0.01,
                        help='Entropy coefficient')
    parser.add_argument('--value-coef', type=float, default=0.5,
                        help='Value loss coefficient')
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                        help='Max gradient norm')
    parser.add_argument('--target-kl', type=float, default=0.01,
                        help='Target KL divergence')
    
    # Network architecture
    parser.add_argument('--hidden-dims', type=int, nargs='+', default=[256, 256],
                        help='Hidden layer dimensions')
    parser.add_argument('--log-std-init', type=float, default=-0.5,
                        help='Initial log std for policy')
    
    # Logging and checkpointing
    parser.add_argument('--exp-name', type=str, default='lane_keeping_ppo',
                        help='Experiment name')
    parser.add_argument('--log-dir', type=str, default='logs',
                        help='Log directory')
    parser.add_argument('--save-freq', type=int, default=50000,
                        help='Checkpoint save frequency')
    parser.add_argument('--eval-freq', type=int, default=10000,
                        help='Evaluation frequency')
    parser.add_argument('--n-eval-episodes', type=int, default=5,
                        help='Number of evaluation episodes')
    
    # Device and seed
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (cpu, cuda, auto)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    return parser.parse_args()


def make_env(args, seed: int = 0, render_mode: str = None):
    """Create a single lane-keeping environment."""
    def _init():
        env = LaneKeepingEnv(
            lane_width=args.lane_width,
            max_speed=args.max_speed,
            max_episode_steps=args.episode_steps,
            random_start=True,
            render_mode=render_mode,
        )
        env.reset(seed=seed)
        return env
    return _init


def evaluate_policy(
    env: LaneKeepingEnv,
    policy: MLPActorCritic,
    n_episodes: int = 5,
    deterministic: bool = True,
    render: bool = False
) -> Dict[str, float]:
    """Evaluate policy for n episodes."""
    episode_rewards = []
    episode_lengths = []
    completion_rate = 0
    
    for episode in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done:
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(policy.device)
            
            with torch.no_grad():
                action, _, _, _ = policy.get_action_and_value(
                    obs_tensor, 
                    deterministic=deterministic
                )
            
            action = action.cpu().numpy()[0]
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            episode_length += 1
            
            if render:
                env.render()
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # Check if completed successfully (reached end without crash)
        if truncated and not terminated:
            completion_rate += 1
    
    return {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'completion_rate': completion_rate / n_episodes,
    }


def train(args):
    """Main training loop."""
    
    # Setup
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create directories
    exp_dir = Path(args.log_dir) / args.exp_name
    checkpoint_dir = exp_dir / 'checkpoints'
    tensorboard_dir = exp_dir / 'tensorboard'
    
    for directory in [exp_dir, checkpoint_dir, tensorboard_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    
    # Save config
    import json
    with open(exp_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Create environment
    env = make_env(args, seed=args.seed)()
    eval_env = make_env(args, seed=args.seed + 1000)()
    
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    print(f"\nEnvironment: Lane Keeping")
    print(f"Observation space: {obs_dim}D vector")
    print(f"Action space: {action_dim}D continuous")
    print(f"State: [lane_offset, heading_error, speed, left_dist, right_dist, curvature]")
    print(f"Action: [steering, throttle]\n")
    
    # Create policy network
    policy = MLPActorCritic(
        observation_dim=obs_dim,
        action_dim=action_dim,
        hidden_dims=tuple(args.hidden_dims),
        log_std_init=args.log_std_init,
    ).to(device)
    
    policy.device = device
    
    # Create optimizer
    optimizer = torch.optim.Adam(
        policy.parameters(),
        lr=args.learning_rate,
        eps=1e-5
    )
    
    # Learning rate scheduler (optional)
    lr_scheduler = LinearSchedule(
        initial_value=args.learning_rate,
        final_value=args.learning_rate * 0.1,
        schedule_timesteps=args.total_timesteps
    )
    
    # Create rollout buffer
    buffer = RolloutBuffer(
        buffer_size=args.n_steps,
        observation_shape=(obs_dim,),
        action_dim=action_dim,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        n_envs=1,
    )
    
    # TensorBoard writer
    writer = SummaryWriter(str(tensorboard_dir))
    
    # Training state
    obs, _ = env.reset()
    episode_reward = 0
    episode_length = 0
    num_updates = 0
    
    print("Starting training...\n")
    start_time = time.time()
    
    for timestep in range(1, args.total_timesteps + 1):
        # Collect rollout
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
        
        with torch.no_grad():
            action, log_prob, _, value = policy.get_action_and_value(obs_tensor)
        
        action_np = action.cpu().numpy()[0]
        next_obs, reward, terminated, truncated, info = env.step(action_np)
        done = terminated or truncated
        
        # Store transition
        buffer.add(
            obs=obs,
            action=action_np,
            reward=np.array([reward], dtype=np.float32),
            value=value.cpu(),
            log_prob=log_prob.cpu(),
            done=np.array([done], dtype=np.float32),
        )
        
        obs = next_obs
        episode_reward += reward
        episode_length += 1
        
        # Episode end
        if done:
            writer.add_scalar('train/episode_reward', episode_reward, timestep)
            writer.add_scalar('train/episode_length', episode_length, timestep)
            writer.add_scalar('train/lane_offset', info['lane_offset_normalized'], timestep)
            writer.add_scalar('train/speed', info['speed'], timestep)
            
            print(f"Step {timestep:>7d} | Reward: {episode_reward:>7.2f} | "
                  f"Length: {episode_length:>4d} | "
                  f"Lane Offset: {info['lane_offset_normalized']:>6.3f}")
            
            obs, _ = env.reset()
            episode_reward = 0
            episode_length = 0
        
        # Update policy
        if timestep % args.n_steps == 0:
            # Compute advantages
            with torch.no_grad():
                next_obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
                next_value = policy.get_value(next_obs_tensor).cpu()
            
            buffer.compute_returns_and_advantages(
                next_value, 
                torch.tensor([[done]], dtype=torch.float32)
            )
            
            # PPO update
            for epoch in range(args.n_epochs):
                # Get minibatches from buffer
                for rollout_data in buffer.get(batch_size=args.batch_size):
                    # Unpack batch
                    batch_obs = rollout_data.observations.to(device)
                    batch_actions = rollout_data.actions.to(device)
                    batch_old_values = rollout_data.old_values.to(device)
                    batch_old_log_probs = rollout_data.old_log_probs.to(device)
                    batch_advantages = rollout_data.advantages.to(device)
                    batch_returns = rollout_data.returns.to(device)
                    
                    # Normalize advantages
                    batch_advantages = (batch_advantages - batch_advantages.mean()) / (
                        batch_advantages.std() + 1e-8
                    )
                    
                    # Forward pass
                    values, log_probs, entropy = policy.evaluate_actions(
                        batch_obs, batch_actions
                    )
                    
                    # Policy loss
                    ratio = torch.exp(log_probs - batch_old_log_probs)
                    surr1 = ratio * batch_advantages
                    surr2 = torch.clamp(ratio, 1 - args.clip_range, 1 + args.clip_range) * batch_advantages
                    policy_loss = -torch.min(surr1, surr2).mean()
                    
                    # Value loss
                    value_loss = ((values.squeeze() - batch_returns) ** 2).mean()
                    
                    # Entropy loss
                    entropy_loss = -entropy.mean()
                    
                    # Total loss
                    loss = (
                        policy_loss +
                        args.value_coef * value_loss +
                        args.entropy_coef * entropy_loss
                    )
                    
                    # Optimize
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
                    optimizer.step()
                    
                    # Check KL divergence
                    with torch.no_grad():
                        kl_div = (batch_old_log_probs - log_probs).mean().item()
                        if kl_div > args.target_kl:
                            break
            
            # Logging
            num_updates += 1
            writer.add_scalar('train/policy_loss', policy_loss.item(), timestep)
            writer.add_scalar('train/value_loss', value_loss.item(), timestep)
            writer.add_scalar('train/entropy', -entropy_loss.item(), timestep)
            writer.add_scalar('train/kl_divergence', kl_div, timestep)
            writer.add_scalar('train/learning_rate', optimizer.param_groups[0]['lr'], timestep)
            
            # Update learning rate
            new_lr = lr_scheduler(timestep / args.total_timesteps)
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr
            
            # Reset buffer
            buffer.reset()
        
        # Evaluation
        if timestep % args.eval_freq == 0:
            print(f"\n{'='*60}")
            print(f"Evaluation at step {timestep}")
            print(f"{'='*60}")
            
            eval_stats = evaluate_policy(
                eval_env, policy, n_episodes=args.n_eval_episodes
            )
            
            print(f"Mean reward: {eval_stats['mean_reward']:.2f} ± {eval_stats['std_reward']:.2f}")
            print(f"Mean length: {eval_stats['mean_length']:.1f}")
            print(f"Completion rate: {eval_stats['completion_rate']:.1%}")
            print(f"{'='*60}\n")
            
            writer.add_scalar('eval/mean_reward', eval_stats['mean_reward'], timestep)
            writer.add_scalar('eval/completion_rate', eval_stats['completion_rate'], timestep)
        
        # Save checkpoint
        if timestep % args.save_freq == 0:
            checkpoint_path = checkpoint_dir / f'model_{timestep}.pt'
            torch.save({
                'timestep': timestep,
                'policy_state_dict': policy.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'args': vars(args),
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
    
    # Final save
    final_path = checkpoint_dir / 'model_final.pt'
    torch.save({
        'timestep': args.total_timesteps,
        'policy_state_dict': policy.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'args': vars(args),
    }, final_path)
    
    elapsed_time = time.time() - start_time
    print(f"\nTraining complete!")
    print(f"Total time: {elapsed_time/3600:.2f} hours")
    print(f"Final model saved: {final_path}")
    
    writer.close()
    env.close()
    eval_env.close()


if __name__ == '__main__':
    args = parse_args()
    train(args)
