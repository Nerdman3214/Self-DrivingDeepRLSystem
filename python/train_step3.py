#!/usr/bin/env python3
"""
STEP 3: Training, Evaluation, and Stability Control

Industry-grade self-driving RL training with:
- 3-phase curriculum learning
- Comprehensive metrics (core + safety + stability)
- Formal evaluation protocol
- Automatic convergence detection
- Stability diagnostics

No camera. Pure state-based control.
"""

import argparse
from pathlib import Path
import time
import json

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from rl.envs import LaneKeepingEnv
from rl.networks import MLPActorCritic
from rl.algorithms.rollout_buffer import RolloutBuffer
from rl.utils.logger import Logger
from rl.utils.scheduler import LinearSchedule
from rl.utils.evaluator import Evaluator
from rl.utils.curriculum import CurriculumScheduler


def parse_args():
    parser = argparse.ArgumentParser(description="Train Lane-Keeping Agent - STEP 3")
    
    # Training
    parser.add_argument('--total-timesteps', type=int, default=500000)
    parser.add_argument('--n-steps', type=int, default=2048, help='Steps per rollout')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--n-epochs', type=int, default=10)
    
    # PPO Stability Controls (Step 3)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--clip-range', type=float, default=0.2)
    parser.add_argument('--entropy-coef', type=float, default=0.01)
    parser.add_argument('--value-coef', type=float, default=0.5)
    parser.add_argument('--max-grad-norm', type=float, default=0.5)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--gae-lambda', type=float, default=0.95)
    
    # Step 3: Curriculum Learning
    parser.add_argument('--curriculum', action='store_true', 
                       help='Enable 3-phase curriculum learning')
    parser.add_argument('--phase1-threshold', type=float, default=50.0)
    parser.add_argument('--phase2-threshold', type=float, default=200.0)
    
    # Step 3: Evaluation
    parser.add_argument('--eval-freq', type=int, default=10000)
    parser.add_argument('--n-eval-episodes', type=int, default=10)
    
    # Step 3: Convergence
    parser.add_argument('--auto-stop', action='store_true',
                       help='Auto-stop when converged')
    parser.add_argument('--patience', type=int, default=20,
                       help='Episodes for convergence check')
    
    # Logging
    parser.add_argument('--exp-name', type=str, default='step3_lane_keeping')
    parser.add_argument('--log-dir', type=str, default='logs')
    parser.add_argument('--save-freq', type=int, default=50000)
    
    # Device
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--seed', type=int, default=42)
    
    return parser.parse_args()


def compute_reward_with_jerk_penalty(env_reward, prev_steering, curr_steering):
    """
    Enhanced reward function (Step 3):
    R = base_reward - 0.2 * |steering_jerk|
    
    Prevents oscillation and reckless driving.
    """
    steering_jerk = abs(curr_steering - prev_steering)
    return env_reward - 0.2 * steering_jerk


def train(args):
    """Main training loop with Step 3 features."""
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() and args.device != 'cpu' else 'cpu')
    print(f"Device: {device}")
    
    # Experiment directory
    exp_dir = Path(args.log_dir) / args.exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = exp_dir / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Save config
    with open(exp_dir / 'config.json', 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, indent=2)
    
    # TensorBoard
    writer = SummaryWriter(log_dir=str(exp_dir))
    
    # Random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Environment
    env = LaneKeepingEnv(
        lane_width=3.5,
        max_speed=30.0,
        episode_steps=1000
    )
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    print("\n" + "="*60)
    print("STEP 3: Training, Evaluation, and Stability Control")
    print("="*60)
    print(f"State: {env.observation_space}")
    print(f"Action: {env.action_space}")
    print(f"Curriculum: {'ENABLED' if args.curriculum else 'DISABLED'}")
    print(f"Auto-stop: {'ENABLED' if args.auto_stop else 'DISABLED'}")
    print("="*60 + "\n")
    
    # Policy network
    policy = MLPActorCritic(
        observation_dim=obs_dim,
        action_dim=action_dim,
        hidden_dims=[256, 256]
    ).to(device)
    
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.learning_rate)
    lr_schedule = LinearSchedule(args.learning_rate, 0.0, args.total_timesteps)
    
    # Rollout buffer
    buffer = RolloutBuffer(
        buffer_size=args.n_steps,
        observation_shape=(obs_dim,),
        action_dim=action_dim,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda
    )
    
    # Step 3: Curriculum scheduler
    curriculum = None
    if args.curriculum:
        curriculum = CurriculumScheduler(
            phase1_threshold=args.phase1_threshold,
            phase2_threshold=args.phase2_threshold
        )
        print(f"📚 {curriculum.get_phase_name()}\n")
    
    # Step 3: Logger with stability tracking
    logger = Logger(log_dir=exp_dir, window_size=100)
    
    # Step 3: Evaluator
    evaluator = Evaluator(env, policy, n_episodes=args.n_eval_episodes, device=device)
    
    # Training loop
    obs, _ = env.reset(seed=args.seed)
    global_step = 0
    episode_reward = 0
    episode_length = 0
    episode_count = 0
    prev_steering = 0.0
    
    # Episode tracking
    episode_lane_offsets = []
    episode_heading_errors = []
    episode_speeds = []
    episode_action_saturations = []
    
    start_time = time.time()
    
    while global_step < args.total_timesteps:
        
        # === COLLECT ROLLOUT ===
        for _ in range(args.n_steps):
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
                action, log_prob, entropy, value = policy.get_action_and_value(obs_tensor)
                action_np = action.cpu().numpy()[0]
            
            # Step environment
            next_obs, env_reward, terminated, truncated, _ = env.step(action_np)
            done = terminated or truncated
            
            # Step 3: Enhanced reward with jerk penalty
            reward = compute_reward_with_jerk_penalty(env_reward, prev_steering, action_np[0])
            prev_steering = action_np[0]
            
            # Store transition
            buffer.add(
                obs=np.array([obs], dtype=np.float32),
                action=action.cpu().numpy(),
                reward=np.array([reward], dtype=np.float32),
                done=np.array([done], dtype=np.float32),
                value=value.cpu().numpy(),
                log_prob=log_prob.cpu().numpy()
            )
            
            episode_reward += env_reward  # Track original reward
            episode_length += 1
            global_step += 1
            
            # Track metrics
            episode_lane_offsets.append(abs(obs[0]))
            episode_heading_errors.append(abs(obs[1]))
            episode_speeds.append(obs[2])
            episode_action_saturations.append(1 if abs(action_np[0]) > 0.9 else 0)
            
            obs = next_obs
            
            if done:
                # Episode finished
                episode_count += 1
                
                # Step 3: Log comprehensive metrics
                episode_info = {
                    'episode_reward': episode_reward,
                    'episode_length': episode_length,
                    'mean_lane_deviation': np.mean(episode_lane_offsets),
                    'mean_heading_error': np.mean(episode_heading_errors),
                    'mean_speed_error': abs(np.mean(episode_speeds) - 20.0),  # Target 20m/s
                    'hard_reset': terminated,  # Crash
                    'action_saturation': np.mean(episode_action_saturations),
                }
                logger.log_episode(episode_info)
                logger.total_steps = global_step
                
                # TensorBoard
                writer.add_scalar('Episode/Reward', episode_reward, global_step)
                writer.add_scalar('Episode/Length', episode_length, global_step)
                writer.add_scalar('Metrics/LaneDeviation', episode_info['mean_lane_deviation'], global_step)
                writer.add_scalar('Metrics/HeadingError', episode_info['mean_heading_error'], global_step)
                writer.add_scalar('Safety/CrashRate', 1 if terminated else 0, global_step)
                writer.add_scalar('Safety/ActionSaturation', episode_info['action_saturation'], global_step)
                
                # Print
                print(f"Step {global_step:7d} | "
                      f"Reward: {episode_reward:8.2f} | "
                      f"Length: {episode_length:5d} | "
                      f"Lane Offset: {episode_info['mean_lane_deviation']:4.1f}")
                
                # Reset
                obs, _ = env.reset()
                episode_reward = 0
                episode_length = 0
                prev_steering = 0.0
                episode_lane_offsets = []
                episode_heading_errors = []
                episode_speeds = []
                episode_action_saturations = []
        
        # === COMPUTE GAE ===
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            next_value = policy.get_value(obs_tensor).cpu().numpy()
        
        buffer.compute_returns_and_advantages(next_value, done)
        
        # === PPO UPDATE ===
        policy_losses = []
        value_losses = []
        entropies_list = []
        clip_fractions = []
        
        for _ in range(args.n_epochs):
            for rollout_data in buffer.get(args.batch_size):
                # Move to device
                obs_batch = rollout_data.observations.to(device)
                actions_batch = rollout_data.actions.to(device)
                advantages_batch = rollout_data.advantages.to(device)
                returns_batch = rollout_data.returns.to(device)
                old_log_probs_batch = rollout_data.old_log_probs.to(device)
                
                # Normalize advantages
                advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)
                
                # Forward pass
                _, log_prob, entropy, value = policy.get_action_and_value(obs_batch, actions_batch)
                
                # Policy loss (PPO clipped objective)
                ratio = torch.exp(log_prob - old_log_probs_batch)
                surr1 = ratio * advantages_batch
                surr2 = torch.clamp(ratio, 1 - args.clip_range, 1 + args.clip_range) * advantages_batch
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = ((value - returns_batch) ** 2).mean()
                
                # Total loss
                loss = policy_loss + args.value_coef * value_loss - args.entropy_coef * entropy.mean()
                
                # Optimize
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
                optimizer.step()
                
                # Metrics
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies_list.append(entropy.mean().item())
                
                # Clip fraction (for monitoring)
                with torch.no_grad():
                    clip_frac = ((ratio - 1.0).abs() > args.clip_range).float().mean().item()
                    clip_fractions.append(clip_frac)
        
        # Step 3: Log training metrics
        train_metrics = {
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses),
            'entropy': np.mean(entropies_list),
            'clip_fraction': np.mean(clip_fractions),
        }
        logger.log_training_step(train_metrics)
        
        # TensorBoard training
        writer.add_scalar('Train/PolicyLoss', train_metrics['policy_loss'], global_step)
        writer.add_scalar('Train/ValueLoss', train_metrics['value_loss'], global_step)
        writer.add_scalar('Train/Entropy', train_metrics['entropy'], global_step)
        writer.add_scalar('Train/ClipFraction', train_metrics['clip_fraction'], global_step)
        
        # Update learning rate
        current_lr = lr_schedule(global_step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
        writer.add_scalar('Train/LearningRate', current_lr, global_step)
        
        # Clear buffer
        buffer.reset()
        
        # === STEP 3: FORMAL EVALUATION ===
        if global_step % args.eval_freq == 0:
            print(f"\n{'='*60}")
            print(f"Evaluation at step {global_step}")
            print(f"{'='*60}")
            
            eval_results = evaluator.evaluate(deterministic=True)
            
            print(f"Mean reward: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")
            print(f"Mean length: {eval_results['mean_length']:.1f}")
            print(f"Completion rate: {eval_results['completion_rate']:.1f}%")
            print(f"{'='*60}\n")
            
            # TensorBoard eval
            for key, value in eval_results.items():
                writer.add_scalar(f'Eval/{key}', value, global_step)
        
        # === STEP 3: STABILITY DIAGNOSTICS ===
        if global_step % 10000 == 0:
            warnings = logger.check_stability()
            if any(warnings.values()):
                print(f"\n⚠️  STABILITY WARNINGS:")
                for warning, active in warnings.items():
                    if active:
                        print(f"   - {warning}")
                print()
        
        # === STEP 3: CURRICULUM UPDATE ===
        if curriculum is not None and episode_count % 20 == 0:
            stats = logger.get_stats()
            if 'mean_reward' in stats:
                curriculum.update(stats['mean_reward'])
        
        # === CHECKPOINT ===
        if global_step % args.save_freq == 0:
            checkpoint_path = checkpoint_dir / f'model_{global_step}.pt'
            torch.save({
                'global_step': global_step,
                'policy_state_dict': policy.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
        
        # === STEP 3: AUTO-STOP ON CONVERGENCE ===
        if args.auto_stop and logger.is_converged(patience=args.patience):
            print(f"\n{'='*60}")
            print("✅ TRAINING CONVERGED")
            print(f"{'='*60}")
            print(f"Reward plateaued with low variance")
            print(f"No stability warnings")
            print(f"Stopping at step {global_step}")
            print(f"{'='*60}\n")
            break
    
    # === FINAL EVALUATION ===
    print(f"\n{'='*60}")
    print("FINAL EVALUATION")
    print(f"{'='*60}\n")
    final_results = evaluator.evaluate(deterministic=True, seeds=list(range(20)))
    evaluator.print_results(final_results)
    
    # Save final model
    final_path = checkpoint_dir / 'model_final.pt'
    torch.save({
        'global_step': global_step,
        'policy_state_dict': policy.state_dict(),
        'eval_results': final_results,
    }, final_path)
    print(f"\nFinal model saved: {final_path}")
    
    # Save training summary
    logger.save_summary()
    
    # Training time
    elapsed = time.time() - start_time
    print(f"Training complete! Total time: {elapsed/3600:.2f} hours")
    
    writer.close()


if __name__ == '__main__':
    args = parse_args()
    train(args)
