#!/usr/bin/env python3
"""
Self-Driving RL Training Script

Train a PPO agent on CarRacing-v2 environment.

Usage:
    python train_self_driving.py --total-timesteps 1000000
    python train_self_driving.py --resume checkpoints/model_500000.pt
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rl.networks.actor_critic import ActorCritic
from rl.algorithms.ppo import PPO
from rl.envs.wrappers import make_car_racing_env
from rl.envs.vec_env import make_vec_env
from rl.utils.helpers import set_seed, get_device
from rl.utils.logger import Logger


def parse_args():
    parser = argparse.ArgumentParser(description="Train Self-Driving RL Agent")
    
    # Environment
    parser.add_argument("--env-id", type=str, default="CarRacing-v2",
                        help="Environment ID")
    parser.add_argument("--n-envs", type=int, default=1,
                        help="Number of parallel environments")
    parser.add_argument("--frame-skip", type=int, default=4,
                        help="Number of frames to skip")
    parser.add_argument("--grayscale", action="store_true",
                        help="Use grayscale observations")
    
    # Training
    parser.add_argument("--total-timesteps", type=int, default=1000000,
                        help="Total timesteps for training")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--n-steps", type=int, default=2048,
                        help="Number of steps per rollout")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Mini-batch size")
    parser.add_argument("--n-epochs", type=int, default=10,
                        help="Number of epochs per update")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
                        help="GAE lambda")
    parser.add_argument("--clip-range", type=float, default=0.2,
                        help="PPO clip range")
    parser.add_argument("--entropy-coef", type=float, default=0.01,
                        help="Entropy coefficient")
    parser.add_argument("--value-coef", type=float, default=0.5,
                        help="Value function coefficient")
    
    # Network
    parser.add_argument("--feature-dim", type=int, default=512,
                        help="Feature dimension")
    
    # Logging
    parser.add_argument("--log-dir", type=str, default="logs",
                        help="Log directory")
    parser.add_argument("--exp-name", type=str, default="self_driving_ppo",
                        help="Experiment name")
    parser.add_argument("--log-interval", type=int, default=1,
                        help="Log every N updates")
    parser.add_argument("--save-freq", type=int, default=50000,
                        help="Save checkpoint every N timesteps")
    
    # Evaluation
    parser.add_argument("--eval-freq", type=int, default=50000,
                        help="Evaluate every N timesteps (-1 to disable)")
    parser.add_argument("--n-eval-episodes", type=int, default=5,
                        help="Number of evaluation episodes")
    
    # Other
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cpu, cuda, cuda:0, etc.)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume training from checkpoint")
    parser.add_argument("--render", action="store_true",
                        help="Render environment during training")
    
    return parser.parse_args()


def create_env(args, render: bool = False):
    """Create the training environment."""
    render_mode = "human" if render else None
    
    def env_fn():
        return make_car_racing_env(
            render_mode=render_mode,
            grayscale=args.grayscale,
            frame_skip=args.frame_skip,
            resize_shape=(96, 96)
        )
    
    if args.n_envs == 1:
        return env_fn()
    else:
        return make_vec_env(env_fn, n_envs=args.n_envs, use_subprocess=True)


def main():
    args = parse_args()
    
    # Set up directories
    log_dir = Path(args.log_dir) / args.exp_name
    checkpoint_dir = log_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Set random seed
    set_seed(args.seed)
    
    # Get device
    device = get_device(args.device)
    print(f"\n{'='*60}")
    print("Self-Driving RL Training")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Environment: {args.env_id}")
    print(f"Total timesteps: {args.total_timesteps:,}")
    print(f"{'='*60}\n")
    
    # Create environment
    print("Creating environment...")
    env = create_env(args, render=args.render)
    
    # Create evaluation environment (separate instance)
    eval_env = create_env(args, render=False) if args.eval_freq > 0 else None
    
    # Get observation and action dimensions
    n_channels = 1 if args.grayscale else 3
    obs_height, obs_width = 96, 96
    action_dim = 3  # steering, gas, brake
    
    print(f"Observation shape: ({n_channels}, {obs_height}, {obs_width})")
    print(f"Action dimension: {action_dim}")
    
    # Create policy network
    print("\nCreating policy network...")
    policy = ActorCritic(
        input_channels=n_channels,
        input_height=obs_height,
        input_width=obs_width,
        action_dim=action_dim,
        feature_dim=args.feature_dim
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in policy.parameters())
    trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create PPO algorithm
    print("\nInitializing PPO...")
    ppo = PPO(
        policy=policy,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        device=device,
        n_envs=args.n_envs
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"\nResuming from: {args.resume}")
        ppo.load(args.resume)
    
    # Create logger
    logger = Logger(
        log_dir=str(log_dir),
        experiment_name=args.exp_name,
        use_tensorboard=True,
        verbose=1
    )
    
    # Training callback for saving checkpoints
    last_save_timesteps = 0
    
    def callback(_locals_dict, _globals_dict):
        nonlocal last_save_timesteps
        timesteps = ppo.num_timesteps
        
        # Save checkpoint
        if timesteps - last_save_timesteps >= args.save_freq:
            checkpoint_path = checkpoint_dir / f"model_{timesteps}.pt"
            ppo.save(str(checkpoint_path))
            last_save_timesteps = timesteps
        
        return True
    # Train
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60 + "\n")
    
    try:
        ppo.learn(
            env=env,
            total_timesteps=args.total_timesteps,
            callback=callback,
            log_interval=args.log_interval,
            eval_env=eval_env,
            eval_freq=args.eval_freq,
            n_eval_episodes=args.n_eval_episodes
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
    
    # Save final model
    final_path = checkpoint_dir / "model_final.pt"
    ppo.save(str(final_path))
    print(f"\nFinal model saved to: {final_path}")
    
    # Clean up
    logger.close()
    env.close()
    if eval_env is not None:
        eval_env.close()
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
