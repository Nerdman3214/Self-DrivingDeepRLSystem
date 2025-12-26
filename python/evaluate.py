#!/usr/bin/env python3
"""
Evaluate a trained self-driving agent.

Usage:
    python evaluate.py --checkpoint checkpoints/model_final.pt
    python evaluate.py --checkpoint model.pt --n-episodes 100 --render
"""

import argparse
import sys
from pathlib import Path
import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from rl.networks.actor_critic import ActorCritic
from rl.envs.wrappers import make_car_racing_env
from rl.utils.helpers import get_device


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Self-Driving Agent")
    
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--n-episodes", type=int, default=10,
                        help="Number of evaluation episodes")
    parser.add_argument("--deterministic", action="store_true",
                        help="Use deterministic actions")
    parser.add_argument("--render", action="store_true",
                        help="Render environment")
    parser.add_argument("--save-video", action="store_true",
                        help="Save video of episodes")
    parser.add_argument("--video-dir", type=str, default="videos",
                        help="Directory to save videos")
    
    # Model architecture
    parser.add_argument("--input-channels", type=int, default=3)
    parser.add_argument("--input-height", type=int, default=96)
    parser.add_argument("--input-width", type=int, default=96)
    parser.add_argument("--action-dim", type=int, default=3)
    parser.add_argument("--feature-dim", type=int, default=512)
    
    # Environment
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--grayscale", action="store_true")
    
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    
    return parser.parse_args()


def evaluate(model, env, n_episodes, deterministic, device, verbose=True):
    """Evaluate the model."""
    model.eval()
    
    episode_rewards = []
    episode_lengths = []
    
    for ep in range(n_episodes):
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        
        done = False
        total_reward = 0.0
        length = 0
        
        while not done:
            # Prepare observation
            obs_tensor = torch.from_numpy(obs).float().to(device)
            if obs_tensor.dim() == 3:
                obs_tensor = obs_tensor.unsqueeze(0)
            
            # Get action
            with torch.no_grad():
                action, _ = model.get_action(obs_tensor, deterministic=deterministic)
            
            action_np = action.cpu().numpy()[0]
            
            # Rescale actions for CarRacing
            action_np = np.clip(action_np, -1.0, 1.0)
            action_np[1] = (action_np[1] + 1.0) / 2.0  # gas
            action_np[2] = (action_np[2] + 1.0) / 2.0  # brake
            
            # Step
            step_result = env.step(action_np)
            if len(step_result) == 5:
                obs, reward, terminated, truncated, _info = step_result
                done = terminated or truncated
            else:
                obs, reward, done, _info = step_result
            
            total_reward += reward
            length += 1
        
        episode_rewards.append(total_reward)
        episode_lengths.append(length)
        
        if verbose:
            print(f"Episode {ep + 1}/{n_episodes}: Reward = {total_reward:.2f}, Length = {length}")
    
    return np.array(episode_rewards), np.array(episode_lengths)


def main():
    args = parse_args()
    device = get_device(args.device)
    
    print(f"\n{'='*60}")
    print("Self-Driving Agent Evaluation")
    print(f"{'='*60}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {device}")
    print(f"Episodes: {args.n_episodes}")
    print(f"Deterministic: {args.deterministic}")
    
    # Load model
    print("\nLoading model...")
    model = ActorCritic(
        input_channels=args.input_channels,
        input_height=args.input_height,
        input_width=args.input_width,
        action_dim=args.action_dim,
        feature_dim=args.feature_dim
    )
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if 'policy_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['policy_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    # Create environment
    print("Creating environment...")
    render_mode = "human" if args.render else None
    env = make_car_racing_env(
        render_mode=render_mode,
        grayscale=args.grayscale,
        frame_skip=args.frame_skip,
        resize_shape=(args.input_height, args.input_width)
    )
    
    if args.seed is not None:
        env.seed(args.seed)
    
    # Evaluate
    print("\nEvaluating...")
    rewards, lengths = evaluate(
        model, env, args.n_episodes, args.deterministic, device
    )
    
    # Print results
    print(f"\n{'='*60}")
    print("Results")
    print(f"{'='*60}")
    print(f"Mean Reward: {np.mean(rewards):.2f} +/- {np.std(rewards):.2f}")
    print(f"Min Reward: {np.min(rewards):.2f}")
    print(f"Max Reward: {np.max(rewards):.2f}")
    print(f"Mean Length: {np.mean(lengths):.1f}")
    
    env.close()


if __name__ == "__main__":
    main()
