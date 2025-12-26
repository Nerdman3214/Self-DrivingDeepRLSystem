"""
PHASE 1: Episode Recording and Playback Demo

Demonstrates deterministic replay of autonomous driving episodes.

Usage:
    python record_and_playback.py --record
    python record_and_playback.py --playback episodes/episode_0000_*.json.gz
    python record_and_playback.py --compare episodes/*.json.gz
"""

import argparse
import sys
from pathlib import Path
import torch
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from rl.envs.lane_keeping_env import LaneKeepingEnv
from rl.networks.mlp_policy import MLPActorCritic
from rl.safety import SafetyShield
from rl.utils.episode_recorder import EpisodeRecorder, record_episode
from rl.utils.episode_playback import EpisodePlayback, compare_episodes, PlaybackConfig


def demo_recording(checkpoint_path: str = None, num_episodes: int = 3):
    """
    Demonstrate episode recording.
    
    Records episodes with policy + safety shield for later analysis.
    """
    print("\n" + "="*70)
    print("🎬 PHASE 1: EPISODE RECORDING DEMO")
    print("="*70)
    print("\nThis demonstrates Event Sourcing for autonomous driving.")
    print("Every decision is recorded for deterministic replay.\n")
    
    # Create environment
    env = LaneKeepingEnv(
        lane_width=3.5,
        max_episode_steps=500,
        render_mode=None  # No GUI needed
    )
    
    # Load or create policy
    if checkpoint_path and Path(checkpoint_path).exists():
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        policy = MLPActorCritic(
            observation_dim=6,
            action_dim=2,
            hidden_dims=(256, 256)
        )
        policy.load_state_dict(checkpoint['policy_state_dict'])
        policy.eval()
    else:
        print("⚠️  No checkpoint provided, using random policy for demonstration")
        policy = MLPActorCritic(
            observation_dim=6,
            action_dim=2,
            hidden_dims=(256, 256)
        )
        policy.eval()
    
    # Create safety shield
    from rl.safety import SafetyShield, SafetyLimits
    safety_shield = SafetyShield(
        limits=SafetyLimits(
            max_steering_rate=0.3,
            max_speed=30.0,
            max_lane_offset=1.5
        )
    )
    
    # Record episodes
    print(f"\n🎥 Recording {num_episodes} episodes...")
    print("Each episode is a complete timeline of:")
    print("  • State observations")
    print("  • Policy decisions")
    print("  • Safety interventions")
    print("  • Rewards and outcomes\n")
    
    for i in range(num_episodes):
        episode_data = record_episode(
            env=env,
            policy=policy,
            safety_shield=safety_shield,
            episode_id=i,
            max_steps=500,
            output_dir="episodes",
            verbose=True
        )
        
        print()
    
    print("=" * 70)
    print("✅ Recording complete!")
    print(f"Episodes saved to: episodes/")
    print("=" * 70)


def demo_playback(episode_path: str):
    """
    Demonstrate episode playback and analysis.
    
    Shows what the agent was doing and why.
    """
    print("\n" + "="*70)
    print("📼 PHASE 1: EPISODE PLAYBACK DEMO")
    print("="*70)
    print("\nThis demonstrates deterministic replay and behavior analysis.")
    print("NO randomness. NO simulation. Just pure event replay.\n")
    
    # Load and analyze episode
    config = PlaybackConfig(
        show_plots=True,
        save_plots=True,
        verbose=True
    )
    
    playback = EpisodePlayback(episode_path, config=config)
    
    # Step-by-step replay (first 10 steps)
    print("\n" + "-" * 70)
    print("First 10 timesteps (step-by-step):")
    print("-" * 70)
    playback.replay_step_by_step(start=0, end=10)
    
    # Behavior analysis
    print("\n" + "-" * 70)
    print("Running behavior analysis...")
    print("-" * 70)
    analysis = playback.analyze_behavior()
    
    # Explain specific actions
    print("\n" + "-" * 70)
    print("Explaining specific timesteps:")
    print("-" * 70)
    
    # Explain timestep with safety intervention
    events = playback.events
    intervention_timesteps = [
        i for i, e in enumerate(events)
        if any(e["safety_flags"].values())
    ]
    
    if intervention_timesteps:
        print(f"\nExplaining timestep with safety intervention:")
        playback.explain_action(intervention_timesteps[0])
    
    # Plot full episode
    print("\n" + "-" * 70)
    print("Creating comprehensive visualization...")
    print("-" * 70)
    playback.plot_episode(save=True)
    
    print("\n" + "=" * 70)
    print("✅ Playback analysis complete!")
    print("=" * 70)
    
    # Key insights
    print("\n🔍 KEY INSIGHTS FROM PLAYBACK:")
    print("-" * 70)
    stats = playback.metadata["statistics"]
    
    print(f"✔ Total Reward: {stats['total_reward']:.2f}")
    print(f"✔ Safety intervened {stats['safety_interventions']} times ({stats['safety_intervention_rate']:.2%})")
    print(f"✔ Max lane deviation: {stats['state_stats']['max_lane_deviation']:.3f}m")
    print(f"✔ Steering smoothness (mean jerk): {analysis['steering_smoothness']['mean_jerk']:.4f}")
    
    if analysis['oscillation_detected']:
        print("⚠️  Oscillation detected - policy may need tuning")
    else:
        print("✔ No oscillation detected")
    
    disagreement_rate = analysis['policy_safety_disagreement']['disagreement_rate']
    if disagreement_rate > 0.2:
        print(f"⚠️  High policy-safety disagreement ({disagreement_rate:.2%}) - policy may be unsafe")
    else:
        print(f"✔ Low policy-safety disagreement ({disagreement_rate:.2%})")
    
    print("=" * 70)


def demo_comparison(episode_pattern: str):
    """
    Compare multiple episodes.
    
    Args:
        episode_pattern: Glob pattern for episode files
    """
    print("\n" + "="*70)
    print("📊 EPISODE COMPARISON DEMO")
    print("="*70)
    
    episode_files = list(Path(".").glob(episode_pattern))
    
    if not episode_files:
        print(f"❌ No episodes found matching: {episode_pattern}")
        return
    
    print(f"\nFound {len(episode_files)} episodes to compare\n")
    
    compare_episodes([str(f) for f in episode_files])
    
    print("\n✅ Comparison complete!")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Phase 1: Episode Recording and Playback System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Record 5 episodes with trained policy
    python record_and_playback.py --record --checkpoint checkpoints/best.pt --episodes 5
    
    # Record episodes with random policy (demo)
    python record_and_playback.py --record --episodes 3
    
    # Playback specific episode
    python record_and_playback.py --playback episodes/episode_0000_*.json.gz
    
    # Compare all episodes
    python record_and_playback.py --compare "episodes/*.json.gz"
    
Phase 1 Completion Criteria:
    ✔ Can run inference
    ✔ Episode file is produced
    ✔ Can replay deterministically
    ✔ Can explain every action
        """
    )
    
    parser.add_argument(
        "--record",
        action="store_true",
        help="Record new episodes"
    )
    
    parser.add_argument(
        "--playback",
        type=str,
        help="Playback episode file"
    )
    
    parser.add_argument(
        "--compare",
        type=str,
        help="Compare multiple episodes (glob pattern)"
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to policy checkpoint (for recording)"
    )
    
    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="Number of episodes to record (default: 3)"
    )
    
    args = parser.parse_args()
    
    # Execute requested operation
    if args.record:
        demo_recording(
            checkpoint_path=args.checkpoint,
            num_episodes=args.episodes
        )
    elif args.playback:
        demo_playback(args.playback)
    elif args.compare:
        demo_comparison(args.compare)
    else:
        parser.print_help()
        print("\n" + "="*70)
        print("💡 TIP: Start with --record to create episodes, then --playback to analyze")
        print("="*70)


if __name__ == "__main__":
    main()
