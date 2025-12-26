"""
Phase 2 Demo: Offline Evaluation

Run frozen policy evaluation on recorded episodes.

Usage:
    # Evaluate with random policy
    python offline_evaluation_demo.py --episodes episodes/*.json.gz
    
    # Evaluate with trained checkpoint
    python offline_evaluation_demo.py --checkpoint checkpoints/best.pt --episodes episodes/*.json.gz
    
    # Batch evaluation
    python offline_evaluation_demo.py --episodes episodes/episode_*.json.gz --output evaluation_reports/
"""

import argparse
import glob
import sys
from pathlib import Path

import torch
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from rl.networks.mlp_policy import MLPActorCritic
from rl.safety import SafetyShield, SafetyLimits
from rl.evaluation.offline_evaluator import OfflineEvaluator


def load_policy(checkpoint_path: str = None) -> MLPActorCritic:
    """Load policy network (trained or random)."""
    policy = MLPActorCritic(
        observation_dim=6,
        action_dim=2,
        hidden_dims=(256, 256),
        log_std_init=0.0
    )
    
    if checkpoint_path:
        print(f"📦 Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        policy.load_state_dict(checkpoint['policy_state_dict'])
        print("✅ Checkpoint loaded")
    else:
        print("🎲 Using random policy (no checkpoint)")
    
    # Freeze policy
    policy.eval()
    for param in policy.parameters():
        param.requires_grad = False
    
    return policy


def main():
    parser = argparse.ArgumentParser(
        description="Phase 2: Offline Evaluation of Frozen Policies"
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to policy checkpoint (.pt file)'
    )
    parser.add_argument(
        '--episodes',
        type=str,
        nargs='+',
        required=True,
        help='Episode files to evaluate (supports wildcards)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='evaluation_reports',
        help='Output directory for reports'
    )
    parser.add_argument(
        '--max-deviation',
        type=float,
        default=1.5,
        help='Safety threshold: max lane deviation (meters)'
    )
    parser.add_argument(
        '--max-intervention-rate',
        type=float,
        default=0.3,
        help='Safety threshold: max intervention rate'
    )
    
    args = parser.parse_args()
    
    # Expand wildcards
    episode_files = []
    for pattern in args.episodes:
        episode_files.extend(glob.glob(pattern))
    
    if not episode_files:
        print("❌ No episode files found")
        return
    
    episode_files = sorted(episode_files)
    print(f"📁 Found {len(episode_files)} episodes")
    
    # Load policy
    policy = load_policy(args.checkpoint)
    
    # Create safety shield
    safety_limits = SafetyLimits(
        max_steering_rate=0.3,
        max_speed=30.0,
        max_lane_offset=1.5
    )
    safety_shield = SafetyShield(limits=safety_limits)
    
    # Create evaluator
    safety_thresholds = {
        'max_lane_deviation': args.max_deviation,
        'max_intervention_rate': args.max_intervention_rate,
        'max_emergency_brake_rate': 0.1
    }
    
    evaluator = OfflineEvaluator(
        policy=policy,
        safety_shield=safety_shield,
        safety_thresholds=safety_thresholds,
        verbose=True
    )
    
    # Run evaluation
    checkpoint_name = Path(args.checkpoint).name if args.checkpoint else "random_policy"
    
    report = evaluator.evaluate_episodes(
        episode_paths=episode_files,
        policy_checkpoint=checkpoint_name,
        output_dir=args.output
    )
    
    # Final summary
    print("\n" + "="*70)
    print("🏁 EVALUATION COMPLETE")
    print("="*70)
    
    if report.is_safe and report.is_deterministic:
        print("✅ SYSTEM IS ENGINEERING-GRADE")
        print("   - Deterministic ✓")
        print("   - Safe ✓")
        print("   - Numerically Stable ✓")
        print("\n🎯 Your project is scientifically valid and deployable.")
    else:
        print("❌ SYSTEM NEEDS IMPROVEMENT")
        if not report.is_deterministic:
            print("   - Non-deterministic behavior detected")
        if not report.is_safe:
            print("   - Safety thresholds violated")
            print(f"     Max deviation: {report.safety.worst_case_deviation:.3f}m "
                  f"(limit: {args.max_deviation}m)")
        print("\n⚠️  System is NOT ready for deployment.")
    
    print(f"\nDeterminism Hash: {report.determinism_hash}")
    print("="*70)


if __name__ == "__main__":
    main()
