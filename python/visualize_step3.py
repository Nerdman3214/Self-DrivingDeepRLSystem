#!/usr/bin/env python3
"""
Visualization Tools for Step 3

Even without camera, we visualize:
- Training curves (reward, deviation, entropy)
- Control signals (steering, throttle)
- Safety metrics (crashes, saturation)
- Stability diagnostics
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json
from typing import List, Dict
import argparse


def plot_training_curves(log_dir: Path, output_dir: Path = None):
    """
    Plot comprehensive training curves.
    
    Creates:
    1. Reward vs episodes
    2. Lane deviation vs time
    3. Entropy vs training steps
    4. Value loss vs training steps
    """
    if output_dir is None:
        output_dir = log_dir / 'plots'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load training summary
    summary_file = log_dir / 'training_summary.json'
    if not summary_file.exists():
        print(f"⚠️  No training summary found at {summary_file}")
        return
    
    with open(summary_file, 'r', encoding='utf-8') as f:
        summary = json.load(f)
    
    print(f"Loaded training summary:")
    print(f"  Total episodes: {summary['total_episodes']}")
    print(f"  Total steps: {summary['total_steps']}")
    print(f"  Converged: {summary['converged']}")
    print()
    
    # Plot summary
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Step 3: Training Summary', fontsize=16)
    
    # Final stats
    stats = summary['final_stats']
    
    # Subplot 1: Core metrics
    ax = axes[0, 0]
    metrics = ['mean_reward', 'mean_lane_deviation', 'mean_heading_error']
    values = [stats.get(m, 0) for m in metrics]
    labels = ['Reward', 'Lane Dev (m)', 'Heading Err (rad)']
    ax.bar(labels, values, color=['green', 'orange', 'red'])
    ax.set_title('Final Performance Metrics')
    ax.grid(True, alpha=0.3)
    
    # Subplot 2: Safety metrics
    ax = axes[0, 1]
    safety_metrics = ['crash_rate', 'action_saturation']
    safety_values = [stats.get(m, 0) * 100 for m in safety_metrics]  # Percentage
    safety_labels = ['Crash Rate (%)', 'Action Sat (%)']
    colors = ['red' if v > 50 else 'yellow' if v > 20 else 'green' for v in safety_values]
    ax.bar(safety_labels, safety_values, color=colors)
    ax.set_title('Safety Metrics')
    ax.set_ylim(0, 100)
    ax.axhline(y=20, color='orange', linestyle='--', alpha=0.5, label='Warning')
    ax.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='Danger')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Subplot 3: PPO stability
    ax = axes[1, 0]
    ppo_metrics = ['entropy', 'value_loss', 'clip_fraction']
    ppo_values = [stats.get(m, 0) for m in ppo_metrics]
    ppo_labels = ['Entropy', 'Value Loss', 'Clip Frac']
    ax.bar(ppo_labels, ppo_values, color=['blue', 'purple', 'cyan'])
    ax.set_title('PPO Stability Indicators')
    ax.grid(True, alpha=0.3)
    
    # Subplot 4: Stability warnings
    ax = axes[1, 1]
    warnings = summary['stability_warnings']
    warning_names = list(warnings.keys())
    warning_values = [1 if warnings[w] else 0 for w in warning_names]
    colors = ['red' if v else 'green' for v in warning_values]
    ax.barh(warning_names, warning_values, color=colors)
    ax.set_xlim(0, 1.2)
    ax.set_title('Stability Warnings')
    ax.set_xlabel('Active (Red = Warning)')
    
    plt.tight_layout()
    output_file = output_dir / 'training_summary.png'
    plt.savefig(output_file, dpi=150)
    print(f"✅ Saved: {output_file}")
    plt.close()


def plot_control_signals(log_dir: Path, output_dir: Path = None):
    """
    Plot control signal analysis.
    
    Visualizes:
    - Steering distribution
    - Throttle distribution
    - Action saturation over time
    """
    if output_dir is None:
        output_dir = log_dir / 'plots'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("📊 Control signals plot would show:")
    print("   - Steering angle histogram")
    print("   - Throttle distribution")
    print("   - Action saturation timeline")
    print("   (Requires episode-by-episode logs)")


def print_diagnostics(log_dir: Path):
    """Print comprehensive diagnostics."""
    summary_file = log_dir / 'training_summary.json'
    
    if not summary_file.exists():
        print(f"❌ No summary found at {summary_file}")
        return
    
    with open(summary_file, 'r', encoding='utf-8') as f:
        summary = json.load(f)
    
    print("="*60)
    print("STEP 3 TRAINING DIAGNOSTICS")
    print("="*60)
    print()
    
    # Overview
    print(f"Episodes: {summary['total_episodes']}")
    print(f"Steps: {summary['total_steps']}")
    print(f"Converged: {'✅ YES' if summary['converged'] else '❌ NO'}")
    print()
    
    # Performance
    stats = summary['final_stats']
    print("Performance:")
    print(f"  Mean Reward: {stats.get('mean_reward', 0):.2f} ± {stats.get('std_reward', 0):.2f}")
    print(f"  Best Reward: {summary.get('best_reward', 0):.2f}")
    print(f"  Worst Reward: {summary.get('worst_reward', 0):.2f}")
    print()
    
    # Driving quality
    print("Driving Quality:")
    print(f"  Lane Deviation: {stats.get('mean_lane_deviation', 0):.3f}m")
    print(f"  Heading Error: {stats.get('mean_heading_error', 0):.3f}rad")
    print(f"  Speed Error: {stats.get('mean_speed_error', 0):.3f}m/s")
    print()
    
    # Safety
    print("Safety:")
    crash_rate = stats.get('crash_rate', 0) * 100
    print(f"  Crash Rate: {crash_rate:.1f}%")
    action_sat = stats.get('action_saturation', 0) * 100
    print(f"  Action Saturation: {action_sat:.1f}%")
    print()
    
    # PPO health
    print("PPO Health:")
    print(f"  Entropy: {stats.get('entropy', 0):.4f}")
    print(f"  Value Loss: {stats.get('value_loss', 0):.4f}")
    print(f"  Clip Fraction: {stats.get('clip_fraction', 0):.4f}")
    print()
    
    # Warnings
    warnings = summary['stability_warnings']
    active_warnings = [w for w, active in warnings.items() if active]
    
    if active_warnings:
        print("⚠️  Stability Warnings:")
        for warning in active_warnings:
            print(f"   - {warning}")
    else:
        print("✅ No Stability Warnings")
    
    print()
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Visualize Step 3 Training Results")
    parser.add_argument('--log-dir', type=str, required=True,
                       help='Path to experiment log directory')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for plots (default: log_dir/plots)')
    parser.add_argument('--diagnostics-only', action='store_true',
                       help='Only print diagnostics, no plots')
    
    args = parser.parse_args()
    
    log_dir = Path(args.log_dir)
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    if not log_dir.exists():
        print(f"❌ Log directory not found: {log_dir}")
        return
    
    # Print diagnostics
    print_diagnostics(log_dir)
    
    # Generate plots
    if not args.diagnostics_only:
        print("\nGenerating plots...")
        plot_training_curves(log_dir, output_dir)
        print("✅ Visualization complete")


if __name__ == '__main__':
    main()
