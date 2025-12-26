"""
Episode Playback Engine

Deterministic replay of recorded episodes for debugging and analysis.

This answers the question: "What is my agent actually doing over time?"

Design Patterns:
- Separation of Concerns: Playback is independent from recording
- Strategy Pattern: Different analysis strategies
"""

import json
import gzip
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass
class PlaybackConfig:
    """Configuration for episode playback."""
    show_plots: bool = True
    save_plots: bool = False
    output_dir: str = "playback_analysis"
    verbose: bool = True


class EpisodePlayback:
    """
    Deterministic playback engine for recorded episodes.
    
    Enables:
    - Step-by-step replay
    - Behavior analysis
    - Safety auditing
    - Debugging policy decisions
    """
    
    def __init__(self, episode_path: str, config: Optional[PlaybackConfig] = None):
        """
        Initialize playback engine.
        
        Args:
            episode_path: Path to recorded episode file
            config: Playback configuration
        """
        self.episode_path = Path(episode_path)
        self.config = config or PlaybackConfig()
        
        # Load episode
        self.episode_data = self._load_episode()
        self.metadata = self.episode_data["metadata"]
        self.events = self.episode_data["events"]
        
        if self.config.verbose:
            self._print_summary()
    
    def _load_episode(self) -> Dict[str, Any]:
        """Load episode from disk."""
        if self.episode_path.suffix == '.gz':
            with gzip.open(self.episode_path, 'rt', encoding='utf-8') as f:
                return json.load(f)
        else:
            with open(self.episode_path, 'r', encoding='utf-8') as f:
                return json.load(f)
    
    def _print_summary(self):
        """Print episode summary."""
        print("=" * 70)
        print("📼 EPISODE PLAYBACK")
        print("=" * 70)
        print(f"Episode ID: {self.metadata['episode_id']}")
        print(f"Total Timesteps: {self.metadata['total_timesteps']}")
        print(f"Total Reward: {self.metadata['statistics']['total_reward']:.2f}")
        print(f"Average Reward: {self.metadata['statistics']['avg_reward']:.4f}")
        print(f"Safety Interventions: {self.metadata['statistics']['safety_interventions']}")
        print(f"Intervention Rate: {self.metadata['statistics']['safety_intervention_rate']:.2%}")
        print("-" * 70)
        
        # Safety breakdown
        breakdown = self.metadata['statistics']['safety_breakdown']
        print("Safety Intervention Breakdown:")
        for intervention, count in breakdown.items():
            if count > 0:
                print(f"  {intervention}: {count}")
        print("=" * 70)
    
    def replay_step_by_step(self, start: int = 0, end: Optional[int] = None):
        """
        Replay episode step by step with human-readable output.
        
        Args:
            start: Starting timestep
            end: Ending timestep (None = all)
        """
        end = end or len(self.events)
        
        print("\n" + "=" * 70)
        print("STEP-BY-STEP REPLAY")
        print("=" * 70)
        
        for event in self.events[start:end]:
            self._print_event(event)
    
    def _print_event(self, event: Dict[str, Any]):
        """Print single event in human-readable format."""
        t = event["timestep"]
        state = event["state"]
        policy = event["policy_action"]
        safety = event["safety_action"]
        flags = event["safety_flags"]
        
        print(f"\n⏱️  Timestep {t}")
        print(f"State:")
        print(f"  Lane Offset: {state['lane_offset']:>7.3f}m  |  Heading: {state['heading_error']:>7.3f} rad")
        print(f"  Speed:       {state['speed']:>7.3f} m/s |  Curvature: {state['curvature']:>7.3f}")
        
        print(f"Actions:")
        print(f"  Policy → Steering: {policy['steering']:>6.3f}  Throttle: {policy['throttle']:>6.3f}")
        print(f"  Safety → Steering: {safety['steering']:>6.3f}  Throttle: {safety['throttle']:>6.3f}")
        
        # Highlight interventions
        interventions = [k for k, v in flags.items() if v]
        if interventions:
            print(f"🛡️  Safety Interventions: {', '.join(interventions)}")
        
        print(f"Reward: {event['reward']:.4f}")
        
        if event['done']:
            print("🏁 Episode Complete")
    
    def analyze_behavior(self) -> Dict[str, Any]:
        """
        Analyze episode behavior patterns.
        
        Returns:
            Analysis results
        """
        # Extract time series
        timesteps = [e["timestep"] for e in self.events]
        
        lane_offsets = [e["state"]["lane_offset"] for e in self.events]
        heading_errors = [e["state"]["heading_error"] for e in self.events]
        speeds = [e["state"]["speed"] for e in self.events]
        
        policy_steering = [e["policy_action"]["steering"] for e in self.events]
        safety_steering = [e["safety_action"]["steering"] for e in self.events]
        
        policy_throttle = [e["policy_action"]["throttle"] for e in self.events]
        safety_throttle = [e["safety_action"]["throttle"] for e in self.events]
        
        rewards = [e["reward"] for e in self.events]
        
        # Detect issues
        analysis = {
            "steering_smoothness": self._analyze_smoothness(safety_steering),
            "oscillation_detected": self._detect_oscillation(lane_offsets),
            "policy_safety_disagreement": self._analyze_disagreement(
                policy_steering, safety_steering
            ),
            "action_saturation": self._detect_saturation(safety_steering),
            "reward_hacking": self._detect_reward_hacking(rewards)
        }
        
        if self.config.verbose:
            self._print_analysis(analysis)
        
        return analysis
    
    def _analyze_smoothness(self, actions: List[float]) -> Dict[str, float]:
        """Analyze action smoothness (detect jerkiness)."""
        actions_arr = np.array(actions)
        jerk = np.diff(actions_arr)
        
        return {
            "mean_jerk": float(np.abs(jerk).mean()),
            "max_jerk": float(np.abs(jerk).max()),
            "std_jerk": float(jerk.std())
        }
    
    def _detect_oscillation(self, signal: List[float], threshold: float = 0.1) -> bool:
        """Detect oscillation in signal."""
        signal_arr = np.array(signal)
        sign_changes = np.sum(np.diff(np.sign(signal_arr)) != 0)
        oscillation_rate = sign_changes / len(signal)
        
        return oscillation_rate > threshold
    
    def _analyze_disagreement(
        self,
        policy_actions: List[float],
        safety_actions: List[float]
    ) -> Dict[str, float]:
        """Analyze policy vs safety disagreement."""
        policy_arr = np.array(policy_actions)
        safety_arr = np.array(safety_actions)
        
        diff = np.abs(policy_arr - safety_arr)
        disagreements = np.sum(diff > 0.01)  # 1% threshold
        
        return {
            "disagreement_rate": float(disagreements / len(policy_actions)),
            "mean_diff": float(diff.mean()),
            "max_diff": float(diff.max())
        }
    
    def _detect_saturation(self, actions: List[float]) -> Dict[str, Any]:
        """Detect action saturation at limits."""
        actions_arr = np.array(actions)
        saturated = np.sum((np.abs(actions_arr) > 0.99))
        
        return {
            "saturation_count": int(saturated),
            "saturation_rate": float(saturated / len(actions))
        }
    
    def _detect_reward_hacking(self, rewards: List[float]) -> Dict[str, bool]:
        """Detect reward hacking patterns."""
        rewards_arr = np.array(rewards)
        
        # Sudden spikes
        diffs = np.diff(rewards_arr)
        sudden_spikes = np.any(np.abs(diffs) > 10 * np.std(diffs))
        
        return {
            "sudden_spikes_detected": bool(sudden_spikes)
        }
    
    def _print_analysis(self, analysis: Dict[str, Any]):
        """Print analysis results."""
        print("\n" + "=" * 70)
        print("📊 BEHAVIOR ANALYSIS")
        print("=" * 70)
        
        print("\n🎯 Steering Smoothness:")
        smoothness = analysis["steering_smoothness"]
        print(f"  Mean Jerk: {smoothness['mean_jerk']:.4f}")
        print(f"  Max Jerk:  {smoothness['max_jerk']:.4f}")
        
        print("\n🌊 Oscillation Detection:")
        print(f"  Oscillating: {'YES ⚠️' if analysis['oscillation_detected'] else 'NO ✅'}")
        
        print("\n🛡️  Policy vs Safety Disagreement:")
        disagreement = analysis["policy_safety_disagreement"]
        print(f"  Disagreement Rate: {disagreement['disagreement_rate']:.2%}")
        print(f"  Mean Difference: {disagreement['mean_diff']:.4f}")
        
        print("\n⚠️  Action Saturation:")
        saturation = analysis["action_saturation"]
        print(f"  Saturation Rate: {saturation['saturation_rate']:.2%}")
        
        print("=" * 70)
    
    def plot_episode(self, save: bool = False):
        """
        Create comprehensive visualization of episode.
        
        Args:
            save: Whether to save plot to disk
        """
        # Extract time series
        timesteps = [e["timestep"] for e in self.events]
        
        lane_offsets = [e["state"]["lane_offset"] for e in self.events]
        heading_errors = [e["state"]["heading_error"] for e in self.events]
        speeds = [e["state"]["speed"] for e in self.events]
        
        policy_steering = [e["policy_action"]["steering"] for e in self.events]
        safety_steering = [e["safety_action"]["steering"] for e in self.events]
        
        policy_throttle = [e["policy_action"]["throttle"] for e in self.events]
        safety_throttle = [e["safety_action"]["throttle"] for e in self.events]
        
        rewards = [e["reward"] for e in self.events]
        
        # Extract safety interventions
        interventions = {
            "emergency_brake": [e["safety_flags"]["emergency_brake"] for e in self.events],
            "rate_limited": [e["safety_flags"]["rate_limited"] for e in self.events],
            "bounds_violated": [e["safety_flags"]["steering_clamped"] for e in self.events]
        }
        
        # Create figure
        fig, axes = plt.subplots(4, 2, figsize=(16, 12))
        fig.suptitle(f'Episode {self.metadata["episode_id"]} Playback Analysis', fontsize=16, fontweight='bold')
        
        # 1. Lane Offset
        ax = axes[0, 0]
        ax.plot(timesteps, lane_offsets, 'b-', linewidth=2, label='Lane Offset')
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax.fill_between(timesteps, -1.5, 1.5, alpha=0.1, color='red', label='Danger Zone')
        ax.set_ylabel('Lane Offset (m)')
        ax.set_title('Lane Keeping Performance')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Heading Error
        ax = axes[0, 1]
        ax.plot(timesteps, heading_errors, 'r-', linewidth=2)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax.set_ylabel('Heading Error (rad)')
        ax.set_title('Heading Alignment')
        ax.grid(True, alpha=0.3)
        
        # 3. Steering (Policy vs Safety)
        ax = axes[1, 0]
        ax.plot(timesteps, policy_steering, 'b--', linewidth=1.5, alpha=0.7, label='Policy')
        ax.plot(timesteps, safety_steering, 'g-', linewidth=2, label='Safety (Actual)')
        ax.axhline(y=1.0, color='r', linestyle=':', alpha=0.5, label='Limits')
        ax.axhline(y=-1.0, color='r', linestyle=':', alpha=0.5)
        ax.set_ylabel('Steering')
        ax.set_title('Steering Commands (Policy vs Safety)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Throttle (Policy vs Safety)
        ax = axes[1, 1]
        ax.plot(timesteps, policy_throttle, 'b--', linewidth=1.5, alpha=0.7, label='Policy')
        ax.plot(timesteps, safety_throttle, 'g-', linewidth=2, label='Safety (Actual)')
        ax.axhline(y=1.0, color='r', linestyle=':', alpha=0.5, label='Limits')
        ax.axhline(y=-1.0, color='r', linestyle=':', alpha=0.5)
        ax.set_ylabel('Throttle')
        ax.set_title('Throttle Commands (Policy vs Safety)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 5. Speed
        ax = axes[2, 0]
        ax.plot(timesteps, speeds, 'purple', linewidth=2)
        ax.set_ylabel('Speed (m/s)')
        ax.set_title('Vehicle Speed')
        ax.grid(True, alpha=0.3)
        
        # 6. Rewards
        ax = axes[2, 1]
        ax.plot(timesteps, rewards, 'orange', linewidth=2)
        cumulative_reward = np.cumsum(rewards)
        ax2 = ax.twinx()
        ax2.plot(timesteps, cumulative_reward, 'g--', linewidth=1.5, alpha=0.7, label='Cumulative')
        ax.set_ylabel('Step Reward', color='orange')
        ax2.set_ylabel('Cumulative Reward', color='green')
        ax.set_title('Reward Over Time')
        ax.grid(True, alpha=0.3)
        
        # 7. Safety Interventions Timeline
        ax = axes[3, 0]
        for i, (name, values) in enumerate(interventions.items()):
            y_pos = [i if v else None for v in values]
            ax.scatter(timesteps, y_pos, s=50, label=name, alpha=0.7)
        ax.set_yticks(range(len(interventions)))
        ax.set_yticklabels(interventions.keys())
        ax.set_xlabel('Timestep')
        ax.set_title('Safety Interventions Timeline')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='x')
        
        # 8. Disagreement Magnitude
        ax = axes[3, 1]
        steering_diff = np.abs(np.array(policy_steering) - np.array(safety_steering))
        throttle_diff = np.abs(np.array(policy_throttle) - np.array(safety_throttle))
        ax.plot(timesteps, steering_diff, 'r-', linewidth=2, label='Steering Diff')
        ax.plot(timesteps, throttle_diff, 'b-', linewidth=2, label='Throttle Diff')
        ax.set_xlabel('Timestep')
        ax.set_ylabel('|Policy - Safety|')
        ax.set_title('Policy vs Safety Disagreement')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save or self.config.save_plots:
            output_dir = Path(self.config.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            filepath = output_dir / f"episode_{self.metadata['episode_id']}_analysis.png"
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            if self.config.verbose:
                print(f"📊 Plot saved: {filepath}")
        
        if self.config.show_plots:
            plt.show()
        else:
            plt.close()
    
    def explain_action(self, timestep: int):
        """
        Explain why a specific action was taken.
        
        This is the "explainability" feature - critical for debugging.
        
        Args:
            timestep: Which timestep to explain
        """
        if timestep >= len(self.events):
            print(f"❌ Timestep {timestep} out of range (max: {len(self.events)-1})")
            return
        
        event = self.events[timestep]
        
        print("\n" + "=" * 70)
        print(f"🔍 EXPLAINING ACTION AT TIMESTEP {timestep}")
        print("=" * 70)
        
        self._print_event(event)
        
        # Reasoning
        print("\n💭 Reasoning:")
        
        state = event["state"]
        policy = event["policy_action"]
        safety = event["safety_action"]
        flags = event["safety_flags"]
        
        # Lane keeping logic
        if abs(state["lane_offset"]) > 0.5:
            print(f"  ⚠️  Large lane deviation ({state['lane_offset']:.3f}m)")
            print(f"      → Policy steering towards center: {policy['steering']:.3f}")
        
        # Heading alignment
        if abs(state["heading_error"]) > 0.1:
            print(f"  ⚠️  Heading misalignment ({state['heading_error']:.3f} rad)")
            print(f"      → Correcting heading with steering")
        
        # Safety interventions
        if flags["emergency_brake"]:
            print("  🛑 EMERGENCY BRAKE activated!")
            print(f"      → Reason: Lane offset {state['lane_offset']:.3f}m exceeded threshold")
        
        if flags["rate_limited"]:
            print("  ⚡ Rate limiting applied")
            print(f"      → Steering changed too fast, limited to prevent instability")
        
        if flags["steering_clamped"]:
            print("  📏 Steering clamped to bounds")
            print(f"      → Policy wanted {policy['steering']:.3f}, limited to {safety['steering']:.3f}")
        
        if flags["speed_limited"]:
            print("  🏎️  Speed limiting applied")
        
        # Policy vs Safety
        steering_diff = abs(policy["steering"] - safety["steering"])
        if steering_diff > 0.05:
            print(f"\n  🔄 Safety shield modified steering by {steering_diff:.3f}")
            print(f"      Policy: {policy['steering']:.3f} → Safety: {safety['steering']:.3f}")
        
        print("=" * 70)


def compare_episodes(episode_paths: List[str], metric: str = "total_reward"):
    """
    Compare multiple episodes side by side.
    
    Args:
        episode_paths: List of episode file paths
        metric: Metric to compare
    """
    print("\n" + "=" * 70)
    print("📊 EPISODE COMPARISON")
    print("=" * 70)
    
    for path in episode_paths:
        playback = EpisodePlayback(path, config=PlaybackConfig(verbose=False, show_plots=False))
        stats = playback.metadata["statistics"]
        
        print(f"\nEpisode {playback.metadata['episode_id']}:")
        print(f"  Total Reward: {stats['total_reward']:.2f}")
        print(f"  Timesteps: {playback.metadata['total_timesteps']}")
        print(f"  Safety Interventions: {stats['safety_interventions']} ({stats['safety_intervention_rate']:.2%})")
        print(f"  Max Lane Deviation: {stats['state_stats']['max_lane_deviation']:.3f}m")
    
    print("=" * 70)
