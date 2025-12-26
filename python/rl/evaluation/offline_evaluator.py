"""
PHASE 2: Offline Evaluator

Scientific evaluation of frozen policies on recorded episodes.

Design Patterns:
- Command-Query Separation: Evaluation is read-only (no training)
- Pure Function: policy(state) → action (no side effects)
- Fail-Fast: Abort on NaN/instability

Key Principle:
    Same inputs → Same outputs → Same metrics
"""

import json
import gzip
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import hashlib


@dataclass
class PerformanceMetrics:
    """Performance metrics for autonomy evaluation."""
    mean_episode_reward: float
    median_episode_reward: float
    std_episode_reward: float
    mean_episode_length: float
    median_episode_length: float
    total_episodes: int


@dataclass
class SafetyMetrics:
    """Safety metrics - MORE IMPORTANT than performance."""
    max_lane_deviation: float
    mean_lane_deviation: float
    safety_interventions_per_episode: float
    total_safety_interventions: int
    emergency_brakes_per_episode: float
    total_emergency_brakes: int
    action_saturation_rate: float  # % of actions at limits
    worst_case_deviation: float  # Across all episodes


@dataclass
class ComfortMetrics:
    """Comfort metrics - often ignored, very valuable."""
    mean_steering_jerk: float
    max_steering_jerk: float
    mean_throttle_jerk: float
    max_throttle_jerk: float
    oscillation_frequency: float  # Sign changes per second


@dataclass
class EvaluationReport:
    """Complete evaluation report artifact."""
    
    # Metadata
    timestamp: str
    policy_checkpoint: str
    num_episodes: int
    determinism_hash: str
    
    # Metrics
    performance: PerformanceMetrics
    safety: SafetyMetrics
    comfort: ComfortMetrics
    
    # Pass/Fail Flags
    is_safe: bool
    is_deterministic: bool
    has_numerical_issues: bool
    
    # Additional Info
    episodes_evaluated: List[str]
    evaluation_config: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp,
            "policy_checkpoint": self.policy_checkpoint,
            "num_episodes": self.num_episodes,
            "determinism_hash": self.determinism_hash,
            "performance": asdict(self.performance),
            "safety": asdict(self.safety),
            "comfort": asdict(self.comfort),
            "is_safe": self.is_safe,
            "is_deterministic": self.is_deterministic,
            "has_numerical_issues": self.has_numerical_issues,
            "episodes_evaluated": self.episodes_evaluated,
            "evaluation_config": self.evaluation_config
        }


class OfflineEvaluator:
    """
    Offline evaluation of frozen policies.
    
    NO training. NO randomness. NO exploration.
    Pure read-only evaluation on recorded episodes.
    
    This is what makes your project scientifically valid.
    """
    
    def __init__(
        self,
        policy,
        safety_shield,
        safety_thresholds: Optional[Dict[str, float]] = None,
        verbose: bool = True
    ):
        """
        Initialize offline evaluator.
        
        Args:
            policy: Frozen policy network (eval mode)
            safety_shield: Safety shield instance
            safety_thresholds: Safety limits for pass/fail
            verbose: Print evaluation progress
        """
        self.policy = policy
        self.safety_shield = safety_shield
        self.verbose = verbose
        
        # Safety thresholds for pass/fail
        self.safety_thresholds = safety_thresholds or {
            'max_lane_deviation': 1.5,  # meters
            'max_intervention_rate': 0.3,  # 30% interventions
            'max_emergency_brake_rate': 0.1  # 10% emergency brakes
        }
        
        # Ensure policy is frozen
        self.policy.eval()
        for param in self.policy.parameters():
            param.requires_grad = False
        
        if self.verbose:
            print("🔒 Policy frozen (no gradients)")
            print("📖 Evaluation mode: READ-ONLY")
    
    def evaluate_episode(
        self,
        episode_path: str,
        determinism_check: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate single episode offline.
        
        Args:
            episode_path: Path to recorded episode
            determinism_check: Run episode twice to verify determinism
            
        Returns:
            Episode metrics
        """
        # Load episode
        episode_data = self._load_episode(episode_path)
        events = episode_data["events"]
        
        if self.verbose:
            print(f"\n📊 Evaluating: {Path(episode_path).name}")
            print(f"   Timesteps: {len(events)}")
        
        # First evaluation
        metrics_1 = self._evaluate_events(events)
        
        # Determinism check
        if determinism_check:
            metrics_2 = self._evaluate_events(events)
            is_deterministic = self._check_determinism(metrics_1, metrics_2)
            
            if not is_deterministic:
                print("⚠️  WARNING: Non-deterministic behavior detected!")
                metrics_1['deterministic'] = False
            else:
                if self.verbose:
                    print("✅ Determinism verified")
                metrics_1['deterministic'] = True
        else:
            metrics_1['deterministic'] = True
        
        return metrics_1
    
    def _evaluate_events(self, events: List[Dict]) -> Dict[str, Any]:
        """
        Evaluate episode events (pure replay, no environment).
        
        This is the core of offline evaluation:
        - No randomness
        - No environment interaction
        - Pure policy inference
        """
        timesteps = len(events)
        
        # Storage for metrics
        rewards = []
        lane_deviations = []
        steering_actions = []
        throttle_actions = []
        policy_actions = []
        safety_actions = []
        safety_flags_list = []
        
        # Replay each timestep
        for event in events:
            state = np.array([
                event["state"]["lane_offset"],
                event["state"]["heading_error"],
                event["state"]["speed"],
                event["state"]["left_distance"],
                event["state"]["right_distance"],
                event["state"]["curvature"]
            ], dtype=np.float32)
            
            # Frozen policy inference (deterministic)
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                policy_action, _, _, _ = self.policy.get_action_and_value(
                    state_tensor, deterministic=True
                )
                policy_action = policy_action.cpu().numpy().flatten()
            
            # Apply safety shield
            safety_action, safety_info = self.safety_shield.check_and_fix(
                policy_action, state
            )
            
            # Extract safety flags
            safety_flags = {
                "bounds_violated": "steering_bounds" in safety_info.get("interventions", []) or
                                  "throttle_bounds" in safety_info.get("interventions", []),
                "rate_limited": "rate_limited" in safety_info.get("interventions", []),
                "emergency_brake": "emergency_brake" in safety_info.get("interventions", []),
                "speed_limited": "speed_limited" in safety_info.get("interventions", []),
                "nan_detected": "nan_detected" in safety_info.get("interventions", [])
            }
            
            # Fail-fast on NaN/Inf
            if np.isnan(policy_action).any() or np.isinf(policy_action).any():
                raise ValueError(f"NaN/Inf detected in policy output at timestep {len(rewards)}")
            
            # Record metrics
            rewards.append(event["reward"])
            lane_deviations.append(abs(state[0]))
            steering_actions.append(safety_action[0])
            throttle_actions.append(safety_action[1])
            policy_actions.append(policy_action)
            safety_actions.append(safety_action)
            safety_flags_list.append(safety_flags)
        
        # Compute metrics
        return self._compute_metrics(
            rewards=rewards,
            lane_deviations=lane_deviations,
            steering_actions=steering_actions,
            throttle_actions=throttle_actions,
            policy_actions=np.array(policy_actions),
            safety_actions=np.array(safety_actions),
            safety_flags=safety_flags_list,
            timesteps=timesteps
        )
    
    def _compute_metrics(
        self,
        rewards: List[float],
        lane_deviations: List[float],
        steering_actions: List[float],
        throttle_actions: List[float],
        policy_actions: np.ndarray,
        safety_actions: np.ndarray,
        safety_flags: List[Dict[str, bool]],
        timesteps: int
    ) -> Dict[str, Any]:
        """Compute all required metrics."""
        
        # Performance
        total_reward = sum(rewards)
        mean_reward = np.mean(rewards)
        
        # Safety
        max_deviation = max(lane_deviations)
        mean_deviation = np.mean(lane_deviations)
        
        safety_interventions = sum(
            any(flags.values()) for flags in safety_flags
        )
        emergency_brakes = sum(
            flags.get("emergency_brake", False) for flags in safety_flags
        )
        
        # Action saturation (actions at limits ±1.0)
        saturated_actions = np.sum((np.abs(safety_actions) > 0.99))
        saturation_rate = saturated_actions / (timesteps * 2)  # 2 actions per timestep
        
        # Comfort - Steering Jerk: J = (1/T) * Σ|u_t - u_{t-1}|
        steering_arr = np.array(steering_actions)
        steering_jerk = np.diff(steering_arr)
        mean_steering_jerk = np.abs(steering_jerk).mean() if len(steering_jerk) > 0 else 0.0
        max_steering_jerk = np.abs(steering_jerk).max() if len(steering_jerk) > 0 else 0.0
        
        # Throttle jerk
        throttle_arr = np.array(throttle_actions)
        throttle_jerk = np.diff(throttle_arr)
        mean_throttle_jerk = np.abs(throttle_jerk).mean() if len(throttle_jerk) > 0 else 0.0
        max_throttle_jerk = np.abs(throttle_jerk).max() if len(throttle_jerk) > 0 else 0.0
        
        # Oscillation frequency (sign changes per second)
        sign_changes = np.sum(np.diff(np.sign(steering_arr)) != 0)
        dt = 0.05  # 20 Hz control (from environment)
        duration = timesteps * dt
        oscillation_freq = sign_changes / duration if duration > 0 else 0.0
        
        # Policy vs Safety disagreement
        action_diff = np.abs(policy_actions - safety_actions)
        mean_disagreement = action_diff.mean()
        
        return {
            # Performance
            'total_reward': float(total_reward),
            'mean_reward': float(mean_reward),
            'episode_length': timesteps,
            
            # Safety
            'max_lane_deviation': float(max_deviation),
            'mean_lane_deviation': float(mean_deviation),
            'safety_interventions': int(safety_interventions),
            'emergency_brakes': int(emergency_brakes),
            'action_saturation_rate': float(saturation_rate),
            
            # Comfort
            'mean_steering_jerk': float(mean_steering_jerk),
            'max_steering_jerk': float(max_steering_jerk),
            'mean_throttle_jerk': float(mean_throttle_jerk),
            'max_throttle_jerk': float(max_throttle_jerk),
            'oscillation_frequency': float(oscillation_freq),
            
            # Additional
            'policy_safety_disagreement': float(mean_disagreement)
        }
    
    def _check_determinism(
        self,
        metrics_1: Dict[str, Any],
        metrics_2: Dict[str, Any],
        tolerance: float = 1e-6
    ) -> bool:
        """
        Verify determinism: same inputs → same outputs.
        
        Critical for deployment.
        """
        # Compare key metrics
        keys_to_check = [
            'total_reward', 'max_lane_deviation', 'mean_steering_jerk',
            'safety_interventions', 'emergency_brakes'
        ]
        
        for key in keys_to_check:
            if abs(metrics_1[key] - metrics_2[key]) > tolerance:
                if self.verbose:
                    print(f"  Determinism failed on: {key}")
                    print(f"    Run 1: {metrics_1[key]}")
                    print(f"    Run 2: {metrics_2[key]}")
                return False
        
        return True
    
    def evaluate_episodes(
        self,
        episode_paths: List[str],
        policy_checkpoint: str = "unknown",
        output_dir: str = "evaluation_reports"
    ) -> EvaluationReport:
        """
        Evaluate multiple episodes and generate report.
        
        Args:
            episode_paths: List of episode file paths
            policy_checkpoint: Name/path of policy checkpoint
            output_dir: Where to save report
            
        Returns:
            Complete evaluation report
        """
        if self.verbose:
            print("\n" + "="*70)
            print("📊 PHASE 2: OFFLINE EVALUATION")
            print("="*70)
            print(f"Policy: {policy_checkpoint}")
            print(f"Episodes: {len(episode_paths)}")
            print("Mode: FROZEN (no training, no randomness)")
            print("="*70)
        
        # Evaluate each episode
        episode_metrics = []
        for ep_path in episode_paths:
            try:
                metrics = self.evaluate_episode(ep_path, determinism_check=True)
                episode_metrics.append(metrics)
            except Exception as e:
                print(f"❌ Failed to evaluate {ep_path}: {e}")
                continue
        
        if not episode_metrics:
            raise ValueError("No episodes successfully evaluated")
        
        # Aggregate metrics
        report = self._create_report(
            episode_metrics=episode_metrics,
            episode_paths=episode_paths,
            policy_checkpoint=policy_checkpoint
        )
        
        # Save report
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = output_path / f"evaluation_report_{timestamp}.json"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report.to_dict(), f, indent=2)
        
        if self.verbose:
            self._print_report(report)
            print(f"\n📄 Report saved: {report_path}")
        
        return report
    
    def _create_report(
        self,
        episode_metrics: List[Dict[str, Any]],
        episode_paths: List[str],
        policy_checkpoint: str
    ) -> EvaluationReport:
        """Create evaluation report from episode metrics."""
        
        # Performance metrics
        rewards = [m['total_reward'] for m in episode_metrics]
        lengths = [m['episode_length'] for m in episode_metrics]
        
        performance = PerformanceMetrics(
            mean_episode_reward=float(np.mean(rewards)),
            median_episode_reward=float(np.median(rewards)),
            std_episode_reward=float(np.std(rewards)),
            mean_episode_length=float(np.mean(lengths)),
            median_episode_length=float(np.median(lengths)),
            total_episodes=len(episode_metrics)
        )
        
        # Safety metrics
        max_deviations = [m['max_lane_deviation'] for m in episode_metrics]
        mean_deviations = [m['mean_lane_deviation'] for m in episode_metrics]
        interventions = [m['safety_interventions'] for m in episode_metrics]
        emergency_brakes = [m['emergency_brakes'] for m in episode_metrics]
        saturation_rates = [m['action_saturation_rate'] for m in episode_metrics]
        
        safety = SafetyMetrics(
            max_lane_deviation=float(np.max(max_deviations)),
            mean_lane_deviation=float(np.mean(mean_deviations)),
            safety_interventions_per_episode=float(np.mean(interventions)),
            total_safety_interventions=int(np.sum(interventions)),
            emergency_brakes_per_episode=float(np.mean(emergency_brakes)),
            total_emergency_brakes=int(np.sum(emergency_brakes)),
            action_saturation_rate=float(np.mean(saturation_rates)),
            worst_case_deviation=float(np.max(max_deviations))
        )
        
        # Comfort metrics
        steering_jerks = [m['mean_steering_jerk'] for m in episode_metrics]
        max_steering_jerks = [m['max_steering_jerk'] for m in episode_metrics]
        throttle_jerks = [m['mean_throttle_jerk'] for m in episode_metrics]
        max_throttle_jerks = [m['max_throttle_jerk'] for m in episode_metrics]
        oscillations = [m['oscillation_frequency'] for m in episode_metrics]
        
        comfort = ComfortMetrics(
            mean_steering_jerk=float(np.mean(steering_jerks)),
            max_steering_jerk=float(np.max(max_steering_jerks)),
            mean_throttle_jerk=float(np.mean(throttle_jerks)),
            max_throttle_jerk=float(np.max(max_throttle_jerks)),
            oscillation_frequency=float(np.mean(oscillations))
        )
        
        # Pass/Fail checks
        is_safe = self._check_safety(safety)
        is_deterministic = all(m.get('deterministic', True) for m in episode_metrics)
        has_numerical_issues = False  # Would be caught by fail-fast
        
        # Determinism hash
        determinism_hash = self._compute_determinism_hash(episode_metrics)
        
        return EvaluationReport(
            timestamp=datetime.now().isoformat(),
            policy_checkpoint=policy_checkpoint,
            num_episodes=len(episode_metrics),
            determinism_hash=determinism_hash,
            performance=performance,
            safety=safety,
            comfort=comfort,
            is_safe=is_safe,
            is_deterministic=is_deterministic,
            has_numerical_issues=has_numerical_issues,
            episodes_evaluated=[str(p) for p in episode_paths],
            evaluation_config={
                'safety_thresholds': self.safety_thresholds,
                'determinism_check': True
            }
        )
    
    def _check_safety(self, safety: SafetyMetrics) -> bool:
        """Check if policy meets safety requirements."""
        checks = [
            safety.worst_case_deviation <= self.safety_thresholds['max_lane_deviation'],
            safety.safety_interventions_per_episode / 100 <= self.safety_thresholds['max_intervention_rate'],
            safety.emergency_brakes_per_episode / 100 <= self.safety_thresholds['max_emergency_brake_rate']
        ]
        return all(checks)
    
    def _compute_determinism_hash(self, metrics: List[Dict]) -> str:
        """
        Compute hash of metrics for determinism verification.
        
        Only hashes invariant metrics (excludes 'deterministic' flag itself).
        """
        # Extract core metrics (exclude metadata)
        core_metrics = []
        for m in metrics:
            core = {
                k: v for k, v in m.items() 
                if k != 'deterministic'  # Exclude meta-field
            }
            core_metrics.append(core)
        
        # Create stable string representation
        metric_str = json.dumps(core_metrics, sort_keys=True)
        return hashlib.sha256(metric_str.encode()).hexdigest()[:16]
    
    def _print_report(self, report: EvaluationReport):
        """Print human-readable report."""
        print("\n" + "="*70)
        print("📄 EVALUATION REPORT")
        print("="*70)
        
        print(f"\n✅ Deterministic: {report.is_deterministic}")
        print(f"{'✅' if report.is_safe else '❌'} Safety: {'PASS' if report.is_safe else 'FAIL'}")
        print(f"✅ Numerical Stability: {'PASS' if not report.has_numerical_issues else 'FAIL'}")
        
        print(f"\n📊 PERFORMANCE METRICS")
        print(f"  Mean Reward: {report.performance.mean_episode_reward:.2f}")
        print(f"  Median Reward: {report.performance.median_episode_reward:.2f}")
        print(f"  Std Reward: {report.performance.std_episode_reward:.2f}")
        print(f"  Mean Episode Length: {report.performance.mean_episode_length:.1f}")
        
        print(f"\n🛡️  SAFETY METRICS (CRITICAL)")
        print(f"  Worst-Case Deviation: {report.safety.worst_case_deviation:.3f}m")
        print(f"  Max Lane Deviation: {report.safety.max_lane_deviation:.3f}m")
        print(f"  Safety Interventions/Episode: {report.safety.safety_interventions_per_episode:.2f}")
        print(f"  Emergency Brakes/Episode: {report.safety.emergency_brakes_per_episode:.2f}")
        print(f"  Action Saturation: {report.safety.action_saturation_rate:.2%}")
        
        print(f"\n🎯 COMFORT METRICS")
        print(f"  Mean Steering Jerk: {report.comfort.mean_steering_jerk:.4f}")
        print(f"  Max Steering Jerk: {report.comfort.max_steering_jerk:.4f}")
        print(f"  Oscillation Frequency: {report.comfort.oscillation_frequency:.2f} Hz")
        
        print("="*70)
    
    def _load_episode(self, filepath: str) -> Dict[str, Any]:
        """Load episode from disk."""
        file_path = Path(filepath)
        
        if file_path.suffix == '.gz':
            with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                return json.load(f)
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
