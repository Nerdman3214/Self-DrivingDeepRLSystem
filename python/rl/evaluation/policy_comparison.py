"""
Policy Comparison Dashboard

Visual and quantitative comparison of multiple policy checkpoints.
Shows learning progression through side-by-side evaluation.

Purpose:
    - Visualize learning progress
    - Compare different training runs
    - Identify regressions
    - Resume-worthy demonstration
"""

import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import json
from datetime import datetime
import matplotlib.pyplot as plt


@dataclass
class PolicyCheckpoint:
    """Metadata for a single policy checkpoint"""
    path: Path
    name: str
    timesteps: int
    episode: Optional[int] = None
    avg_reward: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComparisonResult:
    """Results from comparing multiple policies"""
    checkpoints: List[PolicyCheckpoint]
    metrics: Dict[str, List[float]]
    episode_data: List[Dict[str, Any]]
    best_policy_idx: int
    worst_policy_idx: int
    improvement_pct: float


class PolicyComparator:
    """
    Compare multiple policy checkpoints.
    
    Features:
        - Load multiple checkpoints
        - Run side-by-side evaluation
        - Generate comparison metrics
        - Visualize learning progression
        - Export comparison reports
    
    Usage:
        >>> comparator = PolicyComparator(env_factory)
        >>> comparator.add_checkpoint('checkpoints/model_10000.pt')
        >>> comparator.add_checkpoint('checkpoints/model_50000.pt')
        >>> results = comparator.compare(num_episodes=10)
        >>> comparator.plot_comparison(results)
    """
    
    def __init__(self, env_factory, policy_class):
        """
        Args:
            env_factory: Function that creates environment
            policy_class: Policy network class (e.g., MLPActorCritic)
        """
        self.env_factory = env_factory
        self.policy_class = policy_class
        self.checkpoints: List[PolicyCheckpoint] = []
    
    def add_checkpoint(
        self,
        checkpoint_path: str | Path,
        name: Optional[str] = None
    ):
        """
        Add checkpoint to comparison.
        
        Args:
            checkpoint_path: Path to checkpoint file
            name: Optional custom name
        """
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        
        # Load checkpoint to extract metadata
        checkpoint = torch.load(path, map_location='cpu')
        
        # Extract info
        timesteps = checkpoint.get('total_steps', 0)
        avg_reward = None
        if 'episode_rewards' in checkpoint and checkpoint['episode_rewards']:
            avg_reward = np.mean(checkpoint['episode_rewards'][-100:])
        
        # Create checkpoint info
        cp = PolicyCheckpoint(
            path=path,
            name=name or path.stem,
            timesteps=timesteps,
            avg_reward=avg_reward,
            metadata=checkpoint
        )
        
        self.checkpoints.append(cp)
        print(f"Added checkpoint: {cp.name} (timesteps={cp.timesteps})")
    
    def load_all_from_directory(
        self,
        checkpoint_dir: str | Path,
        pattern: str = "*.pt",
        max_checkpoints: int = 10
    ):
        """
        Load multiple checkpoints from directory.
        
        Args:
            checkpoint_dir: Directory containing checkpoints
            pattern: File pattern to match
            max_checkpoints: Maximum number to load
        """
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_files = sorted(checkpoint_dir.glob(pattern))
        
        # Sample evenly if too many
        if len(checkpoint_files) > max_checkpoints:
            indices = np.linspace(
                0,
                len(checkpoint_files) - 1,
                max_checkpoints,
                dtype=int
            )
            checkpoint_files = [checkpoint_files[i] for i in indices]
        
        for cp_file in checkpoint_files:
            self.add_checkpoint(cp_file)
    
    def _load_policy(self, checkpoint: PolicyCheckpoint):
        """Load policy from checkpoint"""
        # Create policy instance
        env = self.env_factory()
        
        # Extract hyperparameters from checkpoint if available
        hyperparams = checkpoint.metadata.get('hyperparameters', {})
        hidden_dims = hyperparams.get('hidden_dims', (64, 64))  # Default for old checkpoints
        
        policy = self.policy_class(
            observation_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            hidden_dims=hidden_dims
        )
        env.close()
        
        # Load weights
        state_dict = checkpoint.metadata['policy_state_dict']
        policy.load_state_dict(state_dict)
        policy.eval()
        
        return policy
    
    def evaluate_single(
        self,
        policy,
        num_episodes: int = 10,
        render: bool = False
    ) -> Dict[str, Any]:
        """
        Evaluate single policy.
        
        Args:
            policy: Policy to evaluate
            num_episodes: Number of episodes
            render: Whether to render
        
        Returns:
            Evaluation metrics
        """
        env = self.env_factory()
        if render and hasattr(env, 'render_mode'):
            env.render_mode = 'human'
        
        episode_rewards = []
        episode_lengths = []
        lane_offsets = []
        speed_violations = []
        
        for ep in range(num_episodes):
            obs, _ = env.reset()
            episode_reward = 0
            episode_length = 0
            max_lane_offset = 0
            
            done = False
            while not done:
                # Get action
                with torch.no_grad():
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                    action, _, _, _ = policy.get_action_and_value(
                        obs_tensor,
                        deterministic=True
                    )
                    action = action[0].numpy()
                
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                episode_length += 1
                done = terminated or truncated
                
                # Track metrics
                if hasattr(obs, '__iter__') and len(obs) > 0:
                    max_lane_offset = max(max_lane_offset, abs(obs[0]))
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            lane_offsets.append(max_lane_offset)
        
        env.close()
        
        return {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'max_lane_offset': np.mean(lane_offsets),
            'episodes': num_episodes
        }
    
    def compare(
        self,
        num_episodes: int = 10,
        render_best: bool = False
    ) -> ComparisonResult:
        """
        Compare all loaded checkpoints.
        
        Args:
            num_episodes: Episodes to evaluate each checkpoint
            render_best: Whether to render best policy
        
        Returns:
            Comparison results
        """
        if not self.checkpoints:
            raise ValueError("No checkpoints loaded. Use add_checkpoint() first.")
        
        print("\n" + "="*70)
        print("🔍 POLICY COMPARISON")
        print("="*70)
        print(f"Comparing {len(self.checkpoints)} checkpoints")
        print(f"Episodes per checkpoint: {num_episodes}\n")
        
        # Evaluate all
        results = []
        for i, checkpoint in enumerate(self.checkpoints):
            print(f"[{i+1}/{len(self.checkpoints)}] Evaluating: {checkpoint.name}...")
            
            policy = self._load_policy(checkpoint)
            metrics = self.evaluate_single(policy, num_episodes, render=False)
            
            results.append(metrics)
            print(f"  Reward: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
            print(f"  Length: {metrics['mean_length']:.1f}")
        
        # Aggregate metrics
        aggregated_metrics = {
            'mean_rewards': [r['mean_reward'] for r in results],
            'std_rewards': [r['std_reward'] for r in results],
            'mean_lengths': [r['mean_length'] for r in results],
            'max_lane_offsets': [r['max_lane_offset'] for r in results]
        }
        
        # Find best/worst
        rewards = aggregated_metrics['mean_rewards']
        best_idx = int(np.argmax(rewards))
        worst_idx = int(np.argmin(rewards))
        improvement = ((rewards[best_idx] - rewards[worst_idx]) / abs(rewards[worst_idx]) * 100
                      if rewards[worst_idx] != 0 else 0.0)
        
        print("\n" + "="*70)
        print(f"Best: {self.checkpoints[best_idx].name} "
              f"(reward={rewards[best_idx]:.2f})")
        print(f"Worst: {self.checkpoints[worst_idx].name} "
              f"(reward={rewards[worst_idx]:.2f})")
        print(f"Improvement: {improvement:+.1f}%")
        print("="*70)
        
        # Render best if requested
        if render_best:
            print(f"\nRendering best policy: {self.checkpoints[best_idx].name}")
            best_policy = self._load_policy(self.checkpoints[best_idx])
            self.evaluate_single(best_policy, num_episodes=3, render=True)
        
        return ComparisonResult(
            checkpoints=self.checkpoints,
            metrics=aggregated_metrics,
            episode_data=results,
            best_policy_idx=best_idx,
            worst_policy_idx=worst_idx,
            improvement_pct=improvement
        )
    
    def plot_comparison(
        self,
        results: ComparisonResult,
        save_path: Optional[str] = None
    ):
        """
        Create visualization of comparison.
        
        Args:
            results: Comparison results
            save_path: Optional path to save figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Policy Comparison Dashboard', fontsize=16, fontweight='bold')
        
        # Extract data
        timesteps = [cp.timesteps for cp in results.checkpoints]
        names = [cp.name for cp in results.checkpoints]
        
        # Plot 1: Mean Reward
        ax = axes[0, 0]
        ax.plot(timesteps, results.metrics['mean_rewards'], 'o-', linewidth=2, markersize=8)
        ax.fill_between(
            timesteps,
            np.array(results.metrics['mean_rewards']) - np.array(results.metrics['std_rewards']),
            np.array(results.metrics['mean_rewards']) + np.array(results.metrics['std_rewards']),
            alpha=0.3
        )
        ax.set_xlabel('Training Timesteps')
        ax.set_ylabel('Mean Reward')
        ax.set_title('Learning Progression (Reward)')
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Episode Length
        ax = axes[0, 1]
        ax.plot(timesteps, results.metrics['mean_lengths'], 'o-', color='green',
                linewidth=2, markersize=8)
        ax.set_xlabel('Training Timesteps')
        ax.set_ylabel('Mean Episode Length')
        ax.set_title('Survival Time Improvement')
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Lane Offset (Safety)
        ax = axes[1, 0]
        ax.plot(timesteps, results.metrics['max_lane_offsets'], 'o-', color='red',
                linewidth=2, markersize=8)
        ax.set_xlabel('Training Timesteps')
        ax.set_ylabel('Max Lane Offset (m)')
        ax.set_title('Safety Improvement (Lower = Better)')
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Bar comparison
        ax = axes[1, 1]
        x_pos = np.arange(len(names))
        colors = ['green' if i == results.best_policy_idx else
                 'red' if i == results.worst_policy_idx else 'gray'
                 for i in range(len(names))]
        ax.bar(x_pos, results.metrics['mean_rewards'], color=colors, alpha=0.7)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.set_ylabel('Mean Reward')
        ax.set_title('Checkpoint Comparison')
        ax.grid(True, axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nFigure saved: {save_path}")
        else:
            plt.show()
    
    def export_report(
        self,
        results: ComparisonResult,
        output_path: str | Path
    ):
        """
        Export comparison report to JSON.
        
        Args:
            results: Comparison results
            output_path: Path to save JSON report
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'num_checkpoints': len(results.checkpoints),
            'checkpoints': [
                {
                    'name': cp.name,
                    'timesteps': cp.timesteps,
                    'path': str(cp.path)
                }
                for cp in results.checkpoints
            ],
            'metrics': {
                key: [float(v) for v in values]
                for key, values in results.metrics.items()
            },
            'best_policy': results.checkpoints[results.best_policy_idx].name,
            'worst_policy': results.checkpoints[results.worst_policy_idx].name,
            'improvement_percentage': float(results.improvement_pct)
        }
        
        output_path = Path(output_path)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nComparison report saved: {output_path}")


# CLI Interface
def main():
    """Command-line interface for policy comparison"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Compare multiple policy checkpoints"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        required=True,
        help="Directory containing checkpoints"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.pt",
        help="File pattern for checkpoints"
    )
    parser.add_argument(
        "--max-checkpoints",
        type=int,
        default=10,
        help="Maximum checkpoints to compare"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Episodes per checkpoint"
    )
    parser.add_argument(
        "--render-best",
        action="store_true",
        help="Render best policy"
    )
    parser.add_argument(
        "--output-plot",
        type=str,
        help="Path to save comparison plot"
    )
    parser.add_argument(
        "--output-report",
        type=str,
        help="Path to save JSON report"
    )
    
    args = parser.parse_args()
    
    # Import environment and policy
    from rl.envs.pybullet_driving_env import PyBulletDrivingEnv
    from rl.networks.mlp_policy import MLPActorCritic
    
    # Create comparator
    def env_factory():
        return PyBulletDrivingEnv(render_mode=None)
    
    comparator = PolicyComparator(env_factory, MLPActorCritic)
    
    # Load checkpoints
    comparator.load_all_from_directory(
        args.checkpoint_dir,
        pattern=args.pattern,
        max_checkpoints=args.max_checkpoints
    )
    
    # Run comparison
    results = comparator.compare(
        num_episodes=args.episodes,
        render_best=args.render_best
    )
    
    # Generate outputs
    if args.output_plot:
        comparator.plot_comparison(results, args.output_plot)
    
    if args.output_report:
        comparator.export_report(results, args.output_report)


if __name__ == "__main__":
    main()
