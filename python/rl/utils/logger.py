"""
Training Logger for Self-Driving RL

Provides logging utilities for tracking training progress,
saving checkpoints, and visualizing metrics.
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from collections import defaultdict
import numpy as np


class Logger:
    """
    Logger for RL training.
    
    Features:
    - Metric tracking with running statistics
    - TensorBoard integration (optional)
    - JSON log export
    - Checkpoint management
    """
    
    def __init__(
        self,
        log_dir: str,
        experiment_name: str = "experiment",
        use_tensorboard: bool = True,
        verbose: int = 1
    ):
        """
        Args:
            log_dir: Directory for logs
            experiment_name: Name of this experiment
            use_tensorboard: Whether to use TensorBoard
            verbose: Verbosity level (0=silent, 1=info, 2=debug)
        """
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name
        self.verbose = verbose
        
        # Create directories
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Metric storage
        self.metrics = defaultdict(list)
        self.episode_metrics = defaultdict(list)
        
        # Timing
        self.start_time = time.time()
        self.last_log_time = self.start_time
        
        # Step counters
        self.total_timesteps = 0
        self.num_episodes = 0
        
        # TensorBoard
        self.writer = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                tb_dir = self.log_dir / "tensorboard"
                self.writer = SummaryWriter(str(tb_dir))
                if self.verbose:
                    print(f"TensorBoard logging to: {tb_dir}")
            except ImportError:
                if self.verbose:
                    print("TensorBoard not available. Install with: pip install tensorboard")
        
        # JSON log file
        self.log_file = self.log_dir / f"{experiment_name}_log.json"
    
    def log_metric(self, name: str, value: float, step: Optional[int] = None):
        """
        Log a single metric.
        
        Args:
            name: Metric name
            value: Metric value
            step: Optional step number (defaults to total_timesteps)
        """
        if step is None:
            step = self.total_timesteps
        
        self.metrics[name].append({
            'step': step,
            'value': value,
            'time': time.time() - self.start_time
        })
        
        if self.writer is not None:
            self.writer.add_scalar(name, value, step)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log multiple metrics at once.
        
        Args:
            metrics: Dictionary of metric names to values
            step: Optional step number
        """
        for name, value in metrics.items():
            self.log_metric(name, value, step)
    
    def log_episode(
        self,
        reward: float,
        length: int,
        info: Optional[Dict[str, Any]] = None
    ):
        """
        Log episode completion.
        
        Args:
            reward: Total episode reward
            length: Episode length
            info: Additional episode info
        """
        self.num_episodes += 1
        
        self.episode_metrics['reward'].append(reward)
        self.episode_metrics['length'].append(length)
        
        if info:
            for key, value in info.items():
                if isinstance(value, (int, float)):
                    self.episode_metrics[f'info/{key}'].append(value)
        
        # Log to TensorBoard
        if self.writer is not None:
            self.writer.add_scalar('episode/reward', reward, self.num_episodes)
            self.writer.add_scalar('episode/length', length, self.num_episodes)
    
    def log_training(
        self,
        iteration: int,
        timesteps: int,
        metrics: Dict[str, float]
    ):
        """
        Log training iteration.
        
        Args:
            iteration: Training iteration number
            timesteps: Current total timesteps
            metrics: Training metrics (loss, etc.)
        """
        self.total_timesteps = timesteps
        
        # Calculate FPS
        current_time = time.time()
        elapsed = current_time - self.last_log_time
        fps = (timesteps - getattr(self, '_last_timesteps', 0)) / elapsed if elapsed > 0 else 0
        self._last_timesteps = timesteps
        self.last_log_time = current_time
        
        # Log metrics
        self.log_metrics(metrics, timesteps)
        self.log_metric('train/fps', fps, timesteps)
        
        # Print if verbose
        if self.verbose >= 1:
            total_elapsed = current_time - self.start_time
            hours = int(total_elapsed // 3600)
            minutes = int((total_elapsed % 3600) // 60)
            seconds = int(total_elapsed % 60)
            
            print(f"\n{'='*60}")
            print(f"Iteration: {iteration} | Timesteps: {timesteps:,}")
            print(f"Time: {hours:02d}:{minutes:02d}:{seconds:02d} | FPS: {fps:.0f}")
            print(f"{'='*60}")
            
            for name, value in metrics.items():
                print(f"  {name}: {value:.4f}")
            
            # Episode statistics
            if self.episode_metrics['reward']:
                recent_rewards = self.episode_metrics['reward'][-100:]
                print(f"\nRecent Episodes ({len(recent_rewards)}):")
                print(f"  Mean Reward: {np.mean(recent_rewards):.2f}")
                print(f"  Std Reward: {np.std(recent_rewards):.2f}")
                print(f"  Min/Max: {np.min(recent_rewards):.2f} / {np.max(recent_rewards):.2f}")
    
    def log_evaluation(
        self,
        rewards: List[float],
        lengths: List[int],
        step: Optional[int] = None
    ):
        """
        Log evaluation results.
        
        Args:
            rewards: List of episode rewards
            lengths: List of episode lengths
            step: Current timestep
        """
        if step is None:
            step = self.total_timesteps
        
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        mean_length = np.mean(lengths)
        
        self.log_metric('eval/mean_reward', mean_reward, step)
        self.log_metric('eval/std_reward', std_reward, step)
        self.log_metric('eval/mean_length', mean_length, step)
        
        if self.verbose >= 1:
            print(f"\n{'*'*60}")
            print(f"EVALUATION at step {step:,}")
            print(f"{'*'*60}")
            print(f"  Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")
            print(f"  Mean Length: {mean_length:.1f}")
            print(f"  Episodes: {len(rewards)}")
    
    def save(self, filepath: Optional[str] = None):
        """
        Save logs to JSON file.
        
        Args:
            filepath: Optional custom filepath
        """
        if filepath is None:
            filepath = self.log_file
        
        data = {
            'experiment_name': self.experiment_name,
            'total_timesteps': self.total_timesteps,
            'num_episodes': self.num_episodes,
            'total_time': time.time() - self.start_time,
            'metrics': dict(self.metrics),
            'episode_metrics': dict(self.episode_metrics)
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        
        if self.verbose >= 2:
            print(f"Logs saved to: {filepath}")
    
    def close(self):
        """Clean up resources."""
        self.save()
        
        if self.writer is not None:
            self.writer.close()
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics.
        
        Returns:
            Dictionary with summary statistics
        """
        summary = {
            'total_timesteps': self.total_timesteps,
            'num_episodes': self.num_episodes,
            'total_time': time.time() - self.start_time
        }
        
        if self.episode_metrics['reward']:
            rewards = self.episode_metrics['reward']
            summary['mean_reward'] = np.mean(rewards)
            summary['max_reward'] = np.max(rewards)
            summary['min_reward'] = np.min(rewards)
            summary['final_100_mean'] = np.mean(rewards[-100:])
        
        return summary


class DummyLogger:
    """Dummy logger that does nothing (for evaluation without logging)."""
    
    def log_metric(self, *args, **kwargs): pass
    def log_metrics(self, *args, **kwargs): pass
    def log_episode(self, *args, **kwargs): pass
    def log_training(self, *args, **kwargs): pass
    def log_evaluation(self, *args, **kwargs): pass
    def save(self, *args, **kwargs): pass
    def close(self): pass
