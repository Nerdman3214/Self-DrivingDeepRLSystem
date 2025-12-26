"""
Curriculum Learning System

Progressive difficulty scheduling for autonomous driving training.
Automatically advances from easy to hard scenarios based on performance.

Educational + Practical Value:
    - Stabilizes training (easy → hard)
    - Faster convergence
    - Better final performance
    - Industry-standard technique
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional, Callable, Dict, Any
import numpy as np


class DifficultyLevel(Enum):
    """Curriculum difficulty levels"""
    EASY = 1
    MEDIUM = 2
    HARD = 3
    EXPERT = 4


@dataclass
class CurriculumStage:
    """
    Single stage in curriculum progression.
    
    Attributes:
        level: Difficulty level
        description: Human-readable description
        env_config: Environment configuration for this stage
        advancement_threshold: Reward threshold to advance
        min_episodes: Minimum episodes before advancement
    """
    level: DifficultyLevel
    description: str
    env_config: Dict[str, Any]
    advancement_threshold: float
    min_episodes: int = 50


class CurriculumScheduler:
    """
    Curriculum Learning Scheduler
    
    Manages progressive difficulty increase during training.
    Tracks performance and automatically advances difficulty.
    
    Example Progression:
        Level 1 (Easy): Straight road, wide lane, low speed
        Level 2 (Medium): Gentle curves, normal lane, medium speed
        Level 3 (Hard): Sharp turns, narrow lane, high speed
        Level 4 (Expert): Traffic, obstacles, complex scenarios
    
    Usage:
        >>> scheduler = CurriculumScheduler()
        >>> config = scheduler.get_current_config()
        >>> # Train...
        >>> scheduler.update(episode_reward)
        >>> if scheduler.should_advance():
        >>>     scheduler.advance()
    """
    
    def __init__(
        self,
        stages: Optional[list[CurriculumStage]] = None,
        patience: int = 100,
        performance_window: int = 50,
        curriculum_type: str = 'default',
        initial_stage: int = 0
    ):
        """
        Args:
            stages: Custom curriculum stages (uses default if None)
            patience: Episodes to wait after meeting threshold
            performance_window: Window for computing average reward
            curriculum_type: 'default' or 'traffic' (ignored if stages provided)
            initial_stage: Starting stage index
        """
        if stages is None:
            if curriculum_type == 'traffic':
                stages = self._create_traffic_curriculum()
            else:
                stages = self._create_default_curriculum()
        
        self.stages = stages
        self.patience = patience
        self.performance_window = performance_window
        
        # State tracking
        self.current_stage_idx = initial_stage
        self.episode_rewards = []
        self.episodes_in_stage = 0
        self.episodes_above_threshold = 0
        
    def _create_default_curriculum(self) -> list[CurriculumStage]:
        """
        Create default curriculum for lane keeping.
        
        Progression:
            Easy → Medium → Hard → Expert
        """
        stages = [
            # Stage 1: EASY - Learn basic control
            CurriculumStage(
                level=DifficultyLevel.EASY,
                description="Straight road, wide lane (5m), target speed 10 m/s",
                env_config={
                    'lane_width': 5.0,
                    'target_speed': 10.0,
                    'curvature': 0.0,
                    'max_episode_steps': 500
                },
                advancement_threshold=50.0,
                min_episodes=50
            ),
            
            # Stage 2: MEDIUM - Add gentle curves
            CurriculumStage(
                level=DifficultyLevel.MEDIUM,
                description="Gentle curves, normal lane (3.5m), target speed 15 m/s",
                env_config={
                    'lane_width': 3.5,
                    'target_speed': 15.0,
                    'curvature': 0.02,
                    'max_episode_steps': 750
                },
                advancement_threshold=100.0,
                min_episodes=75
            ),
            
            # Stage 3: HARD - Sharp turns, higher speed
            CurriculumStage(
                level=DifficultyLevel.HARD,
                description="Sharp turns, normal lane, target speed 20 m/s",
                env_config={
                    'lane_width': 3.5,
                    'target_speed': 20.0,
                    'curvature': 0.05,
                    'max_episode_steps': 1000
                },
                advancement_threshold=200.0,
                min_episodes=100
            ),
            
            # Stage 4: EXPERT - Full complexity
            CurriculumStage(
                level=DifficultyLevel.EXPERT,
                description="Narrow lane (3m), high speed (25 m/s), complex curves",
                env_config={
                    'lane_width': 3.0,
                    'target_speed': 25.0,
                    'curvature': 0.08,
                    'max_episode_steps': 1000
                },
                advancement_threshold=300.0,
                min_episodes=150
            )
        ]
        
        return stages
    
    @property
    def current_stage(self) -> CurriculumStage:
        """Get current curriculum stage"""
        return self.stages[self.current_stage_idx]
    
    @property
    def is_final_stage(self) -> bool:
        """Check if at final difficulty level"""
        return self.current_stage_idx >= len(self.stages) - 1
    
    def get_current_config(self) -> Dict[str, Any]:
        """Get environment configuration for current stage"""
        return self.current_stage.env_config.copy()
    
    def update(self, episode_reward: float):
        """
        Update curriculum state with new episode result.
        
        Args:
            episode_reward: Reward achieved in episode
        """
        self.episode_rewards.append(episode_reward)
        self.episodes_in_stage += 1
        
        # Check if above threshold
        if len(self.episode_rewards) >= self.performance_window:
            recent_avg = np.mean(self.episode_rewards[-self.performance_window:])
            if recent_avg >= self.current_stage.advancement_threshold:
                self.episodes_above_threshold += 1
            else:
                self.episodes_above_threshold = 0
    
    def should_advance(self) -> bool:
        """
        Check if criteria met to advance to next difficulty.
        
        Returns:
            True if should advance to next stage
        """
        if self.is_final_stage:
            return False
        
        # Requirements:
        # 1. Minimum episodes in stage
        # 2. Average reward above threshold
        # 3. Consistent performance (patience episodes)
        
        min_episodes_met = self.episodes_in_stage >= self.current_stage.min_episodes
        consistent_performance = self.episodes_above_threshold >= self.patience
        
        return min_episodes_met and consistent_performance
    
    def advance(self) -> bool:
        """
        Advance to next difficulty level.
        
        Returns:
            True if advanced, False if already at final stage
        """
        if self.is_final_stage:
            return False
        
        self.current_stage_idx += 1
        self.episodes_in_stage = 0
        self.episodes_above_threshold = 0
        
        return True
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current curriculum status.
        
        Returns:
            Status dictionary with metrics
        """
        recent_avg = 0.0
        if len(self.episode_rewards) >= self.performance_window:
            recent_avg = np.mean(self.episode_rewards[-self.performance_window:])
        
        return {
            'stage': self.current_stage.level.name,
            'stage_index': self.current_stage_idx + 1,
            'total_stages': len(self.stages),
            'description': self.current_stage.description,
            'episodes_in_stage': self.episodes_in_stage,
            'recent_avg_reward': recent_avg,
            'threshold': self.current_stage.advancement_threshold,
            'progress_to_threshold': min(recent_avg / self.current_stage.advancement_threshold * 100, 100),
            'episodes_above_threshold': self.episodes_above_threshold,
            'patience': self.patience,
            'can_advance': self.should_advance(),
            'is_final_stage': self.is_final_stage
        }
    
    def print_status(self):
        """Print curriculum status (for logging)"""
        status = self.get_status()
        print(f"\n{'='*70}")
        print(f"📚 CURRICULUM STATUS")
        print(f"{'='*70}")
        print(f"Stage: {status['stage']} ({status['stage_index']}/{status['total_stages']})")
        print(f"Description: {status['description']}")
        print(f"Episodes in stage: {status['episodes_in_stage']}")
        print(f"Recent avg reward: {status['recent_avg_reward']:.2f}")
        print(f"Threshold: {status['threshold']:.2f} "
              f"({status['progress_to_threshold']:.1f}% progress)")
        print(f"Consistent episodes: {status['episodes_above_threshold']}/{status['patience']}")
        if status['can_advance']:
            print(f"✅ Ready to advance!")
        elif status['is_final_stage']:
            print(f"🏆 Final stage - training at maximum difficulty")
        else:
            print(f"⏳ Keep training to advance...")
        print(f"{'='*70}\n")
    
    def reset(self):
        """Reset curriculum to beginning"""
        self.current_stage_idx = 0
        self.episode_rewards = []
        self.episodes_in_stage = 0
        self.episodes_above_threshold = 0


def create_traffic_curriculum() -> CurriculumScheduler:
    """
    Create curriculum for multi-agent traffic scenarios.
    
    Returns:
        Curriculum scheduler for traffic environments
    """
    stages = [
        # Stage 1: No traffic
        CurriculumStage(
            level=DifficultyLevel.EASY,
            description="No traffic, focus on lane keeping",
            env_config={
                'num_traffic_agents': 0,
                'traffic_density': 0.0,
                'target_speed': 15.0
            },
            advancement_threshold=100.0,
            min_episodes=50
        ),
        
        # Stage 2: Light traffic
        CurriculumStage(
            level=DifficultyLevel.MEDIUM,
            description="Light traffic, 2-3 vehicles",
            env_config={
                'num_traffic_agents': 3,
                'traffic_density': 0.3,
                'target_speed': 20.0
            },
            advancement_threshold=150.0,
            min_episodes=75
        ),
        
        # Stage 3: Dense traffic
        CurriculumStage(
            level=DifficultyLevel.HARD,
            description="Dense traffic, 5-7 vehicles",
            env_config={
                'num_traffic_agents': 7,
                'traffic_density': 0.6,
                'target_speed': 20.0
            },
            advancement_threshold=200.0,
            min_episodes=100
        ),
        
        # Stage 4: Heavy traffic with aggressive drivers
        CurriculumStage(
            level=DifficultyLevel.EXPERT,
            description="Heavy traffic, aggressive drivers, challenging scenarios",
            env_config={
                'num_traffic_agents': 10,
                'traffic_density': 0.8,
                'target_speed': 25.0,
                'aggressive_drivers': True
            },
            advancement_threshold=250.0,
            min_episodes=150
        )
    ]
    
    return CurriculumScheduler(stages=stages)
