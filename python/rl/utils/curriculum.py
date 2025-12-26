"""
Curriculum Learning for Self-Driving RL

3-Phase Training:
1. Phase 1 - Lane Following (Easy): Straight roads, low speed
2. Phase 2 - Curves & Speed: Variable curvature, higher speed
3. Phase 3 - Robustness: Sensor noise, random offsets
"""

from enum import Enum
from typing import Dict, Any
import numpy as np


class TrainingPhase(Enum):
    """Training curriculum phases."""
    PHASE_1_STRAIGHT = 1  # Easy: straight roads
    PHASE_2_CURVES = 2    # Medium: curves and speed
    PHASE_3_ROBUST = 3    # Hard: noise and disturbances


class CurriculumScheduler:
    """
    Manages curriculum learning progression.
    
    Automatically advances phases based on performance thresholds.
    Mirrors Waymo/Tesla staged training approach.
    """
    
    def __init__(
        self,
        phase1_threshold: float = 50.0,   # Advance when mean reward > 50
        phase2_threshold: float = 200.0,  # Advance when mean reward > 200
        min_episodes_per_phase: int = 100,
    ):
        self.current_phase = TrainingPhase.PHASE_1_STRAIGHT
        self.phase1_threshold = phase1_threshold
        self.phase2_threshold = phase2_threshold
        self.min_episodes_per_phase = min_episodes_per_phase
        self.episodes_in_phase = 0
    
    def get_phase_config(self) -> Dict[str, Any]:
        """Get environment configuration for current phase."""
        if self.current_phase == TrainingPhase.PHASE_1_STRAIGHT:
            return {
                'max_curvature': 0.0,  # Straight only
                'speed_range': (5.0, 15.0),  # Low speed
                'noise_scale': 0.0,  # No noise
                'init_offset_range': (-0.5, 0.5),  # Small offset
                'description': 'Phase 1: Straight Roads, Low Speed'
            }
        elif self.current_phase == TrainingPhase.PHASE_2_CURVES:
            return {
                'max_curvature': 0.03,  # Moderate curves
                'speed_range': (10.0, 25.0),  # Higher speed
                'noise_scale': 0.0,  # No noise yet
                'init_offset_range': (-1.0, 1.0),  # Larger offset
                'description': 'Phase 2: Curves & Variable Speed'
            }
        else:  # PHASE_3_ROBUST
            return {
                'max_curvature': 0.05,  # Tight curves
                'speed_range': (15.0, 30.0),  # Full speed
                'noise_scale': 0.02,  # Sensor noise
                'init_offset_range': (-1.5, 1.5),  # Large offset
                'description': 'Phase 3: Robustness Testing'
            }
    
    def update(self, mean_reward: float) -> bool:
        """
        Update phase based on performance.
        
        Returns:
            True if phase advanced
        """
        self.episodes_in_phase += 1
        
        # Need minimum episodes before advancing
        if self.episodes_in_phase < self.min_episodes_per_phase:
            return False
        
        # Check advancement criteria
        advanced = False
        
        if self.current_phase == TrainingPhase.PHASE_1_STRAIGHT:
            if mean_reward >= self.phase1_threshold:
                self.current_phase = TrainingPhase.PHASE_2_CURVES
                self.episodes_in_phase = 0
                advanced = True
                print(f"\n{'='*60}")
                print(f"🎓 ADVANCING TO PHASE 2: Curves & Variable Speed")
                print(f"Mean reward: {mean_reward:.2f} >= threshold {self.phase1_threshold}")
                print(f"{'='*60}\n")
        
        elif self.current_phase == TrainingPhase.PHASE_2_CURVES:
            if mean_reward >= self.phase2_threshold:
                self.current_phase = TrainingPhase.PHASE_3_ROBUST
                self.episodes_in_phase = 0
                advanced = True
                print(f"\n{'='*60}")
                print(f"🎓 ADVANCING TO PHASE 3: Robustness Testing")
                print(f"Mean reward: {mean_reward:.2f} >= threshold {self.phase2_threshold}")
                print(f"{'='*60}\n")
        
        return advanced
    
    def get_phase_name(self) -> str:
        """Get current phase description."""
        return self.get_phase_config()['description']
