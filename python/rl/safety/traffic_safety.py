"""
Traffic-Aware Safety Shield

Enhanced safety layer with Time-to-Collision (TTC) logic.

Emergency Override Rules:
1. TTC < threshold → Hard brake
2. Lead braking fast → Throttle = 0
3. Unsafe lane change → Cancel steer
4. Sensor NaN → Full stop

This is the final authority - NOT the policy.
"""

import numpy as np
from typing import Tuple, Dict, Any
from dataclasses import dataclass


@dataclass
class TrafficSafetyLimits:
    """Traffic-aware safety constraints."""
    
    # TTC thresholds (seconds)
    ttc_emergency: float = 1.5  # Emergency brake
    ttc_warning: float = 3.0    # Reduce throttle
    
    # Steering limits
    max_steering_angle: float = 0.5
    max_steering_rate: float = 0.3
    
    # Speed limits
    max_speed: float = 35.0
    min_speed: float = 0.0
    
    # Lane boundaries
    max_lane_offset: float = 1.75
    
    # Action bounds
    action_min: float = -1.0
    action_max: float = 1.0
    
    # Minimum safe gap (meters)
    min_safe_gap: float = 5.0


class TrafficSafetyShield:
    """
    Safety shield with traffic awareness.
    
    Priority Order (highest to lowest):
    1. NaN/Inf detection (corrupted model)
    2. Emergency TTC brake
    3. Action bounds
    4. Steering rate limiting
    5. Speed limiting
    6. Lane boundary enforcement
    
    Used in:
    - Automotive systems
    - Waymo, Cruise, Tesla (similar concepts)
    - Any safety-critical autonomy
    """
    
    def __init__(self, limits: TrafficSafetyLimits = None):
        self.limits = limits or TrafficSafetyLimits()
        self.prev_steering = 0.0
        
        # Intervention tracking
        self.interventions = {
            'nan_detected': 0,
            'ttc_emergency': 0,
            'ttc_warning': 0,
            'bounds_violated': 0,
            'rate_limited': 0,
            'speed_limited': 0,
            'lane_boundary': 0,
            'unsafe_gap': 0
        }
    
    def check_and_fix(
        self,
        action: np.ndarray,
        state: np.ndarray,
        traffic_info: Dict[str, Any]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply traffic-aware safety shield.
        
        Args:
            action: [steering, throttle] from policy
            state: [lane_offset, heading_error, speed, lead_distance, 
                    lead_relative_speed, left_free, right_free, ttc]
            traffic_info: Additional traffic context
            
        Returns:
            safe_action: Clamped/overridden action
            safety_info: Intervention details
        """
        safe_action = action.copy()
        interventions_this_step = []
        
        # Extract state
        lane_offset = state[0]
        heading_error = state[1]
        speed = state[2]
        lead_distance = state[3]
        lead_relative_speed = state[4]
        ttc = state[7]
        
        # 1. NaN/Inf check (CRITICAL)
        if np.isnan(action).any() or np.isinf(action).any():
            safe_action = np.array([0.0, -1.0])  # Full brake
            interventions_this_step.append('nan_detected')
            self.interventions['nan_detected'] += 1
            
            return safe_action, {
                'interventions': interventions_this_step,
                'reason': 'NaN detected - emergency stop'
            }
        
        # 2. Emergency TTC brake
        if ttc < self.limits.ttc_emergency:
            safe_action[1] = -1.0  # Full brake
            safe_action[0] = np.clip(action[0], -0.2, 0.2)  # Limit steering
            interventions_this_step.append('ttc_emergency')
            self.interventions['ttc_emergency'] += 1
        
        # 3. TTC warning (reduce throttle)
        elif ttc < self.limits.ttc_warning:
            if safe_action[1] > 0.0:
                safe_action[1] = 0.0  # Cut throttle
                interventions_this_step.append('ttc_warning')
                self.interventions['ttc_warning'] += 1
        
        # 4. Unsafe gap (too close)
        if lead_distance < self.limits.min_safe_gap and lead_relative_speed > 0:
            # Closing in on leader
            safe_action[1] = min(safe_action[1], -0.5)  # Brake
            interventions_this_step.append('unsafe_gap')
            self.interventions['unsafe_gap'] += 1
        
        # 5. Action bounds
        original_action = safe_action.copy()
        safe_action = np.clip(
            safe_action,
            self.limits.action_min,
            self.limits.action_max
        )
        if not np.allclose(original_action, safe_action):
            interventions_this_step.append('bounds_violated')
            self.interventions['bounds_violated'] += 1
        
        # 6. Steering rate limiting (prevent oscillation)
        steering_change = abs(safe_action[0] - self.prev_steering)
        if steering_change > self.limits.max_steering_rate:
            # Clamp steering change
            if safe_action[0] > self.prev_steering:
                safe_action[0] = self.prev_steering + self.limits.max_steering_rate
            else:
                safe_action[0] = self.prev_steering - self.limits.max_steering_rate
            
            interventions_this_step.append('rate_limited')
            self.interventions['rate_limited'] += 1
        
        # 7. Speed limiting
        if speed > self.limits.max_speed:
            safe_action[1] = min(safe_action[1], -0.5)  # Force decel
            interventions_this_step.append('speed_limited')
            self.interventions['speed_limited'] += 1
        
        # 8. Lane boundary enforcement
        if abs(lane_offset) > self.limits.max_lane_offset * 0.9:
            # Near boundary - steer back
            if lane_offset > 0:
                safe_action[0] = min(safe_action[0], 0.0)  # Only left steering
            else:
                safe_action[0] = max(safe_action[0], 0.0)  # Only right steering
            
            interventions_this_step.append('lane_boundary')
            self.interventions['lane_boundary'] += 1
        
        # Update state
        self.prev_steering = safe_action[0]
        
        # Build safety info
        safety_info = {
            'interventions': interventions_this_step,
            'original_action': action,
            'safe_action': safe_action,
            'ttc': ttc,
            'lead_distance': lead_distance,
            'intervention_count': len(interventions_this_step)
        }
        
        return safe_action, safety_info
    
    def reset(self):
        """Reset shield state."""
        self.prev_steering = 0.0
    
    def get_intervention_stats(self) -> Dict[str, int]:
        """Get intervention statistics."""
        return self.interventions.copy()
    
    def get_total_interventions(self) -> int:
        """Get total number of interventions."""
        return sum(self.interventions.values())
