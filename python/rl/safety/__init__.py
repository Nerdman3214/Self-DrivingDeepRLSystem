"""
STEP 4: Safety Shield

Hard safety constraints that override RL policy.
Guarantees safe operation even if policy fails.

Design Pattern: Guarded Command Pattern
- Policy outputs = suggestions
- Safety layer = authority
- Actuators = dumb executors
"""

import numpy as np
from typing import Tuple, Dict, Any
from dataclasses import dataclass


@dataclass
class SafetyLimits:
    """Hard safety constraints (non-learned)."""
    
    # Steering limits
    max_steering_angle: float = 0.5  # radians (~28 degrees)
    max_steering_rate: float = 0.3   # radians/step (prevent oscillation)
    
    # Speed limits
    max_speed: float = 30.0  # m/s
    min_speed: float = 0.0   # m/s
    
    # Lane boundaries
    max_lane_offset: float = 1.5  # meters (emergency brake threshold)
    
    # Action bounds
    action_min: float = -1.0
    action_max: float = 1.0


class SafetyShield:
    """
    Safety layer that validates and constrains RL actions.
    
    Rules (in priority order):
    1. NaN/Inf check (corrupt model)
    2. Action bounds enforcement
    3. Steering rate limiting (oscillation prevention)
    4. Emergency brake (lane loss)
    5. Speed limiting
    
    Used in:
    - Automotive ECUs
    - Robotics middleware
    - Flight control systems
    """
    
    def __init__(self, limits: SafetyLimits = None):
        self.limits = limits or SafetyLimits()
        self.prev_steering = 0.0
        self.emergency_brake_active = False
        
        # Metrics
        self.interventions = {
            'nan_detected': 0,
            'bounds_violated': 0,
            'rate_limited': 0,
            'emergency_brake': 0,
            'speed_limited': 0,
        }
    
    def check_and_fix(
        self,
        action: np.ndarray,
        state: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply safety shield to policy action.
        
        Args:
            action: [steering, throttle] from policy
            state: [lane_offset, heading_error, speed, ...]
        
        Returns:
            safe_action: Validated action
            info: Safety intervention details
        """
        safe_action = action.copy()
        info = {
            'safe': True,
            'interventions': [],
            'original_action': action.copy(),
        }
        
        # Extract state components
        lane_offset = state[0]
        speed = state[2] if len(state) > 2 else 20.0
        
        # === RULE 1: NaN/Inf Check (CRITICAL) ===
        if np.isnan(safe_action).any() or np.isinf(safe_action).any():
            print("⚠️  SAFETY: NaN/Inf detected! Emergency stop.")
            safe_action = np.array([0.0, -1.0])  # Straight + brake
            info['safe'] = False
            info['interventions'].append('nan_detected')
            self.interventions['nan_detected'] += 1
            return safe_action, info
        
        # === RULE 2: Action Bounds ===
        if not (self.limits.action_min <= safe_action[0] <= self.limits.action_max):
            safe_action[0] = np.clip(safe_action[0], self.limits.action_min, self.limits.action_max)
            info['interventions'].append('steering_bounds')
            self.interventions['bounds_violated'] += 1
        
        if not (self.limits.action_min <= safe_action[1] <= self.limits.action_max):
            safe_action[1] = np.clip(safe_action[1], self.limits.action_min, self.limits.action_max)
            info['interventions'].append('throttle_bounds')
            self.interventions['bounds_violated'] += 1
        
        # === RULE 3: Steering Rate Limiting ===
        steering_change = abs(safe_action[0] - self.prev_steering)
        if steering_change > self.limits.max_steering_rate:
            # Limit to max rate
            direction = np.sign(safe_action[0] - self.prev_steering)
            safe_action[0] = self.prev_steering + direction * self.limits.max_steering_rate
            info['interventions'].append('steering_rate_limited')
            self.interventions['rate_limited'] += 1
        
        # === RULE 4: Emergency Brake (Lane Loss) ===
        if abs(lane_offset) > self.limits.max_lane_offset:
            print(f"🚨 EMERGENCY BRAKE: Lane offset {lane_offset:.2f}m exceeds {self.limits.max_lane_offset}m")
            safe_action[0] = 0.0  # Straighten
            safe_action[1] = -1.0  # Full brake
            info['safe'] = False
            info['interventions'].append('emergency_brake')
            self.interventions['emergency_brake'] += 1
            self.emergency_brake_active = True
        else:
            self.emergency_brake_active = False
        
        # === RULE 5: Steering Angle Limiting ===
        if abs(safe_action[0]) > self.limits.max_steering_angle:
            safe_action[0] = np.sign(safe_action[0]) * self.limits.max_steering_angle
            info['interventions'].append('max_steering_angle')
        
        # === RULE 6: Speed Limiting ===
        # Convert throttle to speed check (simplified)
        if speed > self.limits.max_speed:
            safe_action[1] = min(safe_action[1], -0.5)  # Force deceleration
            info['interventions'].append('speed_limited')
            self.interventions['speed_limited'] += 1
        
        # Update state
        self.prev_steering = safe_action[0]
        
        return safe_action, info
    
    def reset(self):
        """Reset shield state (call at episode start)."""
        self.prev_steering = 0.0
        self.emergency_brake_active = False
    
    def get_statistics(self) -> Dict[str, int]:
        """Get safety intervention statistics."""
        return self.interventions.copy()
    
    def is_safe(self, state: np.ndarray) -> bool:
        """
        Check if current state is safe.
        
        Returns:
            True if within safety bounds
        """
        lane_offset = state[0]
        speed = state[2] if len(state) > 2 else 20.0
        
        # Check constraints
        if abs(lane_offset) > self.limits.max_lane_offset:
            return False
        if speed > self.limits.max_speed:
            return False
        
        return True


def test_safety_shield():
    """Unit tests for safety shield."""
    shield = SafetyShield()
    
    print("="*60)
    print("SAFETY SHIELD UNIT TESTS")
    print("="*60)
    print()
    
    # Test 1: Normal operation
    print("Test 1: Normal operation")
    state = np.array([0.1, 0.0, 20.0, 1.75, 1.75, 0.0])
    action = np.array([0.1, 0.5])
    safe_action, info = shield.check_and_fix(action, state)
    assert info['safe'], "Should be safe"
    assert len(info['interventions']) == 0, "No interventions expected"
    print(f"  ✅ PASS: {safe_action}")
    print()
    
    # Test 2: NaN detection
    print("Test 2: NaN detection")
    shield.reset()
    action = np.array([np.nan, 0.5])
    safe_action, info = shield.check_and_fix(action, state)
    assert not info['safe'], "Should detect NaN"
    assert 'nan_detected' in info['interventions']
    assert safe_action[0] == 0.0 and safe_action[1] == -1.0
    print(f"  ✅ PASS: Emergency stop triggered")
    print()
    
    # Test 3: Bounds violation
    print("Test 3: Action bounds")
    shield.reset()
    action = np.array([1.5, -1.5])  # Out of [-1, 1]
    safe_action, info = shield.check_and_fix(action, state)
    assert -1.0 <= safe_action[0] <= 1.0
    assert -1.0 <= safe_action[1] <= 1.0
    print(f"  ✅ PASS: Clipped to {safe_action}")
    print()
    
    # Test 4: Steering rate limiting
    print("Test 4: Steering rate limiting")
    shield.reset()
    shield.prev_steering = 0.0
    action = np.array([0.8, 0.5])  # Large steering change
    safe_action, info = shield.check_and_fix(action, state)
    assert abs(safe_action[0]) <= shield.limits.max_steering_rate
    assert 'steering_rate_limited' in info['interventions']
    print(f"  ✅ PASS: Rate limited to {safe_action[0]:.3f}")
    print()
    
    # Test 5: Emergency brake
    print("Test 5: Emergency brake (lane loss)")
    shield.reset()
    dangerous_state = np.array([2.0, 0.5, 20.0, 0.5, 3.0, 0.0])  # 2m offset
    action = np.array([0.3, 0.7])
    safe_action, info = shield.check_and_fix(action, dangerous_state)
    assert not info['safe'], "Should trigger emergency"
    assert 'emergency_brake' in info['interventions']
    assert safe_action[0] == 0.0  # Straighten
    assert safe_action[1] == -1.0  # Full brake
    print(f"  ✅ PASS: Emergency brake activated")
    print()
    
    # Statistics
    print("Safety Statistics:")
    stats = shield.get_statistics()
    for key, count in stats.items():
        print(f"  {key}: {count}")
    print()
    
    print("="*60)
    print("✅ ALL SAFETY TESTS PASSED")
    print("="*60)


if __name__ == '__main__':
    test_safety_shield()
