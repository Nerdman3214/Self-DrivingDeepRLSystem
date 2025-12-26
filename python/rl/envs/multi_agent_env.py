"""
Multi-Agent Lane Keeping Environment

Environment with ego vehicle (learning) + traffic agents (rule-based).

Design:
- Ego: Deep RL (PPO)
- Traffic: IDM-based (deterministic)
- Safety: Enhanced shield with TTC
- Observation: 8D vector (ego + traffic)

This is industry-standard multi-agent RL.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Dict, Any, Optional

from .traffic_agents import TrafficScenario


class MultiAgentLaneKeepingEnv(gym.Env):
    """
    Lane keeping environment with traffic.
    
    Observation Space (8D):
        0: lane_offset (ego)
        1: heading_error (ego)
        2: speed (ego)
        3: lead_distance
        4: lead_relative_speed
        5: left_lane_free (binary)
        6: right_lane_free (binary)
        7: time_to_collision (TTC)
    
    Action Space (2D):
        0: steering [-1, 1]
        1: throttle [-1, 1]
    
    Key Features:
    - Only ego learns
    - Traffic is predictable (IDM)
    - TTC computed every step
    - Collision detection
    - Phase 1/2 compatible
    """
    
    metadata = {'render_modes': []}
    
    def __init__(
        self,
        scenario_type: str = 'highway',
        max_episode_steps: int = 500,
        dt: float = 0.05,  # 20 Hz
        collision_penalty: float = -5.0,
        unsafe_gap_penalty: float = -2.0
    ):
        super().__init__()
        
        self.scenario_type = scenario_type
        self.max_episode_steps = max_episode_steps
        self.dt = dt
        self.collision_penalty = collision_penalty
        self.unsafe_gap_penalty = unsafe_gap_penalty
        
        # Observation space: 8D vector
        self.observation_space = spaces.Box(
            low=np.array([-2.0, -0.5, 0.0, 0.0, -30.0, 0.0, 0.0, 0.0]),
            high=np.array([2.0, 0.5, 40.0, 200.0, 30.0, 1.0, 1.0, 100.0]),
            dtype=np.float32
        )
        
        # Action space: [steering, throttle]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )
        
        # Traffic scenario
        self.traffic = TrafficScenario(scenario_type)
        
        # Ego state
        self.ego_lane = 0
        self.ego_position = 0.0
        self.ego_speed = 10.0
        self.ego_lane_offset = 0.0
        self.ego_heading_error = 0.0
        
        # Episode state
        self.current_step = 0
        self.total_reward = 0.0
        self.collision_occurred = False
        self.prev_steering = 0.0
        
        # Statistics
        self.stats = {
            'collisions': 0,
            'near_misses': 0,
            'avg_ttc': [],
            'min_ttc': float('inf')
        }
    
    def _compute_ttc(
        self,
        lead_distance: float,
        ego_speed: float,
        lead_speed: float,
        epsilon: float = 0.1
    ) -> float:
        """
        Compute Time-to-Collision.
        
        TTC = d / max(v_ego - v_lead, ε)
        
        Critical safety metric:
        - TTC < 1.5s: Emergency
        - TTC < 3.0s: Warning
        - TTC > 5.0s: Safe
        
        Args:
            lead_distance: Distance to lead vehicle (m)
            ego_speed: Ego speed (m/s)
            lead_speed: Lead vehicle speed (m/s)
            epsilon: Prevents division by zero
            
        Returns:
            TTC in seconds (clamped to [0, 100])
        """
        closing_speed = ego_speed - lead_speed
        
        if closing_speed <= 0.0:
            # Not closing in (safe)
            return 100.0
        
        ttc = lead_distance / max(closing_speed, epsilon)
        return np.clip(ttc, 0.0, 100.0)
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation for ego agent."""
        # Get traffic observations
        traffic_obs = self.traffic.get_observations_for_ego(
            self.ego_lane,
            self.ego_position
        )
        
        # Compute TTC
        ttc = self._compute_ttc(
            traffic_obs['lead_distance'],
            self.ego_speed,
            traffic_obs['lead_speed']
        )
        
        # Relative speed
        lead_relative_speed = self.ego_speed - traffic_obs['lead_speed']
        
        obs = np.array([
            self.ego_lane_offset,
            self.ego_heading_error,
            self.ego_speed,
            traffic_obs['lead_distance'],
            lead_relative_speed,
            float(traffic_obs['left_lane_free']),
            float(traffic_obs['right_lane_free']),
            ttc
        ], dtype=np.float32)
        
        return obs
    
    def _compute_reward(
        self,
        action: np.ndarray,
        traffic_obs: Dict[str, Any],
        ttc: float
    ) -> float:
        """
        Compute traffic-aware reward.
        
        R = +1.0 * exp(-|lane_offset|)        # Stay in lane
            -0.5 * |heading_error|             # Face forward
            -0.2 * steering_jerk               # Smooth control
            -5.0 * collision                   # HARD penalty
            -2.0 * unsafe_gap                  # Maintain safe distance
            +0.3 * smooth_merge                # Reward courtesy
        """
        reward = 0.0
        
        # Lane keeping
        reward += 1.0 * np.exp(-abs(self.ego_lane_offset))
        
        # Heading alignment
        reward -= 0.5 * abs(self.ego_heading_error)
        
        # Smooth control (steering jerk)
        steering_jerk = abs(action[0] - self.prev_steering)
        reward -= 0.2 * steering_jerk
        
        # Collision (terminal)
        if self.collision_occurred:
            reward += self.collision_penalty
        
        # Unsafe gap (TTC-based)
        if ttc < 2.0:
            reward += self.unsafe_gap_penalty
            self.stats['near_misses'] += 1
        
        # Smooth merging (reward maintaining safe distance)
        if 3.0 < ttc < 5.0:
            reward += 0.3
        
        # Speed maintenance (stay near desired speed)
        desired_speed = 20.0
        speed_error = abs(self.ego_speed - desired_speed)
        reward -= 0.1 * speed_error / 10.0
        
        return reward
    
    def _update_ego_dynamics(self, action: np.ndarray):
        """
        Update ego vehicle dynamics.
        
        Simplified dynamics:
        - Steering affects lateral position
        - Throttle affects speed
        """
        steering = action[0]
        throttle = action[1]
        
        # Lateral dynamics (simplified)
        # steering -> lateral velocity -> lane offset
        lateral_velocity = steering * 2.0  # Max 2 m/s lateral
        self.ego_lane_offset += lateral_velocity * self.dt
        
        # Heading error (coupled to lane offset)
        self.ego_heading_error = np.arctan2(lateral_velocity, max(self.ego_speed, 1.0))
        
        # Longitudinal dynamics
        # throttle -> acceleration -> speed
        max_accel = 3.0  # m/s²
        accel = throttle * max_accel
        self.ego_speed += accel * self.dt
        self.ego_speed = np.clip(self.ego_speed, 0.0, 35.0)
        
        # Position update
        self.ego_position += self.ego_speed * self.dt
        
        # Lane boundaries
        self.ego_lane_offset = np.clip(self.ego_lane_offset, -1.75, 1.75)
    
    def step(
        self,
        action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one timestep.
        
        Args:
            action: [steering, throttle]
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        # Update traffic
        self.traffic.step(self.ego_position)
        
        # Update ego dynamics
        self._update_ego_dynamics(action)
        
        # Get observation
        obs = self._get_observation()
        traffic_obs = self.traffic.get_observations_for_ego(
            self.ego_lane,
            self.ego_position
        )
        ttc = obs[7]
        
        # Check collision
        self.collision_occurred = self.traffic.check_collision(
            self.ego_position,
            self.ego_lane
        )
        
        if self.collision_occurred:
            self.stats['collisions'] += 1
        
        # Compute reward
        reward = self._compute_reward(action, traffic_obs, ttc)
        
        # Episode termination
        terminated = self.collision_occurred or abs(self.ego_lane_offset) > 1.75
        truncated = self.current_step >= self.max_episode_steps
        
        # Update statistics
        self.stats['avg_ttc'].append(ttc)
        self.stats['min_ttc'] = min(self.stats['min_ttc'], ttc)
        
        # Info
        info = {
            'collision': self.collision_occurred,
            'ttc': ttc,
            'lead_distance': traffic_obs['lead_distance'],
            'near_miss': ttc < 2.0,
            'ego_speed': self.ego_speed,
            'ego_position': self.ego_position
        }
        
        # Update state
        self.prev_steering = action[0]
        self.current_step += 1
        self.total_reward += reward
        
        return obs, reward, terminated, truncated, info
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment."""
        super().reset(seed=seed)
        
        # Reset traffic
        self.traffic.reset()
        
        # Reset ego
        self.ego_lane = 0
        self.ego_position = 0.0
        self.ego_speed = 15.0 + self.np_random.uniform(-2, 2)
        self.ego_lane_offset = self.np_random.uniform(-0.3, 0.3)
        self.ego_heading_error = self.np_random.uniform(-0.1, 0.1)
        
        # Reset episode state
        self.current_step = 0
        self.total_reward = 0.0
        self.collision_occurred = False
        self.prev_steering = 0.0
        
        # Get initial observation
        obs = self._get_observation()
        info = {'episode_start': True}
        
        return obs, info
    
    def get_episode_stats(self) -> Dict[str, Any]:
        """Get episode statistics."""
        return {
            'total_reward': self.total_reward,
            'collisions': self.stats['collisions'],
            'near_misses': self.stats['near_misses'],
            'avg_ttc': np.mean(self.stats['avg_ttc']) if self.stats['avg_ttc'] else 100.0,
            'min_ttc': self.stats['min_ttc'],
            'episode_length': self.current_step
        }
