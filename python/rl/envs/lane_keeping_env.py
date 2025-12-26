"""
Lane-Keeping Gym Environment for Self-Driving RL

A custom environment that teaches fundamental self-driving skills:
- Lane centering
- Heading alignment  
- Speed control
- Smooth control

This mirrors real autonomous vehicle state estimation layers.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Optional, Dict, Any
import math


class LaneKeepingEnv(gym.Env):
    """
    Lane-keeping environment for self-driving RL.
    
    State Space (6D):
        0. lane_offset:    Distance from lane center [-1, 1]
        1. heading_error:  Angle relative to lane [-π/4, π/4]
        2. speed:          Current speed [0, 1]
        3. left_lane_dist: Distance to left boundary [0, 1]
        4. right_lane_dist: Distance to right boundary [0, 1]
        5. curvature:      Road curvature ahead [-1, 1]
    
    Action Space (2D continuous):
        0. steering:  Wheel angle [-1, 1]
        1. throttle:  Acceleration/brake [-1, 1]
    
    Reward Function:
        R = w1·(1-|lane_offset|) - w2·|heading_error| - w3·|steering| - w4·collision
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(
        self,
        lane_width: float = 3.5,  # meters
        max_speed: float = 30.0,  # m/s (~108 km/h)
        dt: float = 0.05,  # 20 Hz control
        max_episode_steps: int = 1000,
        reward_weights: Optional[Dict[str, float]] = None,
        random_start: bool = True,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        
        # Environment parameters
        self.lane_width = lane_width
        self.max_speed = max_speed
        self.dt = dt
        self.max_episode_steps = max_episode_steps
        self.random_start = random_start
        self.render_mode = render_mode
        
        # Reward weights (default values)
        self.reward_weights = reward_weights or {
            'lane_centering': 1.0,
            'heading_alignment': 0.5,
            'smooth_steering': 0.1,
            'collision': 10.0,
            'speed_target': 0.3,
        }
        
        # Observation space: [lane_offset, heading_error, speed, left_dist, right_dist, curvature]
        self.observation_space = spaces.Box(
            low=np.array([-1.0, -np.pi/4, 0.0, 0.0, 0.0, -1.0], dtype=np.float32),
            high=np.array([1.0, np.pi/4, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        
        # Action space: [steering, throttle]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        
        # Vehicle state
        self.lateral_position = 0.0  # meters from lane center
        self.heading = 0.0  # radians from lane direction
        self.speed = 0.0  # m/s
        self.longitudinal_position = 0.0  # meters along road
        
        # Road parameters
        self.road_curvature = 0.0
        self.road_length = 1000.0  # meters
        
        # Episode tracking
        self.steps = 0
        self.previous_steering = 0.0
        
        # Rendering
        self.viewer = None
        
    def reset(
        self, 
        *, 
        seed: Optional[int] = None, 
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        if self.random_start:
            # Random initial conditions (slight perturbations)
            self.lateral_position = self.np_random.uniform(-0.3, 0.3)
            self.heading = self.np_random.uniform(-0.1, 0.1)
            self.speed = self.np_random.uniform(0.3, 0.5) * self.max_speed
        else:
            # Deterministic start
            self.lateral_position = 0.0
            self.heading = 0.0
            self.speed = 0.4 * self.max_speed
        
        self.longitudinal_position = 0.0
        self.steps = 0
        self.previous_steering = 0.0
        
        # Generate road curvature profile
        self._generate_road_profile()
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one environment step.
        
        Args:
            action: [steering, throttle]
        
        Returns:
            observation, reward, terminated, truncated, info
        """
        # Clamp actions to valid range
        steering = np.clip(action[0], -1.0, 1.0)
        throttle = np.clip(action[1], -1.0, 1.0)
        
        # Store for smoothness penalty
        steering_change = abs(steering - self.previous_steering)
        self.previous_steering = steering
        
        # === Vehicle Dynamics ===
        # Bicycle model (simplified lateral dynamics)
        max_steering_angle = np.pi / 6  # 30 degrees
        wheel_angle = steering * max_steering_angle
        
        # Lateral dynamics
        # v_lateral = speed * sin(heading + wheel_angle)
        lateral_velocity = self.speed * np.sin(self.heading + wheel_angle * 0.5)
        self.lateral_position += lateral_velocity * self.dt
        
        # Heading dynamics
        # Assumes wheelbase of 2.5m
        wheelbase = 2.5
        heading_rate = (self.speed / wheelbase) * np.tan(wheel_angle)
        self.heading += heading_rate * self.dt
        
        # Account for road curvature
        road_heading_change = self.road_curvature * self.speed * self.dt
        self.heading -= road_heading_change  # Subtract because we want heading error
        
        # Longitudinal dynamics
        max_accel = 5.0  # m/s^2
        max_brake = 8.0  # m/s^2
        
        if throttle >= 0:
            acceleration = throttle * max_accel
        else:
            acceleration = throttle * max_brake
        
        self.speed += acceleration * self.dt
        self.speed = np.clip(self.speed, 0.0, self.max_speed)
        
        self.longitudinal_position += self.speed * self.dt
        
        # Update road curvature based on position
        self._update_road_curvature()
        
        # === Compute Reward ===
        reward = self._compute_reward(steering, throttle, steering_change)
        
        # === Check Termination ===
        terminated = False
        truncated = False
        
        # Off-road detection
        lane_offset_normalized = self.lateral_position / (self.lane_width / 2)
        if abs(lane_offset_normalized) > 1.0:
            terminated = True
            reward -= self.reward_weights['collision']
        
        # Heading too far off
        if abs(self.heading) > np.pi / 4:
            terminated = True
            reward -= self.reward_weights['collision']
        
        # Episode length limit
        self.steps += 1
        if self.steps >= self.max_episode_steps:
            truncated = True
        
        # Reached end of road
        if self.longitudinal_position >= self.road_length:
            truncated = True
            reward += 10.0  # Bonus for completing road
        
        observation = self._get_observation()
        info = self._get_info()
        info['steering_change'] = steering_change
        
        return observation, reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """
        Get current observation vector.
        
        Returns:
            [lane_offset, heading_error, speed, left_dist, right_dist, curvature]
        """
        # Normalize lateral position to [-1, 1]
        lane_offset = np.clip(
            self.lateral_position / (self.lane_width / 2), 
            -1.0, 1.0
        )
        
        # Heading error (already in radians)
        heading_error = np.clip(self.heading, -np.pi/4, np.pi/4)
        
        # Normalized speed
        speed_normalized = self.speed / self.max_speed
        
        # Distance to lane boundaries (normalized)
        half_width = self.lane_width / 2
        left_dist = (half_width - self.lateral_position) / half_width
        right_dist = (half_width + self.lateral_position) / half_width
        left_dist = np.clip(left_dist, 0.0, 1.0)
        right_dist = np.clip(right_dist, 0.0, 1.0)
        
        # Road curvature (already normalized)
        curvature = np.clip(self.road_curvature, -1.0, 1.0)
        
        observation = np.array([
            lane_offset,
            heading_error,
            speed_normalized,
            left_dist,
            right_dist,
            curvature
        ], dtype=np.float32)
        
        return observation
    
    def _compute_reward(
        self, 
        steering: float, 
        throttle: float,
        steering_change: float
    ) -> float:
        """
        Compute reward based on current state and action.
        
        Reward function:
            R = w1·(1-|lane_offset|) - w2·|heading_error| - w3·|steering| - w4·collision
        """
        w = self.reward_weights
        
        # Lane centering reward (0 to 1)
        lane_offset_normalized = abs(self.lateral_position / (self.lane_width / 2))
        lane_centering = (1.0 - lane_offset_normalized)
        
        # Heading alignment reward
        heading_alignment = -abs(self.heading)
        
        # Smooth steering penalty
        smooth_steering = -abs(steering)
        
        # Smooth steering change penalty
        steering_jerk = -steering_change
        
        # Speed target reward (encourage maintaining reasonable speed)
        target_speed = 0.7  # 70% of max
        speed_normalized = self.speed / self.max_speed
        speed_reward = -(abs(speed_normalized - target_speed))
        
        # Total reward
        reward = (
            w['lane_centering'] * lane_centering +
            w['heading_alignment'] * heading_alignment +
            w['smooth_steering'] * smooth_steering +
            w['speed_target'] * speed_reward +
            0.05 * steering_jerk  # Minor jerk penalty
        )
        
        return float(reward)
    
    def _generate_road_profile(self):
        """Generate a road curvature profile for the episode."""
        # Simple sinusoidal road with random parameters
        self.road_frequency = self.np_random.uniform(0.01, 0.03)
        self.road_amplitude = self.np_random.uniform(0.3, 0.8)
        self.road_phase = self.np_random.uniform(0, 2 * np.pi)
    
    def _update_road_curvature(self):
        """Update road curvature based on longitudinal position."""
        # Sinusoidal curvature
        position_rad = self.longitudinal_position * self.road_frequency + self.road_phase
        self.road_curvature = self.road_amplitude * np.sin(position_rad)
        
        # Add occasional sharper turns
        if self.longitudinal_position % 200 < 50:
            turn_intensity = self.np_random.uniform(-0.5, 0.5)
            self.road_curvature += turn_intensity
        
        self.road_curvature = np.clip(self.road_curvature, -1.0, 1.0)
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional info dict."""
        return {
            'lateral_position': self.lateral_position,
            'heading': self.heading,
            'speed': self.speed,
            'longitudinal_position': self.longitudinal_position,
            'road_curvature': self.road_curvature,
            'steps': self.steps,
            'lane_offset_normalized': self.lateral_position / (self.lane_width / 2),
        }
    
    def render(self):
        """Render the environment (optional, for visualization)."""
        if self.render_mode == "rgb_array":
            return self._render_frame()
        elif self.render_mode == "human":
            if self.viewer is None:
                # Lazy import to avoid pygame dependency if not rendering
                try:
                    import pygame
                    pygame.init()
                    self.viewer = pygame.display.set_mode((800, 400))
                    pygame.display.set_caption("Lane Keeping")
                except ImportError:
                    print("Warning: pygame not available, rendering disabled")
                    return None
            
            return self._render_frame()
    
    def _render_frame(self):
        """Internal rendering logic."""
        try:
            import pygame
            
            if self.viewer is None:
                return None
            
            # Clear screen
            self.viewer.fill((50, 50, 50))
            
            # Draw lane
            lane_center_x = 400
            lane_width_px = 150
            
            # Lane boundaries
            pygame.draw.line(
                self.viewer, (255, 255, 255),
                (lane_center_x - lane_width_px // 2, 0),
                (lane_center_x - lane_width_px // 2, 400),
                3
            )
            pygame.draw.line(
                self.viewer, (255, 255, 255),
                (lane_center_x + lane_width_px // 2, 0),
                (lane_center_x + lane_width_px // 2, 400),
                3
            )
            
            # Lane center line (dashed)
            for y in range(0, 400, 40):
                pygame.draw.line(
                    self.viewer, (255, 255, 0),
                    (lane_center_x, y),
                    (lane_center_x, y + 20),
                    2
                )
            
            # Draw vehicle
            vehicle_x = lane_center_x + int(self.lateral_position / (self.lane_width / 2) * lane_width_px / 2)
            vehicle_y = 300
            vehicle_width = 30
            vehicle_height = 50
            
            # Rotate vehicle based on heading
            vehicle_surface = pygame.Surface((vehicle_width, vehicle_height), pygame.SRCALPHA)
            pygame.draw.rect(vehicle_surface, (0, 150, 255), (0, 0, vehicle_width, vehicle_height))
            
            rotated = pygame.transform.rotate(vehicle_surface, -math.degrees(self.heading))
            rect = rotated.get_rect(center=(vehicle_x, vehicle_y))
            self.viewer.blit(rotated, rect.topleft)
            
            # Display info
            font = pygame.font.Font(None, 24)
            info_texts = [
                f"Speed: {self.speed:.1f} m/s",
                f"Lane Offset: {self.lateral_position:.2f} m",
                f"Heading: {math.degrees(self.heading):.1f}°",
                f"Position: {self.longitudinal_position:.0f} m",
            ]
            
            for i, text in enumerate(info_texts):
                surface = font.render(text, True, (255, 255, 255))
                self.viewer.blit(surface, (10, 10 + i * 30))
            
            pygame.display.flip()
            
            if self.render_mode == "rgb_array":
                return pygame.surfarray.array3d(self.viewer)
            
        except ImportError:
            pass
        
        return None
    
    def close(self):
        """Clean up resources."""
        if self.viewer is not None:
            try:
                import pygame
                pygame.quit()
            except ImportError:
                pass
            self.viewer = None


def make_lane_keeping_env(**kwargs) -> LaneKeepingEnv:
    """
    Factory function to create lane-keeping environment.
    
    Args:
        **kwargs: Arguments passed to LaneKeepingEnv
    
    Returns:
        LaneKeepingEnv instance
    """
    return LaneKeepingEnv(**kwargs)
