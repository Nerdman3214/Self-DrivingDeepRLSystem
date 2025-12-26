"""
PyBullet 3D Driving Environment

Realistic physics-based driving simulation using PyBullet.
Compatible with existing PPO training pipeline.

Phase 1 Enhancement: Abstract 2D → Realistic 3D Physics
"""

import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
import numpy as np
from typing import Tuple, Dict, Any, Optional
import math


class PyBulletDrivingEnv(gym.Env):
    """
    3D Driving Environment with PyBullet Physics
    
    Observation Space (6D - matches LaneKeepingEnv):
        - lane_offset: lateral distance from lane center [-2.0, 2.0]
        - heading_error: angle deviation from road direction [-π, π]
        - speed: current velocity [0, 30]
        - left_distance: distance to left boundary [0, 3]
        - right_distance: distance to right boundary [0, 3]
        - curvature: road curvature [-0.1, 0.1]
    
    Action Space (2D continuous):
        - steering: [-1, 1] (left/right)
        - throttle: [-1, 1] (brake/accelerate)
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}
    
    def __init__(
        self,
        render_mode: Optional[str] = None,
        max_episode_steps: int = 1000,
        lane_width: float = 3.5,
        target_speed: float = 20.0,
        **kwargs
    ):
        super().__init__()
        
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.lane_width = lane_width
        self.target_speed = target_speed
        
        # Episode tracking
        self.current_step = 0
        self.total_reward = 0.0
        
        # Action space: [steering, throttle]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        
        # Observation space: 6D to match existing system
        self.observation_space = spaces.Box(
            low=np.array([-2.0, -np.pi, 0.0, 0.0, 0.0, -0.1], dtype=np.float32),
            high=np.array([2.0, np.pi, 30.0, 3.0, 3.0, 0.1], dtype=np.float32),
            dtype=np.float32
        )
        
        # PyBullet connection
        self.client = None
        self.car = None
        self.plane = None
        self.road_markers = []
        
        # Initialize PyBullet
        self._connect_pybullet()
        
    def _connect_pybullet(self):
        """Connect to PyBullet physics engine"""
        if self.render_mode == "human":
            self.client = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        else:
            self.client = p.connect(p.DIRECT)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        p.setRealTimeSimulation(0)
        
    def _setup_world(self):
        """Create the driving environment"""
        # Load ground plane
        self.plane = p.loadURDF("plane.urdf")
        
        # Load car (racecar model from PyBullet)
        car_start_pos = [0, 0, 0.3]
        car_start_orientation = p.getQuaternionFromEuler([0, 0, 0])
        self.car = p.loadURDF(
            "racecar/racecar.urdf",
            car_start_pos,
            car_start_orientation
        )
        
        # Create visual lane markers
        self._create_lane_markers()
        
    def _create_lane_markers(self):
        """Create visual lane boundary markers"""
        # Clear old markers
        for marker in self.road_markers:
            p.removeBody(marker)
        self.road_markers = []
        
        # Create lane boundaries (simple cones for visualization)
        road_length = 100
        marker_spacing = 5
        
        for i in range(0, road_length, marker_spacing):
            # Left boundary
            left_pos = [-self.lane_width/2, i, 0.1]
            left_marker = p.loadURDF(
                "sphere2.urdf",
                left_pos,
                globalScaling=0.2
            )
            p.changeVisualShape(left_marker, -1, rgbaColor=[1, 1, 0, 1])
            self.road_markers.append(left_marker)
            
            # Right boundary
            right_pos = [self.lane_width/2, i, 0.1]
            right_marker = p.loadURDF(
                "sphere2.urdf",
                right_pos,
                globalScaling=0.2
            )
            p.changeVisualShape(right_marker, -1, rgbaColor=[1, 1, 0, 1])
            self.road_markers.append(right_marker)
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        # Reset PyBullet simulation
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        
        # Rebuild world
        self._setup_world()
        
        # Reset tracking variables
        self.current_step = 0
        self.total_reward = 0.0
        
        # Add small random perturbation to initial state
        if self.np_random is not None:
            initial_offset = self.np_random.uniform(-0.5, 0.5)
            initial_heading = self.np_random.uniform(-0.1, 0.1)
            
            # Apply perturbation
            pos, orn = p.getBasePositionAndOrientation(self.car)
            new_pos = [initial_offset, pos[1], pos[2]]
            new_orn = p.getQuaternionFromEuler([0, 0, initial_heading])
            p.resetBasePositionAndOrientation(self.car, new_pos, new_orn)
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(
        self,
        action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment
        
        Args:
            action: [steering, throttle] both in [-1, 1]
        
        Returns:
            observation, reward, terminated, truncated, info
        """
        steering, throttle = action
        
        # Apply steering (front wheels)
        max_steering_angle = 0.5  # ~30 degrees
        steering_angle = steering * max_steering_angle
        
        # Steering joints (typically joints 4 and 6 for racecar)
        p.setJointMotorControl2(
            self.car, 4,
            p.POSITION_CONTROL,
            targetPosition=steering_angle
        )
        p.setJointMotorControl2(
            self.car, 6,
            p.POSITION_CONTROL,
            targetPosition=steering_angle
        )
        
        # Apply throttle (rear wheels)
        max_force = 50
        target_velocity = throttle * 50  # Max ~50 rad/s
        
        # Drive joints (typically joints 2 and 3 for racecar)
        for wheel_joint in [2, 3, 5, 7]:  # All wheels for better control
            p.setJointMotorControl2(
                self.car, wheel_joint,
                p.VELOCITY_CONTROL,
                targetVelocity=target_velocity,
                force=max_force
            )
        
        # Step physics simulation
        p.stepSimulation()
        
        # Get new state
        observation = self._get_observation()
        reward = self._compute_reward(observation, action)
        terminated = self._check_terminated(observation)
        truncated = self.current_step >= self.max_episode_steps
        info = self._get_info()
        
        self.current_step += 1
        self.total_reward += reward
        
        return observation, reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """
        Get current observation (6D to match existing system)
        
        Returns:
            [lane_offset, heading_error, speed, left_dist, right_dist, curvature]
        """
        # Get car state
        pos, orn = p.getBasePositionAndOrientation(self.car)
        vel, ang_vel = p.getBaseVelocity(self.car)
        
        # Extract position and orientation
        x, y, z = pos
        euler = p.getEulerFromQuaternion(orn)
        yaw = euler[2]  # Heading angle
        
        # Compute observation components
        lane_offset = x  # Lateral offset from center (x=0 is center)
        heading_error = yaw  # Heading deviation (0 is forward along y-axis)
        
        # Speed (magnitude of velocity)
        speed = math.sqrt(vel[0]**2 + vel[1]**2)
        
        # Distance to lane boundaries
        left_distance = self.lane_width/2 + x
        right_distance = self.lane_width/2 - x
        
        # Curvature (0 for straight road, can be enhanced later)
        curvature = 0.0
        
        # Clip to observation space
        observation = np.array([
            np.clip(lane_offset, -2.0, 2.0),
            np.clip(heading_error, -np.pi, np.pi),
            np.clip(speed, 0.0, 30.0),
            np.clip(left_distance, 0.0, 3.0),
            np.clip(right_distance, 0.0, 3.0),
            np.clip(curvature, -0.1, 0.1)
        ], dtype=np.float32)
        
        return observation
    
    def _compute_reward(
        self,
        observation: np.ndarray,
        action: np.ndarray
    ) -> float:
        """
        Compute reward (matches existing reward structure)
        
        Reward components:
            - Lane centering: penalty for deviation
            - Heading alignment: penalty for angle error
            - Speed maintenance: reward for target speed
            - Smoothness: penalty for jerky actions
        """
        lane_offset, heading_error, speed, left_dist, right_dist, curvature = observation
        steering, throttle = action
        
        # Lane centering reward (exponential penalty)
        lane_penalty = -abs(lane_offset) ** 2
        
        # Heading alignment reward
        heading_penalty = -abs(heading_error) ** 2
        
        # Speed reward (maintain target speed)
        speed_error = abs(speed - self.target_speed)
        speed_reward = -speed_error * 0.1
        
        # Smoothness penalty (avoid jerky control)
        action_penalty = -(abs(steering) * 0.01 + abs(throttle) * 0.01)
        
        # Total reward
        reward = (
            lane_penalty * 1.0 +
            heading_penalty * 0.5 +
            speed_reward +
            action_penalty
        )
        
        # Bonus for staying centered
        if abs(lane_offset) < 0.5 and abs(heading_error) < 0.1:
            reward += 1.0
        
        return reward
    
    def _check_terminated(self, observation: np.ndarray) -> bool:
        """Check if episode should terminate (safety violation)"""
        lane_offset, heading_error, speed, left_dist, right_dist, curvature = observation
        
        # Terminate if car goes off road
        if abs(lane_offset) > 1.75:  # Lane width is 3.5m
            return True
        
        # Terminate if heading too wrong
        if abs(heading_error) > np.pi / 2:
            return True
        
        # Check if car flipped
        pos, orn = p.getBasePositionAndOrientation(self.car)
        euler = p.getEulerFromQuaternion(orn)
        if abs(euler[0]) > 0.5 or abs(euler[1]) > 0.5:  # Roll or pitch too large
            return True
        
        return False
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional information"""
        observation = self._get_observation()
        lane_offset, heading_error, speed, left_dist, right_dist, curvature = observation
        
        return {
            "lane_offset": float(lane_offset),
            "heading_error": float(heading_error),
            "speed": float(speed),
            "total_reward": float(self.total_reward),
            "steps": self.current_step,
        }
    
    def render(self):
        """Render environment (handled by PyBullet GUI)"""
        if self.render_mode == "human":
            # Camera follows car
            pos, _ = p.getBasePositionAndOrientation(self.car)
            p.resetDebugVisualizerCamera(
                cameraDistance=10,
                cameraYaw=0,
                cameraPitch=-30,
                cameraTargetPosition=pos
            )
        return None
    
    def close(self):
        """Cleanup PyBullet connection"""
        if self.client is not None:
            p.disconnect()
            self.client = None
