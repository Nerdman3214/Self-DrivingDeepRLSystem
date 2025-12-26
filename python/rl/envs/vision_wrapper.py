"""
Vision Environment Wrapper

Adds camera-based visual observations to existing environments.
Compatible with PyBullet and other Gymnasium environments.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
from typing import Optional, Tuple, Dict, Any
import cv2


class VisionWrapper(gym.ObservationWrapper):
    """
    Wraps environment to add visual observations from PyBullet camera.
    
    Modes:
        - 'vision_only': Only camera images
        - 'hybrid': Both images and state vectors
        - 'grayscale': Convert to grayscale
        - 'stack': Stack multiple frames (temporal information)
    
    Use Cases:
        - End-to-end learning from pixels
        - Realistic sensor simulation
        - Vision + state fusion
    """
    
    def __init__(
        self,
        env: gym.Env,
        image_size: int = 84,
        grayscale: bool = True,
        frame_stack: int = 1,
        mode: str = 'vision_only',
        camera_distance: float = 5.0,
        camera_yaw: float = 0,
        camera_pitch: float = -20
    ):
        """
        Args:
            env: Base environment (must use PyBullet)
            image_size: Size to resize images (square)
            grayscale: Convert to grayscale
            frame_stack: Number of frames to stack
            mode: 'vision_only', 'hybrid', or 'state_only'
            camera_distance: Camera distance from car
            camera_yaw: Camera yaw angle
            camera_pitch: Camera pitch angle
        """
        super().__init__(env)
        
        self.image_size = image_size
        self.grayscale = grayscale
        self.frame_stack = frame_stack
        self.mode = mode
        
        # Camera parameters
        self.camera_distance = camera_distance
        self.camera_yaw = camera_yaw
        self.camera_pitch = camera_pitch
        
        # Frame buffer for stacking
        self.frames = []
        
        # Update observation space
        self._update_observation_space()
    
    def _update_observation_space(self):
        """Create new observation space with vision"""
        channels = 1 if self.grayscale else 3
        channels *= self.frame_stack
        
        if self.mode == 'vision_only':
            # Only images
            self.observation_space = spaces.Box(
                low=0,
                high=255,
                shape=(channels, self.image_size, self.image_size),
                dtype=np.uint8
            )
        elif self.mode == 'hybrid':
            # Dictionary: both images and state
            self.observation_space = spaces.Dict({
                'image': spaces.Box(
                    low=0,
                    high=255,
                    shape=(channels, self.image_size, self.image_size),
                    dtype=np.uint8
                ),
                'state': self.env.observation_space
            })
    
    def _capture_image(self) -> np.ndarray:
        """
        Capture RGB image from PyBullet camera.
        
        Returns:
            image: [height, width, 3] RGB uint8 array
        """
        # Get car position for camera follow
        if hasattr(self.env.unwrapped, 'car'):
            car_pos, _ = p.getBasePositionAndOrientation(
                self.env.unwrapped.car
            )
        else:
            car_pos = [0, 0, 0]
        
        # Compute camera position
        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=car_pos,
            distance=self.camera_distance,
            yaw=self.camera_yaw,
            pitch=self.camera_pitch,
            roll=0,
            upAxisIndex=2
        )
        
        # Projection matrix
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60,
            aspect=1.0,
            nearVal=0.1,
            farVal=100.0
        )
        
        # Render
        width = height = 256  # Native resolution
        (_, _, px, _, _) = p.getCameraImage(
            width=width,
            height=height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )
        
        # Extract RGB
        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = rgb_array[:, :, :3]  # Remove alpha channel
        
        return rgb_array
    
    def _process_image(self, image: np.ndarray) -> np.ndarray:
        """
        Process image: resize, grayscale, etc.
        
        Args:
            image: [height, width, 3] RGB
        
        Returns:
            processed: [channels, image_size, image_size]
        """
        # Resize
        image = cv2.resize(
            image,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_AREA
        )
        
        # Grayscale
        if self.grayscale:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            image = np.expand_dims(image, axis=0)  # [1, H, W]
        else:
            image = np.transpose(image, (2, 0, 1))  # [3, H, W]
        
        return image
    
    def observation(self, observation: np.ndarray) -> Any:
        """
        Convert state observation to visual observation.
        
        Args:
            observation: State vector from base environment
        
        Returns:
            visual_obs: Image or dict with image+state
        """
        # Capture and process image
        raw_image = self._capture_image()
        processed_image = self._process_image(raw_image)
        
        # Frame stacking
        self.frames.append(processed_image)
        if len(self.frames) > self.frame_stack:
            self.frames.pop(0)
        
        # Pad if needed (first reset)
        while len(self.frames) < self.frame_stack:
            self.frames.append(processed_image)
        
        # Stack frames
        stacked_image = np.concatenate(self.frames, axis=0)
        
        # Return based on mode
        if self.mode == 'vision_only':
            return stacked_image
        elif self.mode == 'hybrid':
            return {
                'image': stacked_image,
                'state': observation
            }
        else:
            return observation
    
    def reset(self, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        """Reset and capture initial image"""
        self.frames = []
        obs, info = self.env.reset(**kwargs)
        visual_obs = self.observation(obs)
        return visual_obs, info


def make_vision_env(
    env_id: str = 'PyBulletDriving',
    image_size: int = 84,
    grayscale: bool = True,
    frame_stack: int = 4,
    mode: str = 'vision_only',
    **env_kwargs
) -> gym.Env:
    """
    Factory function to create vision-wrapped environment.
    
    Args:
        env_id: Environment ID or class name
        image_size: Image dimensions
        grayscale: Use grayscale
        frame_stack: Number of frames to stack
        mode: Observation mode
        **env_kwargs: Additional environment arguments
    
    Returns:
        Vision-wrapped environment
    
    Example:
        >>> env = make_vision_env('PyBulletDriving', image_size=84, frame_stack=4)
        >>> obs, _ = env.reset()
        >>> print(obs.shape)  # (4, 84, 84) for 4 grayscale frames
    """
    # Import environment class
    if env_id == 'PyBulletDriving':
        from rl.envs.pybullet_driving_env import PyBulletDrivingEnv
        base_env = PyBulletDrivingEnv(**env_kwargs)
    else:
        base_env = gym.make(env_id, **env_kwargs)
    
    # Wrap with vision
    vision_env = VisionWrapper(
        base_env,
        image_size=image_size,
        grayscale=grayscale,
        frame_stack=frame_stack,
        mode=mode
    )
    
    return vision_env
