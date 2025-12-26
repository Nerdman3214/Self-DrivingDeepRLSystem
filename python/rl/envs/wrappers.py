"""
Environment Wrappers for Self-Driving RL

Provides preprocessing wrappers for driving simulators:
- Frame stacking for temporal information
- Grayscale conversion
- Observation resizing
- Normalization
- CarRacing-specific wrapper
"""

import gymnasium as gym
import numpy as np
from collections import deque
from typing import Tuple, Optional, Dict, Any
import cv2


class FrameStack(gym.Wrapper):
    """
    Stack consecutive frames for temporal information.
    
    Useful for:
    - Understanding velocity/motion from static observations
    - Providing temporal context to the policy
    """
    
    def __init__(self, env: gym.Env, n_frames: int = 4):
        """
        Args:
            env: Environment to wrap
            n_frames: Number of frames to stack
        """
        super().__init__(env)
        self.n_frames = n_frames
        self.frames = deque(maxlen=n_frames)
        
        # Update observation space
        old_space = env.observation_space
        low = np.repeat(old_space.low[np.newaxis, ...], n_frames, axis=0)
        high = np.repeat(old_space.high[np.newaxis, ...], n_frames, axis=0)
        
        # Stack along first axis (for CHW format)
        if len(old_space.shape) == 3:
            # Image observation
            new_shape = (n_frames * old_space.shape[0],) + old_space.shape[1:]
            low = low.reshape(new_shape)
            high = high.reshape(new_shape)
        
        self.observation_space = gym.spaces.Box(
            low=low.astype(np.float32),
            high=high.astype(np.float32),
            dtype=np.float32
        )
    
    def reset(self, **kwargs) -> np.ndarray:
        obs = self.env.reset(**kwargs)
        if isinstance(obs, tuple):
            obs = obs[0]  # Handle gym 0.26+ API
        
        # Fill frames with initial observation
        for _ in range(self.n_frames):
            self.frames.append(obs)
        
        return self._get_observation()
    
    def step(self, action) -> Tuple[np.ndarray, float, bool, Dict]:
        obs, reward, done, *extra = self.env.step(action)
        self.frames.append(obs)
        
        if len(extra) == 2:
            # Gym 0.26+ API
            return self._get_observation(), reward, done, extra[0], extra[1]
        return self._get_observation(), reward, done, extra[0] if extra else {}
    
    def _get_observation(self) -> np.ndarray:
        """Stack frames along channel axis."""
        frames = np.array(self.frames)
        if frames.ndim == 4:
            # Stack along channel axis (assumes CHW format)
            return np.concatenate(frames, axis=0)
        return frames


class GrayScaleObservation(gym.ObservationWrapper):
    """
    Convert RGB observations to grayscale.
    
    Reduces input dimensionality while preserving important visual features.
    """
    
    def __init__(self, env: gym.Env, keep_dim: bool = False):
        """
        Args:
            env: Environment to wrap
            keep_dim: If True, keep channel dimension (H, W, 1)
        """
        super().__init__(env)
        self.keep_dim = keep_dim
        
        old_space = env.observation_space
        
        if keep_dim:
            new_shape = old_space.shape[:2] + (1,)
        else:
            new_shape = old_space.shape[:2]
        
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=255.0 if old_space.high.max() > 1.0 else 1.0,
            shape=new_shape,
            dtype=np.float32
        )
    
    def observation(self, obs: np.ndarray) -> np.ndarray:
        """Convert observation to grayscale."""
        # Handle different input formats
        if obs.shape[-1] == 3:
            # RGB -> Grayscale using luminosity method
            gray = 0.299 * obs[..., 0] + 0.587 * obs[..., 1] + 0.114 * obs[..., 2]
        elif obs.shape[-1] == 4:
            # RGBA -> Grayscale
            gray = 0.299 * obs[..., 0] + 0.587 * obs[..., 1] + 0.114 * obs[..., 2]
        else:
            gray = obs
        
        if self.keep_dim:
            gray = gray[..., np.newaxis]
        
        return gray.astype(np.float32)


class ResizeObservation(gym.ObservationWrapper):
    """
    Resize observations to a target shape.
    
    Useful for:
    - Reducing computational requirements
    - Standardizing input size across different simulators
    """
    
    def __init__(self, env: gym.Env, shape: Tuple[int, int]):
        """
        Args:
            env: Environment to wrap
            shape: Target (height, width)
        """
        super().__init__(env)
        self.shape = shape
        
        old_space = env.observation_space
        
        # Determine new shape based on input format
        if len(old_space.shape) == 3:
            new_shape = shape + (old_space.shape[-1],)
        else:
            new_shape = shape
        
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=255.0 if old_space.high.max() > 1.0 else 1.0,
            shape=new_shape,
            dtype=np.float32
        )
    
    def observation(self, obs: np.ndarray) -> np.ndarray:
        """Resize observation."""
        resized = cv2.resize(
            obs.astype(np.float32),
            (self.shape[1], self.shape[0]),  # cv2 uses (width, height)
            interpolation=cv2.INTER_AREA
        )
        
        # Restore channel dimension if needed
        if len(self.observation_space.shape) == 3 and resized.ndim == 2:
            resized = resized[..., np.newaxis]
        
        return resized


class NormalizeObservation(gym.ObservationWrapper):
    """
    Normalize observations to [0, 1] range.
    """
    
    def __init__(self, env: gym.Env):
        super().__init__(env)
        
        old_space = env.observation_space
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=old_space.shape,
            dtype=np.float32
        )
    
    def observation(self, obs: np.ndarray) -> np.ndarray:
        """Normalize observation to [0, 1]."""
        if obs.max() > 1.0:
            return obs.astype(np.float32) / 255.0
        return obs.astype(np.float32)


class TransposeObservation(gym.ObservationWrapper):
    """
    Transpose observation from HWC to CHW format.
    
    Required for PyTorch CNN processing.
    """
    
    def __init__(self, env: gym.Env):
        super().__init__(env)
        
        old_space = env.observation_space
        
        if len(old_space.shape) == 3:
            # HWC -> CHW
            new_shape = (old_space.shape[-1], old_space.shape[0], old_space.shape[1])
        else:
            new_shape = old_space.shape
        
        self.observation_space = gym.spaces.Box(
            low=old_space.low.transpose((2, 0, 1)) if len(old_space.shape) == 3 else old_space.low,
            high=old_space.high.transpose((2, 0, 1)) if len(old_space.shape) == 3 else old_space.high,
            shape=new_shape,
            dtype=np.float32
        )
    
    def observation(self, obs: np.ndarray) -> np.ndarray:
        """Transpose HWC to CHW."""
        if obs.ndim == 3:
            return obs.transpose((2, 0, 1)).astype(np.float32)
        return obs.astype(np.float32)


class ClipReward(gym.RewardWrapper):
    """
    Clip rewards to a specified range.
    
    Helps stabilize training by preventing extreme reward values.
    """
    
    def __init__(self, env: gym.Env, min_reward: float = -1.0, max_reward: float = 1.0):
        super().__init__(env)
        self.min_reward = min_reward
        self.max_reward = max_reward
    
    def reward(self, reward: float) -> float:
        return np.clip(reward, self.min_reward, self.max_reward)


class CarRacingWrapper(gym.Wrapper):
    """
    Complete wrapper for CarRacing-v2 environment.
    
    Combines:
    - Observation preprocessing (resize, normalize, transpose)
    - Action space handling
    - Episode termination conditions
    - Reward shaping
    """
    
    def __init__(
        self,
        env: gym.Env,
        frame_skip: int = 4,
        grayscale: bool = False,
        resize_shape: Tuple[int, int] = (96, 96),
        negative_reward_limit: float = -100.0,
        max_steps: int = 1000
    ):
        """
        Args:
            env: CarRacing environment
            frame_skip: Number of frames to repeat each action
            grayscale: Whether to convert to grayscale
            resize_shape: Target observation shape (height, width)
            negative_reward_limit: Terminate if cumulative reward below this
            max_steps: Maximum steps per episode
        """
        super().__init__(env)
        
        self.frame_skip = frame_skip
        self.grayscale = grayscale
        self.resize_shape = resize_shape
        self.negative_reward_limit = negative_reward_limit
        self.max_steps = max_steps
        
        self.cumulative_reward = 0.0
        self.step_count = 0
        
        # Determine observation shape
        n_channels = 1 if grayscale else 3
        obs_shape = (n_channels, resize_shape[0], resize_shape[1])
        
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=obs_shape,
            dtype=np.float32
        )
        
        # Keep original action space (continuous)
        self.action_space = env.action_space
    
    def reset(self, **kwargs) -> np.ndarray:
        self.cumulative_reward = 0.0
        self.step_count = 0
        
        obs = self.env.reset(**kwargs)
        if isinstance(obs, tuple):
            obs = obs[0]
        
        return self._process_observation(obs)
    
    def step(self, action) -> Tuple[np.ndarray, float, bool, Dict]:
        total_reward = 0.0
        done = False
        info = {}
        
        # Frame skipping
        for _ in range(self.frame_skip):
            obs, reward, done, *extra = self.env.step(action)
            total_reward += reward
            self.step_count += 1
            
            if done:
                break
        
        self.cumulative_reward += total_reward
        
        # Early termination conditions
        if self.cumulative_reward < self.negative_reward_limit:
            done = True
            info['early_termination'] = 'negative_reward'
        
        if self.step_count >= self.max_steps:
            done = True
            info['early_termination'] = 'max_steps'
        
        processed_obs = self._process_observation(obs)
        
        if len(extra) == 2:
            return processed_obs, total_reward, done, extra[0], {**extra[1], **info}
        return processed_obs, total_reward, done, {**extra[0], **info} if extra else info
    
    def _process_observation(self, obs: np.ndarray) -> np.ndarray:
        """Process raw observation."""
        # Resize
        if obs.shape[:2] != self.resize_shape:
            obs = cv2.resize(
                obs,
                (self.resize_shape[1], self.resize_shape[0]),
                interpolation=cv2.INTER_AREA
            )
        
        # Grayscale conversion
        if self.grayscale:
            obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
            obs = obs[..., np.newaxis]
        
        # Normalize to [0, 1]
        obs = obs.astype(np.float32) / 255.0
        
        # HWC -> CHW
        if obs.ndim == 3:
            obs = obs.transpose((2, 0, 1))
        else:
            obs = obs[np.newaxis, ...]
        
        return obs


class RecordEpisodeStatistics(gym.Wrapper):
    """
    Record episode statistics for logging.
    """
    
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_reward = 0.0
        self.current_length = 0
    
    def reset(self, **kwargs):
        self.current_reward = 0.0
        self.current_length = 0
        return self.env.reset(**kwargs)
    
    def step(self, action):
        result = self.env.step(action)
        obs, reward, done = result[0], result[1], result[2]
        
        self.current_reward += reward
        self.current_length += 1
        
        info = result[-1] if isinstance(result[-1], dict) else {}
        
        if done:
            info['episode'] = {
                'r': self.current_reward,
                'l': self.current_length
            }
            self.episode_rewards.append(self.current_reward)
            self.episode_lengths.append(self.current_length)
        
        if len(result) == 5:
            return obs, reward, done, result[3], info
        return obs, reward, done, info


def make_car_racing_env(
    render_mode: Optional[str] = None,
    grayscale: bool = False,
    frame_skip: int = 4,
    resize_shape: Tuple[int, int] = (96, 96)
) -> gym.Env:
    """
    Create a wrapped CarRacing-v2 environment.
    
    Args:
        render_mode: 'human' for visualization, None for headless
        grayscale: Whether to convert to grayscale
        frame_skip: Action repeat frames
        resize_shape: Target observation size
    
    Returns:
        Wrapped CarRacing environment
    """
    # Create base environment
    env = gym.make(
        "CarRacing-v2",
        continuous=True,
        render_mode=render_mode
    )
    
    # Apply wrappers
    env = RecordEpisodeStatistics(env)
    env = CarRacingWrapper(
        env,
        frame_skip=frame_skip,
        grayscale=grayscale,
        resize_shape=resize_shape
    )
    
    return env
