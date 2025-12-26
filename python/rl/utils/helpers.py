"""
Utility functions for RL training.
"""

import torch
import numpy as np
import random
from typing import Optional


def set_seed(seed: int, deterministic: bool = False):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed
        deterministic: If True, use deterministic algorithms (slower)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(device: Optional[str] = None) -> torch.device:
    """
    Get the best available device.
    
    Args:
        device: Specific device to use ('cpu', 'cuda', 'cuda:0', etc.)
                If None, automatically select best available.
    
    Returns:
        torch.device
    """
    if device is not None:
        return torch.device(device)
    
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    Calculate explained variance.
    
    EV = 1 - Var(y_true - y_pred) / Var(y_true)
    
    Perfect predictions: EV = 1
    Constant predictions: EV = 0
    Worse than mean: EV < 0
    
    Args:
        y_pred: Predicted values
        y_true: True values
    
    Returns:
        Explained variance score
    """
    var_y = np.var(y_true)
    if var_y == 0:
        return 1.0 if np.var(y_true - y_pred) == 0 else 0.0
    return 1.0 - np.var(y_true - y_pred) / var_y


def polyak_update(
    source: torch.nn.Module,
    target: torch.nn.Module,
    tau: float
):
    """
    Soft update of target network parameters.
    
    target = tau * source + (1 - tau) * target
    
    Args:
        source: Source network
        target: Target network
        tau: Interpolation factor (0 < tau <= 1)
    """
    with torch.no_grad():
        for source_param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.mul_(1 - tau)
            target_param.data.add_(tau * source_param.data)


def get_linear_fn(start: float, end: float, end_fraction: float):
    """
    Create a linear interpolation function.
    
    Args:
        start: Start value
        end: End value
        end_fraction: Fraction of training where end value is reached
    
    Returns:
        Function that takes progress (0 to 1) and returns interpolated value
    """
    def func(progress: float) -> float:
        if progress >= end_fraction:
            return end
        return start + progress / end_fraction * (end - start)
    
    return func


def get_schedule_fn(value):
    """
    Convert a value to a schedule function.
    
    Args:
        value: Either a constant or a callable
    
    Returns:
        Schedule function
    """
    if callable(value):
        return value
    
    return lambda _: value


class RunningMeanStd:
    """
    Calculates running mean and standard deviation.
    
    Uses Welford's online algorithm for numerical stability.
    """
    
    def __init__(self, epsilon: float = 1e-4, shape: tuple = ()):
        """
        Args:
            epsilon: Small constant for numerical stability
            shape: Shape of the values being tracked
        """
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon
    
    def update(self, x: np.ndarray):
        """Update statistics with a batch of values."""
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)
    
    def _update_from_moments(
        self,
        batch_mean: np.ndarray,
        batch_var: np.ndarray,
        batch_count: int
    ):
        """Update from precomputed moments."""
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count
        new_var = m_2 / total_count
        
        self.mean = new_mean
        self.var = new_var
        self.count = total_count


def normalize_obs(
    obs: np.ndarray,
    obs_rms: RunningMeanStd,
    clip: float = 10.0
) -> np.ndarray:
    """
    Normalize observation using running statistics.
    
    Args:
        obs: Observation to normalize
        obs_rms: Running mean/std statistics
        clip: Clip normalized values to [-clip, clip]
    
    Returns:
        Normalized observation
    """
    return np.clip(
        (obs - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8),
        -clip,
        clip
    )


def normalize_reward(
    reward: np.ndarray,
    ret_rms: RunningMeanStd,
    clip: float = 10.0
) -> np.ndarray:
    """
    Normalize reward using running statistics.
    
    Args:
        reward: Reward to normalize
        ret_rms: Running mean/std of returns
        clip: Clip normalized values to [-clip, clip]
    
    Returns:
        Normalized reward
    """
    return np.clip(
        reward / np.sqrt(ret_rms.var + 1e-8),
        -clip,
        clip
    )
