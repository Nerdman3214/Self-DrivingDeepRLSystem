"""
Learning Rate and Hyperparameter Schedulers

Provides schedules for learning rate, exploration, and other hyperparameters.
"""

from typing import Callable, Union


class LinearSchedule:
    """
    Linear interpolation schedule.
    
    Useful for:
    - Learning rate decay
    - Exploration rate decay (epsilon-greedy)
    - Entropy coefficient annealing
    """
    
    def __init__(
        self,
        initial_value: float,
        final_value: float,
        schedule_timesteps: int
    ):
        """
        Args:
            initial_value: Starting value
            final_value: Ending value
            schedule_timesteps: Number of timesteps for the schedule
        """
        self.initial_value = initial_value
        self.final_value = final_value
        self.schedule_timesteps = schedule_timesteps
    
    def __call__(self, progress: float) -> float:
        """
        Get scheduled value.
        
        Args:
            progress: Current progress (0 to 1)
        
        Returns:
            Scheduled value
        """
        # progress is fraction of total training completed
        progress = min(progress, 1.0)
        return self.initial_value + progress * (self.final_value - self.initial_value)
    
    def value(self, timestep: int) -> float:
        """
        Get value at a specific timestep.
        
        Args:
            timestep: Current timestep
        
        Returns:
            Scheduled value
        """
        fraction = min(timestep / self.schedule_timesteps, 1.0)
        return self.__call__(fraction)


class ExponentialSchedule:
    """
    Exponential decay schedule.
    
    value = initial_value * decay^(timestep / decay_steps)
    """
    
    def __init__(
        self,
        initial_value: float,
        decay_rate: float,
        decay_steps: int,
        min_value: float = 0.0
    ):
        """
        Args:
            initial_value: Starting value
            decay_rate: Decay rate per decay_steps
            decay_steps: Steps per decay
            min_value: Minimum value
        """
        self.initial_value = initial_value
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.min_value = min_value
    
    def value(self, timestep: int) -> float:
        """Get value at a specific timestep."""
        decayed = self.initial_value * (self.decay_rate ** (timestep / self.decay_steps))
        return max(decayed, self.min_value)


class PiecewiseSchedule:
    """
    Piecewise linear schedule with multiple segments.
    """
    
    def __init__(
        self,
        endpoints: list,
        outside_value: float = None
    ):
        """
        Args:
            endpoints: List of (timestep, value) pairs
            outside_value: Value to use after last endpoint (default: last value)
        """
        self.endpoints = sorted(endpoints, key=lambda x: x[0])
        
        if outside_value is None:
            self.outside_value = self.endpoints[-1][1]
        else:
            self.outside_value = outside_value
    
    def value(self, timestep: int) -> float:
        """Get value at a specific timestep."""
        # Before first endpoint
        if timestep <= self.endpoints[0][0]:
            return self.endpoints[0][1]
        
        # After last endpoint
        if timestep >= self.endpoints[-1][0]:
            return self.outside_value
        
        # Find segment
        for i in range(len(self.endpoints) - 1):
            t1, v1 = self.endpoints[i]
            t2, v2 = self.endpoints[i + 1]
            
            if t1 <= timestep < t2:
                # Linear interpolation
                fraction = (timestep - t1) / (t2 - t1)
                return v1 + fraction * (v2 - v1)
        
        return self.outside_value


class ConstantSchedule:
    """Constant value schedule."""
    
    def __init__(self, value: float):
        self._value = value
    
    def __call__(self, progress: float) -> float:
        return self._value
    
    def value(self, timestep: int) -> float:
        return self._value


def get_schedule(
    value: Union[float, Callable, "LinearSchedule"]
) -> Callable[[float], float]:
    """
    Convert various inputs to a schedule function.
    
    Args:
        value: Constant, callable, or Schedule object
    
    Returns:
        Schedule function: progress -> value
    """
    if callable(value):
        return value
    elif isinstance(value, (int, float)):
        return lambda _: float(value)
    else:
        raise ValueError(f"Invalid schedule type: {type(value)}")


class WarmupSchedule:
    """
    Learning rate warmup followed by constant or decay.
    """
    
    def __init__(
        self,
        warmup_steps: int,
        initial_lr: float,
        target_lr: float,
        decay_schedule: Callable = None
    ):
        """
        Args:
            warmup_steps: Number of warmup steps
            initial_lr: Starting learning rate
            target_lr: Learning rate after warmup
            decay_schedule: Optional decay after warmup
        """
        self.warmup_steps = warmup_steps
        self.initial_lr = initial_lr
        self.target_lr = target_lr
        self.decay_schedule = decay_schedule
    
    def value(self, timestep: int) -> float:
        """Get learning rate at timestep."""
        if timestep < self.warmup_steps:
            # Linear warmup
            fraction = timestep / self.warmup_steps
            return self.initial_lr + fraction * (self.target_lr - self.initial_lr)
        
        if self.decay_schedule is None:
            return self.target_lr
        
        # Apply decay after warmup
        post_warmup_step = timestep - self.warmup_steps
        return self.decay_schedule.value(post_warmup_step)
