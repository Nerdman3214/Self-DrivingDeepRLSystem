"""
Vectorized Environment Implementation

Provides parallel environment execution for faster data collection.
"""

import numpy as np
import multiprocessing as mp
from abc import ABC, abstractmethod
from typing import Tuple, List, Optional, Callable, Any
import gymnasium as gym


class VecEnv(ABC):
    """
    Abstract base class for vectorized environments.
    """
    
    def __init__(self, num_envs: int, observation_space, action_space):
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.action_space = action_space
    
    @abstractmethod
    def reset(self) -> np.ndarray:
        """Reset all environments."""
        pass
    
    @abstractmethod
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[dict]]:
        """
        Step all environments.
        
        Args:
            actions: Array of actions for each environment
        
        Returns:
            observations: Stacked observations
            rewards: Array of rewards
            dones: Array of done flags
            infos: List of info dicts
        """
        pass
    
    @abstractmethod
    def close(self):
        """Clean up resources."""
        pass
    
    def env_is_wrapped(self, wrapper_class) -> List[bool]:
        """Check if environments are wrapped with a specific wrapper."""
        return [False] * self.num_envs


class DummyVecEnv(VecEnv):
    """
    Vectorized environment that runs environments sequentially.
    
    Simple implementation suitable for:
    - Debugging
    - Small number of environments
    - When multiprocessing overhead is too high
    """
    
    def __init__(self, env_fns: List[Callable[[], gym.Env]]):
        """
        Args:
            env_fns: List of environment constructor functions
        """
        self.envs = [fn() for fn in env_fns]
        
        env = self.envs[0]
        super().__init__(
            num_envs=len(env_fns),
            observation_space=env.observation_space,
            action_space=env.action_space
        )
        
        self.buf_obs = np.zeros(
            (self.num_envs,) + self.observation_space.shape,
            dtype=self.observation_space.dtype
        )
        self.buf_dones = np.zeros(self.num_envs, dtype=bool)
        self.buf_rews = np.zeros(self.num_envs, dtype=np.float32)
        self.buf_infos = [{} for _ in range(self.num_envs)]
    
    def reset(self) -> np.ndarray:
        for i, env in enumerate(self.envs):
            obs = env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]
            self.buf_obs[i] = obs
        return self.buf_obs.copy()
    
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[dict]]:
        for i, (env, action) in enumerate(zip(self.envs, actions)):
            result = env.step(action)
            
            if len(result) == 5:
                obs, reward, terminated, truncated, info = result
                done = terminated or truncated
            else:
                obs, reward, done, info = result
            
            self.buf_obs[i] = obs
            self.buf_rews[i] = reward
            self.buf_dones[i] = done
            self.buf_infos[i] = info
            
            if done:
                # Auto-reset
                reset_obs = env.reset()
                if isinstance(reset_obs, tuple):
                    reset_obs = reset_obs[0]
                self.buf_obs[i] = reset_obs
                info['terminal_observation'] = obs
        
        return (
            self.buf_obs.copy(),
            self.buf_rews.copy(),
            self.buf_dones.copy(),
            self.buf_infos.copy()
        )
    
    def close(self):
        for env in self.envs:
            env.close()
    
    def render(self, mode: str = 'human'):
        return self.envs[0].render()
    
    def env_method(self, method_name: str, *args, indices: Optional[List[int]] = None, **kwargs):
        """Call a method on specified environments."""
        if indices is None:
            indices = range(self.num_envs)
        return [getattr(self.envs[i], method_name)(*args, **kwargs) for i in indices]


def _worker(
    remote: mp.connection.Connection,
    parent_remote: mp.connection.Connection,
    env_fn_wrapper
):
    """
    Worker function for subprocess environments.
    """
    parent_remote.close()
    env = env_fn_wrapper.fn()
    
    while True:
        try:
            cmd, data = remote.recv()
            
            if cmd == 'step':
                result = env.step(data)
                if len(result) == 5:
                    obs, reward, terminated, truncated, info = result
                    done = terminated or truncated
                else:
                    obs, reward, done, info = result
                
                if done:
                    info['terminal_observation'] = obs
                    reset_obs = env.reset()
                    if isinstance(reset_obs, tuple):
                        reset_obs = reset_obs[0]
                    obs = reset_obs
                
                remote.send((obs, reward, done, info))
            
            elif cmd == 'reset':
                obs = env.reset()
                if isinstance(obs, tuple):
                    obs = obs[0]
                remote.send(obs)
            
            elif cmd == 'close':
                env.close()
                remote.close()
                break
            
            elif cmd == 'get_spaces':
                remote.send((env.observation_space, env.action_space))
            
            elif cmd == 'env_method':
                method = getattr(env, data[0])
                remote.send(method(*data[1], **data[2]))
            
            else:
                raise NotImplementedError(f"Unknown command: {cmd}")
        
        except EOFError:
            break


class CloudpickleWrapper:
    """
    Wrapper for functions to enable pickling with cloudpickle.
    """
    def __init__(self, fn):
        self.fn = fn
    
    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.fn)
    
    def __setstate__(self, fn):
        import cloudpickle
        self.fn = cloudpickle.loads(fn)


class SubprocVecEnv(VecEnv):
    """
    Vectorized environment using subprocesses for parallel execution.
    
    Best for:
    - CPU-bound environments
    - Large number of parallel environments
    - When environments don't share GPU resources
    """
    
    def __init__(self, env_fns: List[Callable[[], gym.Env]], start_method: str = 'forkserver'):
        """
        Args:
            env_fns: List of environment constructor functions
            start_method: Multiprocessing start method ('fork', 'spawn', 'forkserver')
        """
        self.waiting = False
        self.closed = False
        
        n_envs = len(env_fns)
        
        # Create multiprocessing context
        ctx = mp.get_context(start_method)
        
        # Create pipes for communication
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(n_envs)])
        
        # Start worker processes
        self.processes = []
        for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes, env_fns):
            args = (work_remote, remote, CloudpickleWrapper(env_fn))
            process = ctx.Process(target=_worker, args=args, daemon=True)
            process.start()
            self.processes.append(process)
            work_remote.close()
        
        # Get observation and action spaces from first environment
        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()
        
        super().__init__(n_envs, observation_space, action_space)
        
        self.buf_obs = np.zeros(
            (self.num_envs,) + self.observation_space.shape,
            dtype=self.observation_space.dtype
        )
    
    def reset(self) -> np.ndarray:
        for remote in self.remotes:
            remote.send(('reset', None))
        
        for i, remote in enumerate(self.remotes):
            self.buf_obs[i] = remote.recv()
        
        return self.buf_obs.copy()
    
    def step_async(self, actions: np.ndarray):
        """Send actions to environments asynchronously."""
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True
    
    def step_wait(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[dict]]:
        """Wait for step results from all environments."""
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        
        obs, rews, dones, infos = zip(*results)
        
        self.buf_obs = np.stack(obs)
        
        return (
            self.buf_obs.copy(),
            np.array(rews, dtype=np.float32),
            np.array(dones, dtype=bool),
            list(infos)
        )
    
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[dict]]:
        """Step all environments."""
        self.step_async(actions)
        return self.step_wait()
    
    def close(self):
        if self.closed:
            return
        
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        
        for remote in self.remotes:
            remote.send(('close', None))
        
        for process in self.processes:
            process.join()
        
        self.closed = True
    
    def env_method(
        self,
        method_name: str,
        *args,
        indices: Optional[List[int]] = None,
        **kwargs
    ) -> List[Any]:
        """Call a method on specified environments."""
        if indices is None:
            indices = range(self.num_envs)
        
        for i in indices:
            self.remotes[i].send(('env_method', (method_name, args, kwargs)))
        
        return [self.remotes[i].recv() for i in indices]


def make_vec_env(
    env_fn: Callable[[], gym.Env],
    n_envs: int = 1,
    use_subprocess: bool = False,
    start_method: str = 'forkserver'
) -> VecEnv:
    """
    Create a vectorized environment.
    
    Args:
        env_fn: Environment constructor function
        n_envs: Number of parallel environments
        use_subprocess: Use SubprocVecEnv instead of DummyVecEnv
        start_method: Multiprocessing start method
    
    Returns:
        Vectorized environment
    """
    env_fns = [env_fn for _ in range(n_envs)]
    
    if use_subprocess and n_envs > 1:
        return SubprocVecEnv(env_fns, start_method=start_method)
    else:
        return DummyVecEnv(env_fns)
