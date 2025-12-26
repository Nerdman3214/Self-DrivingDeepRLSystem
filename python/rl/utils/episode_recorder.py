"""
Episode Recorder for Autonomous Driving

Implements Event Sourcing pattern to record complete inference episodes.
This enables deterministic replay, debugging, and safety auditing.

Design Patterns:
- Event Sourcing: Every decision is an immutable event
- Observer Pattern: Non-intrusive observation of inference loop
- Separation of Concerns: Record without interfering
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import gzip


def _make_json_serializable(obj):
    """Convert numpy types to Python types for JSON serialization."""
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_json_serializable(item) for item in obj]
    elif isinstance(obj, bool):
        return bool(obj)
    else:
        return obj


class EpisodeRecorder:
    """
    Records complete inference episodes for deterministic playback.
    
    Each timestep is a complete snapshot:
    - State (observations)
    - Policy output (raw action)
    - Safety output (filtered action)
    - Safety interventions (flags)
    - Metadata (rewards, termination, etc.)
    
    This is how real autonomy teams debug policies.
    """
    
    def __init__(
        self,
        output_dir: str = "episodes",
        compress: bool = True,
        verbose: bool = True
    ):
        """
        Initialize episode recorder.
        
        Args:
            output_dir: Directory to save episode files
            compress: Whether to gzip compress episode files
            verbose: Print recording status
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.compress = compress
        self.verbose = verbose
        
        # Current episode data
        self.current_episode: List[Dict[str, Any]] = []
        self.episode_metadata: Dict[str, Any] = {}
        self.recording = False
        
        # Statistics
        self.episodes_recorded = 0
        
    def start_episode(
        self,
        episode_id: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Start recording a new episode.
        
        Args:
            episode_id: Optional episode identifier
            metadata: Optional metadata (model info, config, etc.)
        """
        if self.recording:
            raise RuntimeError("Episode already in progress. Call end_episode() first.")
        
        self.current_episode = []
        self.episode_metadata = {
            "episode_id": episode_id if episode_id is not None else self.episodes_recorded,
            "start_time": datetime.now().isoformat(),
            "recorder_version": "1.0",
            **(metadata or {})
        }
        self.recording = True
        
        if self.verbose:
            print(f"📼 Recording Episode {self.episode_metadata['episode_id']}...")
    
    def record_timestep(
        self,
        timestep: int,
        state: np.ndarray,
        policy_action: np.ndarray,
        safety_action: np.ndarray,
        safety_flags: Dict[str, bool],
        reward: float = 0.0,
        done: bool = False,
        info: Optional[Dict[str, Any]] = None
    ):
        """
        Record a single timestep.
        
        This is the core of Event Sourcing - every decision is stored.
        
        Args:
            timestep: Timestep number
            state: Observation state (6D for lane-keeping)
            policy_action: Raw policy output
            safety_action: Action after safety shield
            safety_flags: Dictionary of safety interventions
            reward: Reward received
            done: Episode termination flag
            info: Additional information
        """
        if not self.recording:
            raise RuntimeError("No episode in progress. Call start_episode() first.")
        
        # Create immutable event
        event = {
            "timestep": int(timestep),
            
            # State snapshot
            "state": {
                "lane_offset": float(state[0]),
                "heading_error": float(state[1]),
                "speed": float(state[2]),
                "left_distance": float(state[3]),
                "right_distance": float(state[4]),
                "curvature": float(state[5])
            },
            
            # Policy output (what the agent wanted to do)
            "policy_action": {
                "steering": float(policy_action[0]),
                "throttle": float(policy_action[1])
            },
            
            # Safety output (what actually happened)
            "safety_action": {
                "steering": float(safety_action[0]),
                "throttle": float(safety_action[1])
            },
            
            # Safety interventions (explicit decisions)
            "safety_flags": {
                "steering_clamped": bool(safety_flags.get("bounds_violated", False)),
                "rate_limited": bool(safety_flags.get("rate_limited", False)),
                "emergency_brake": bool(safety_flags.get("emergency_brake", False)),
                "speed_limited": bool(safety_flags.get("speed_limited", False)),
                "nan_detected": bool(safety_flags.get("nan_detected", False))
            },
            
            # Metadata
            "reward": float(reward),
            "done": bool(done),
            "info": _make_json_serializable(info or {})
        }
        
        self.current_episode.append(event)
    
    def end_episode(self, save: bool = True) -> Dict[str, Any]:
        """
        End current episode and optionally save to disk.
        
        Args:
            save: Whether to save episode file
            
        Returns:
            Complete episode data
        """
        if not self.recording:
            raise RuntimeError("No episode in progress.")
        
        # Finalize metadata
        self.episode_metadata["end_time"] = datetime.now().isoformat()
        self.episode_metadata["total_timesteps"] = len(self.current_episode)
        
        # Calculate episode statistics
        self.episode_metadata["statistics"] = self._calculate_statistics()
        
        # Create complete episode
        episode_data = {
            "metadata": self.episode_metadata,
            "events": self.current_episode
        }
        
        if save:
            filepath = self._save_episode(episode_data)
            if self.verbose:
                print(f"💾 Episode saved: {filepath}")
                print(f"   Timesteps: {self.episode_metadata['total_timesteps']}")
                print(f"   Total reward: {self.episode_metadata['statistics']['total_reward']:.2f}")
                print(f"   Safety interventions: {self.episode_metadata['statistics']['safety_interventions']}")
        
        # Reset state
        self.recording = False
        self.episodes_recorded += 1
        
        return episode_data
    
    def _calculate_statistics(self) -> Dict[str, Any]:
        """Calculate episode-level statistics."""
        if not self.current_episode:
            return {}
        
        total_reward = sum(event["reward"] for event in self.current_episode)
        
        # Count safety interventions
        safety_counts = {
            "steering_clamped": 0,
            "rate_limited": 0,
            "emergency_brake": 0,
            "speed_limited": 0,
            "nan_detected": 0
        }
        
        for event in self.current_episode:
            for flag, value in event["safety_flags"].items():
                if value:
                    safety_counts[flag] += 1
        
        # Action statistics
        policy_actions = np.array([
            [e["policy_action"]["steering"], e["policy_action"]["throttle"]]
            for e in self.current_episode
        ])
        safety_actions = np.array([
            [e["safety_action"]["steering"], e["safety_action"]["throttle"]]
            for e in self.current_episode
        ])
        
        # State statistics
        states = np.array([
            [
                e["state"]["lane_offset"],
                e["state"]["heading_error"],
                e["state"]["speed"]
            ]
            for e in self.current_episode
        ])
        
        return {
            "total_reward": float(total_reward),
            "avg_reward": float(total_reward / len(self.current_episode)),
            "safety_interventions": sum(safety_counts.values()),
            "safety_intervention_rate": sum(safety_counts.values()) / len(self.current_episode),
            "safety_breakdown": safety_counts,
            
            "policy_action_stats": {
                "steering_mean": float(policy_actions[:, 0].mean()),
                "steering_std": float(policy_actions[:, 0].std()),
                "throttle_mean": float(policy_actions[:, 1].mean()),
                "throttle_std": float(policy_actions[:, 1].std())
            },
            
            "safety_action_stats": {
                "steering_mean": float(safety_actions[:, 0].mean()),
                "steering_std": float(safety_actions[:, 0].std()),
                "throttle_mean": float(safety_actions[:, 1].mean()),
                "throttle_std": float(safety_actions[:, 1].std())
            },
            
            "state_stats": {
                "max_lane_deviation": float(np.abs(states[:, 0]).max()),
                "avg_lane_deviation": float(np.abs(states[:, 0]).mean()),
                "max_heading_error": float(np.abs(states[:, 1]).max()),
                "avg_speed": float(states[:, 2].mean())
            }
        }
    
    def _save_episode(self, episode_data: Dict[str, Any]) -> Path:
        """Save episode to disk."""
        episode_id = episode_data["metadata"]["episode_id"]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        filename = f"episode_{episode_id:04d}_{timestamp}.json"
        if self.compress:
            filename += ".gz"
        
        filepath = self.output_dir / filename
        
        # Convert to JSON
        json_str = json.dumps(episode_data, indent=2)
        
        if self.compress:
            with gzip.open(filepath, 'wt', encoding='utf-8') as f:
                f.write(json_str)
        else:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(json_str)
        
        return filepath
    
    @staticmethod
    def load_episode(filepath: str) -> Dict[str, Any]:
        """
        Load episode from disk.
        
        Args:
            filepath: Path to episode file
            
        Returns:
            Complete episode data
        """
        filepath = Path(filepath)
        
        if filepath.suffix == '.gz':
            with gzip.open(filepath, 'rt', encoding='utf-8') as f:
                return json.load(f)
        else:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get recorder summary statistics."""
        return {
            "episodes_recorded": self.episodes_recorded,
            "output_directory": str(self.output_dir),
            "compression_enabled": self.compress
        }


# Convenience function for single-episode recording
def record_episode(
    env,
    policy,
    safety_shield,
    episode_id: int = 0,
    max_steps: int = 1000,
    output_dir: str = "episodes",
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Record a single episode with policy and safety shield.
    
    Args:
        env: Environment instance
        policy: Policy network
        safety_shield: Safety shield instance
        episode_id: Episode identifier
        max_steps: Maximum timesteps
        output_dir: Output directory
        verbose: Print status
        
    Returns:
        Complete episode data
    """
    import torch
    
    recorder = EpisodeRecorder(output_dir=output_dir, verbose=verbose)
    
    # Start recording
    recorder.start_episode(episode_id=episode_id)
    
    # Run episode
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    
    for t in range(max_steps):
        # Get policy action
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            action_mean, value = policy(obs_tensor)
            policy_action = action_mean.cpu().numpy()[0]
        
        # Apply safety shield
        safety_action, safety_info = safety_shield.check_and_fix(policy_action, obs)
        
        # Extract safety flags
        safety_flags = {
            "bounds_violated": "steering_bounds" in safety_info.get("interventions", []) or
                              "throttle_bounds" in safety_info.get("interventions", []),
            "rate_limited": "rate_limited" in safety_info.get("interventions", []),
            "emergency_brake": "emergency_brake" in safety_info.get("interventions", []),
            "speed_limited": "speed_limited" in safety_info.get("interventions", []),
            "nan_detected": "nan_detected" in safety_info.get("interventions", [])
        }
        
        # Step environment
        step_result = env.step(safety_action)
        if len(step_result) == 5:
            next_obs, reward, terminated, truncated, info = step_result
            done = terminated or truncated
        else:
            next_obs, reward, done, info = step_result
        
        # Record timestep
        recorder.record_timestep(
            timestep=t,
            state=obs,
            policy_action=policy_action,
            safety_action=safety_action,
            safety_flags=safety_flags,
            reward=reward,
            done=done,
            info=info
        )
        
        if done:
            break
        
        obs = next_obs
    
    # End and save
    return recorder.end_episode(save=True)
