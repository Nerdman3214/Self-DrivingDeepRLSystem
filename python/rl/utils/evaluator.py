"""
Formal Evaluation Protocol for Self-Driving RL

NO TRAINING - pure evaluation with:
- Frozen policy
- No exploration noise
- Fixed seeds
- Worst-case analysis
"""

import numpy as np
import torch
from typing import Dict, List, Tuple
from rl.envs import LaneKeepingEnv
from rl.networks import MLPActorCritic


class Evaluator:
    """
    Formal evaluation system.
    
    Prevents lying metrics by:
    - Freezing policy (no gradient updates)
    - Disabling exploration (deterministic actions)
    - Using fixed seeds (reproducible)
    - Reporting worst-case scenarios
    """
    
    def __init__(
        self,
        env: LaneKeepingEnv,
        policy: MLPActorCritic,
        n_episodes: int = 10,
        device: str = 'cuda'
    ):
        self.env = env
        self.policy = policy
        self.n_episodes = n_episodes
        self.device = device
    
    @torch.no_grad()
    def evaluate(self, deterministic: bool = True, seeds: List[int] = None) -> Dict[str, float]:
        """
        Run formal evaluation.
        
        Args:
            deterministic: Use mean action (no sampling)
            seeds: Fixed seeds for reproducibility
        
        Returns:
            Dict with:
            - mean_reward
            - std_reward
            - worst_case_deviation
            - max_steering_jerk
            - survival_rate
        """
        self.policy.eval()  # Freeze BatchNorm, Dropout, etc.
        
        if seeds is None:
            seeds = list(range(self.n_episodes))
        
        episode_rewards = []
        episode_lengths = []
        max_deviations = []
        max_steering_jerks = []
        completions = []
        
        # Track detailed metrics
        all_lane_offsets = []
        all_heading_errors = []
        all_speeds = []
        
        for seed in seeds[:self.n_episodes]:
            obs, _ = self.env.reset(seed=seed)
            episode_reward = 0
            episode_length = 0
            done = False
            
            lane_offsets = []
            heading_errors = []
            speeds = []
            steering_history = []
            
            while not done:
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                
                if deterministic:
                    # Use mean action (no exploration)
                    action_mean = self.policy.actor_mean(self.policy.shared_net(obs_tensor))
                    action = torch.tanh(action_mean).cpu().numpy()[0]
                else:
                    action, _, _, _ = self.policy.get_action_and_value(obs_tensor)
                    action = action.cpu().numpy()[0]
                
                obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                episode_reward += reward
                episode_length += 1
                
                # Track metrics
                lane_offsets.append(abs(obs[0]))  # lane_offset
                heading_errors.append(abs(obs[1]))  # heading_error
                speeds.append(obs[2])  # speed
                steering_history.append(action[0])  # steering
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            # Worst-case metrics
            max_deviations.append(max(lane_offsets) if lane_offsets else 0)
            
            # Steering jerk (derivative of steering)
            if len(steering_history) > 1:
                jerks = np.abs(np.diff(steering_history))
                max_steering_jerks.append(max(jerks))
            else:
                max_steering_jerks.append(0)
            
            # Completion (did full episode?)
            completions.append(1 if episode_length >= 1000 else 0)
            
            # Aggregate for statistics
            all_lane_offsets.extend(lane_offsets)
            all_heading_errors.extend(heading_errors)
            all_speeds.extend(speeds)
        
        self.policy.train()  # Restore training mode
        
        # Compile results
        results = {
            # Core performance
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'min_reward': np.min(episode_rewards),
            'max_reward': np.max(episode_rewards),
            
            # Episode statistics
            'mean_length': np.mean(episode_lengths),
            'completion_rate': np.mean(completions) * 100,  # percentage
            
            # Worst-case safety
            'worst_case_deviation': np.max(max_deviations),
            'mean_max_deviation': np.mean(max_deviations),
            
            # Control smoothness
            'max_steering_jerk': np.max(max_steering_jerks),
            'mean_steering_jerk': np.mean(max_steering_jerks),
            
            # Driving quality
            'mean_lane_offset': np.mean(all_lane_offsets),
            'mean_heading_error': np.mean(all_heading_errors),
            'mean_speed': np.mean(all_speeds),
        }
        
        return results
    
    def print_results(self, results: Dict[str, float]):
        """Pretty print evaluation results."""
        print("\n" + "=" * 60)
        print("FORMAL EVALUATION RESULTS")
        print("=" * 60)
        print(f"Episodes: {self.n_episodes}")
        print()
        print("Performance:")
        print(f"  Mean reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
        print(f"  Range: [{results['min_reward']:.2f}, {results['max_reward']:.2f}]")
        print()
        print("Safety:")
        print(f"  Completion rate: {results['completion_rate']:.1f}%")
        print(f"  Worst-case deviation: {results['worst_case_deviation']:.3f}m")
        print(f"  Mean max deviation: {results['mean_max_deviation']:.3f}m")
        print()
        print("Control Quality:")
        print(f"  Max steering jerk: {results['max_steering_jerk']:.3f}")
        print(f"  Mean steering jerk: {results['mean_steering_jerk']:.3f}")
        print()
        print("Driving Quality:")
        print(f"  Mean lane offset: {results['mean_lane_offset']:.3f}m")
        print(f"  Mean heading error: {results['mean_heading_error']:.3f}rad")
        print(f"  Mean speed: {results['mean_speed']:.2f}m/s")
        print("=" * 60)
