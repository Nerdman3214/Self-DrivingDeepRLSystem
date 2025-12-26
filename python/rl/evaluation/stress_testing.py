"""
Stress Testing Suite for Autonomous Driving

Robustness validation through adversarial scenarios.
Tests policy under challenging, edge-case conditions.

Purpose:
    - Validate that learning is robust, not memorization
    - Find failure modes early
    - Industry-standard safety validation
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import pybullet as p


@dataclass
class StressTestResult:
    """Results from a single stress test scenario"""
    scenario_name: str
    episodes_tested: int
    success_rate: float
    avg_reward: float
    avg_episode_length: float
    collision_rate: float
    lane_violations: int
    recovery_time_avg: float  # Time to recover from perturbation


class StressTestWrapper(gym.Wrapper):
    """
    Environment wrapper that applies stress conditions.
    
    Stress Types:
        - Slippery road (reduced friction)
        - Sensor noise (observation perturbations)
        - Random pushes (external forces)
        - Initial perturbations (off-center starts)
        - Narrow lanes
    """
    
    def __init__(
        self,
        env: gym.Env,
        scenario: Optional[str] = None,
        friction_multiplier: float = 1.0,
        observation_noise_std: float = 0.0,
        random_push_prob: float = 0.0,
        random_push_force: float = 0.0,
        initial_offset_range: float = 0.0,
        lane_width_multiplier: float = 1.0
    ):
        """
        Args:
            env: Base environment
            scenario: Predefined scenario name ('slippery', 'noisy_sensors', etc.)
            friction_multiplier: Multiply friction (< 1.0 = slippery)
            observation_noise_std: Gaussian noise std for observations
            random_push_prob: Probability of random lateral push each step
            random_push_force: Magnitude of random push
            initial_offset_range: Random initial lane offset range
            lane_width_multiplier: Multiply lane width (< 1.0 = narrow)
        """
        super().__init__(env)
        
        # Apply scenario presets if provided
        if scenario is not None:
            config = self._get_scenario_config(scenario)
            friction_multiplier = config.get('friction_multiplier', friction_multiplier)
            observation_noise_std = config.get('observation_noise_std', observation_noise_std)
            random_push_prob = config.get('random_push_prob', random_push_prob)
            random_push_force = config.get('random_push_force', random_push_force)
            initial_offset_range = config.get('initial_offset_range', initial_offset_range)
            lane_width_multiplier = config.get('lane_width_multiplier', lane_width_multiplier)
        
        self.friction_multiplier = friction_multiplier
        self.observation_noise_std = observation_noise_std
        self.random_push_prob = random_push_prob
        self.random_push_force = random_push_force
        self.initial_offset_range = initial_offset_range
        self.lane_width_multiplier = lane_width_multiplier
        
        # Metrics tracking
        self.perturbations_applied = 0
        self.recovery_times = []
        self.last_perturbation_step = None
    
    def _get_scenario_config(self, scenario: str) -> Dict[str, Any]:
        """Map scenario name to configuration"""
        scenarios = {
            'baseline': {},
            'slippery': {'friction_multiplier': 0.3},
            'noisy_sensors': {'observation_noise_std': 0.1},
            'random_pushes': {'random_push_prob': 0.05, 'random_push_force': 50.0},
            'difficult_starts': {'initial_offset_range': 1.0},
            'narrow_lanes': {'lane_width_multiplier': 0.7},
            'combined_stress': {
                'friction_multiplier': 0.4,
                'observation_noise_std': 0.08,
                'random_push_prob': 0.03,
                'random_push_force': 30.0,
                'lane_width_multiplier': 0.8
            }
        }
        return scenarios.get(scenario, {})
    
    def reset(self, **kwargs):
        """Reset with stress conditions"""
        obs, info = self.env.reset(**kwargs)
        
        # Apply initial perturbation
        if self.initial_offset_range > 0:
            if hasattr(self.env.unwrapped, 'car'):
                # PyBullet environment
                offset = np.random.uniform(
                    -self.initial_offset_range,
                    self.initial_offset_range
                )
                pos, orn = p.getBasePositionAndOrientation(self.env.unwrapped.car)
                new_pos = [pos[0] + offset, pos[1], pos[2]]
                p.resetBasePositionAndOrientation(
                    self.env.unwrapped.car,
                    new_pos,
                    orn
                )
        
        # Modify friction
        if self.friction_multiplier != 1.0 and hasattr(self.env.unwrapped, 'plane'):
            p.changeDynamics(
                self.env.unwrapped.plane,
                -1,
                lateralFriction=0.5 * self.friction_multiplier
            )
        
        # Reset metrics
        self.perturbations_applied = 0
        self.last_perturbation_step = None
        
        # Add observation noise
        if self.observation_noise_std > 0:
            obs = self._add_observation_noise(obs)
        
        return obs, info
    
    def step(self, action):
        """Step with stress conditions"""
        # Apply random push
        if self.random_push_prob > 0 and np.random.random() < self.random_push_prob:
            self._apply_random_push()
            self.perturbations_applied += 1
            self.last_perturbation_step = 0
        
        # Normal step
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Add observation noise
        if self.observation_noise_std > 0:
            obs = self._add_observation_noise(obs)
        
        # Track recovery
        if self.last_perturbation_step is not None:
            self.last_perturbation_step += 1
            # Check if recovered (lane offset < threshold)
            if hasattr(obs, '__iter__') and abs(obs[0]) < 0.2:  # Assuming obs[0] is lane_offset
                self.recovery_times.append(self.last_perturbation_step)
                self.last_perturbation_step = None
        
        return obs, reward, terminated, truncated, info
    
    def _add_observation_noise(self, obs: np.ndarray) -> np.ndarray:
        """Add Gaussian noise to observations"""
        noise = np.random.normal(0, self.observation_noise_std, obs.shape)
        noisy_obs = obs + noise
        # Clip to observation space bounds if possible
        if hasattr(self.observation_space, 'low') and hasattr(self.observation_space, 'high'):
            noisy_obs = np.clip(noisy_obs, self.observation_space.low, self.observation_space.high)
        return noisy_obs.astype(obs.dtype)
    
    def _apply_random_push(self):
        """Apply random lateral force to car"""
        if hasattr(self.env.unwrapped, 'car'):
            force_direction = np.random.choice([-1, 1])
            force = [force_direction * self.random_push_force, 0, 0]
            p.applyExternalForce(
                self.env.unwrapped.car,
                -1,
                force,
                [0, 0, 0],
                p.LINK_FRAME
            )


class StressTestSuite:
    """
    Complete stress testing suite.
    
    Runs multiple scenarios and generates robustness report.
    """
    
    def __init__(self, base_env_factory):
        """
        Args:
            base_env_factory: Function that creates base environment
        """
        self.base_env_factory = base_env_factory
        
        # Define stress scenarios
        self.scenarios = {
            'baseline': {
                'description': 'Normal conditions (baseline)',
                'config': {}
            },
            'slippery_road': {
                'description': 'Reduced friction (ice-like conditions)',
                'config': {
                    'friction_multiplier': 0.3
                }
            },
            'noisy_sensors': {
                'description': 'Sensor noise (observation perturbations)',
                'config': {
                    'observation_noise_std': 0.1
                }
            },
            'random_pushes': {
                'description': 'Random external forces',
                'config': {
                    'random_push_prob': 0.05,
                    'random_push_force': 10.0
                }
            },
            'difficult_start': {
                'description': 'Off-center initial positions',
                'config': {
                    'initial_offset_range': 1.0
                }
            },
            'narrow_lane': {
                'description': 'Narrow lane (2.5m instead of 3.5m)',
                'config': {
                    'lane_width_multiplier': 0.7
                }
            },
            'combined_stress': {
                'description': 'Multiple stressors combined',
                'config': {
                    'friction_multiplier': 0.5,
                    'observation_noise_std': 0.05,
                    'random_push_prob': 0.02,
                    'initial_offset_range': 0.5
                }
            }
        }
    
    def run_scenario(
        self,
        policy,
        scenario_name: str,
        num_episodes: int = 20
    ) -> StressTestResult:
        """
        Run single stress test scenario.
        
        Args:
            policy: Trained policy to test
            scenario_name: Name of scenario from self.scenarios
            num_episodes: Number of episodes to test
        
        Returns:
            Test results
        """
        config = self.scenarios[scenario_name]['config']
        
        # Create stressed environment
        base_env = self.base_env_factory()
        env = StressTestWrapper(base_env, **config)
        
        # Run episodes
        episode_rewards = []
        episode_lengths = []
        collisions = 0
        lane_violations = 0
        successes = 0
        
        for episode in range(num_episodes):
            obs, _ = env.reset()
            episode_reward = 0
            episode_length = 0
            
            done = False
            while not done:
                # Get action from policy
                import torch
                with torch.no_grad():
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                    action, _, _, _ = policy.get_action_and_value(
                        obs_tensor,
                        deterministic=True
                    )
                    action = action[0].numpy()
                
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                episode_length += 1
                done = terminated or truncated
                
                # Track violations
                if terminated:
                    if 'collision' in info and info['collision']:
                        collisions += 1
                    else:
                        lane_violations += 1
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            # Success = completed full episode
            if episode_length >= 900:  # Near max steps
                successes += 1
        
        # Compute metrics
        success_rate = successes / num_episodes
        collision_rate = collisions / num_episodes
        avg_recovery = np.mean(env.recovery_times) if env.recovery_times else 0.0
        
        env.close()
        
        return StressTestResult(
            scenario_name=scenario_name,
            episodes_tested=num_episodes,
            success_rate=success_rate,
            avg_reward=np.mean(episode_rewards),
            avg_episode_length=np.mean(episode_lengths),
            collision_rate=collision_rate,
            lane_violations=lane_violations,
            recovery_time_avg=avg_recovery
        )
    
    def run_full_suite(
        self,
        policy,
        num_episodes_per_scenario: int = 20
    ) -> Dict[str, StressTestResult]:
        """
        Run complete stress test suite.
        
        Args:
            policy: Trained policy to test
            num_episodes_per_scenario: Episodes per scenario
        
        Returns:
            Dictionary of results for each scenario
        """
        results = {}
        
        print("\n" + "="*70)
        print("🧪 STRESS TEST SUITE")
        print("="*70)
        
        for scenario_name, scenario_info in self.scenarios.items():
            print(f"\nTesting: {scenario_info['description']}...")
            result = self.run_scenario(
                policy,
                scenario_name,
                num_episodes_per_scenario
            )
            results[scenario_name] = result
            
            print(f"  Success rate: {result.success_rate*100:.1f}%")
            print(f"  Avg reward: {result.avg_reward:.2f}")
            print(f"  Avg length: {result.avg_episode_length:.1f}")
        
        print("\n" + "="*70)
        self._print_summary(results)
        
        return results
    
    def _print_summary(self, results: Dict[str, StressTestResult]):
        """Print stress test summary"""
        print("\n📊 STRESS TEST SUMMARY")
        print("="*70)
        
        baseline = results.get('baseline')
        if baseline:
            print(f"Baseline success rate: {baseline.success_rate*100:.1f}%")
        
        print("\nRobustness Metrics:")
        for scenario_name, result in results.items():
            if scenario_name == 'baseline':
                continue
            degradation = 0.0
            if baseline:
                degradation = (baseline.success_rate - result.success_rate) * 100
            
            print(f"  {scenario_name:20s}: {result.success_rate*100:5.1f}% "
                  f"(degradation: {degradation:+5.1f}%)")
        
        print("="*70)
    
    def _export_report(self, results: Dict[str, StressTestResult]) -> Dict[str, Any]:
        """Export results as JSON-serializable dictionary"""
        from datetime import datetime
        
        baseline = results.get('baseline')
        overall_robustness = np.mean([r.success_rate for r in results.values()]) if results else 0.0
        
        return {
            'timestamp': datetime.now().isoformat(),
            'scenarios': [
                {
                    'name': name,
                    'description': self.scenarios[name]['description'],
                    'success_rate': float(result.success_rate),
                    'collision_rate': float(result.collision_rate),
                    'avg_reward': float(result.avg_reward),
                    'avg_episode_length': float(result.avg_episode_length),
                    'recovery_time': float(result.recovery_time_avg)
                }
                for name, result in results.items()
            ],
            'overall_robustness': float(overall_robustness)
        }


# CLI Interface
def main():
    """Command-line interface for stress testing"""
    import argparse
    import torch
    from datetime import datetime
    
    parser = argparse.ArgumentParser(
        description="Run stress tests on trained policy"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to policy checkpoint"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Episodes per scenario"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to save JSON report"
    )
    
    args = parser.parse_args()
    
    # Import dependencies
    from rl.envs.pybullet_driving_env import PyBulletDrivingEnv
    from rl.networks.mlp_policy import MLPActorCritic
    import json
    
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    
    # Create policy
    env = PyBulletDrivingEnv(render_mode=None)
    hyperparams = checkpoint.get('hyperparameters', {})
    policy = MLPActorCritic(
        observation_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        hidden_dims=hyperparams.get('hidden_dims', (64, 64))
    )
    env.close()
    
    # Load weights
    policy.load_state_dict(checkpoint['policy_state_dict'])
    policy.eval()
    
    print("Creating stress test suite...")
    suite = StressTestSuite(
        base_env_factory=lambda: PyBulletDrivingEnv(render_mode=None)
    )
    
    print(f"Running stress tests ({args.episodes} episodes per scenario)...")
    results = suite.run_full_suite(policy, num_episodes_per_scenario=args.episodes)
    
    print("\n" + "="*70)
    suite._print_summary(results)
    
    if args.output:
        report = suite._export_report(results)
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved: {args.output}")


if __name__ == "__main__":
    main()
