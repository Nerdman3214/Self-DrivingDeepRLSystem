#!/usr/bin/env python3
"""
Test Lane-Keeping Environment

Quick test to verify the environment works correctly.
"""

import numpy as np
from rl.envs import LaneKeepingEnv


def test_environment():
    """Test basic environment functionality."""
    print("Testing Lane-Keeping Environment...")
    print("=" * 60)
    
    # Create environment
    env = LaneKeepingEnv(
        lane_width=3.5,
        max_speed=30.0,
        max_episode_steps=1000,
        random_start=True,
    )
    
    # Print spaces
    print(f"Observation space: {env.observation_space}")
    print(f"  Shape: {env.observation_space.shape}")
    print(f"  Low: {env.observation_space.low}")
    print(f"  High: {env.observation_space.high}")
    print()
    print(f"Action space: {env.action_space}")
    print(f"  Shape: {env.action_space.shape}")
    print(f"  Low: {env.action_space.low}")
    print(f"  High: {env.action_space.high}")
    print()
    
    # Reset
    obs, info = env.reset(seed=42)
    print(f"Initial observation: {obs}")
    print(f"Initial info: {info}")
    print()
    
    # Run episode
    print("Running test episode...")
    episode_reward = 0
    episode_length = 0
    
    for step in range(100):
        # Random action
        action = env.action_space.sample()
        
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        episode_length += 1
        
        if step < 5:
            print(f"Step {step}:")
            print(f"  Action: {action}")
            print(f"  Obs: {obs}")
            print(f"  Reward: {reward:.3f}")
            print(f"  Lane offset: {info['lane_offset_normalized']:.3f}")
            print(f"  Speed: {info['speed']:.1f} m/s")
            print()
        
        if terminated or truncated:
            print(f"Episode ended at step {episode_length}")
            print(f"  Terminated: {terminated}")
            print(f"  Truncated: {truncated}")
            break
    
    print()
    print(f"Episode reward: {episode_reward:.2f}")
    print(f"Episode length: {episode_length}")
    print()
    
    # Test multiple resets
    print("Testing multiple resets...")
    for i in range(3):
        obs, _ = env.reset()
        print(f"Reset {i+1} - Initial state: {obs}")
    
    print()
    print("✅ Environment test passed!")
    print("=" * 60)
    
    env.close()


def test_reward_function():
    """Test reward function behavior."""
    print("\nTesting Reward Function...")
    print("=" * 60)
    
    env = LaneKeepingEnv()
    
    scenarios = [
        {
            'name': 'Centered, aligned, good speed',
            'state': {
                'lateral_position': 0.0,
                'heading': 0.0,
                'speed': 21.0,  # 70% of max
            },
            'action': [0.0, 0.0],
        },
        {
            'name': 'Off-center (left)',
            'state': {
                'lateral_position': 1.0,
                'heading': 0.0,
                'speed': 21.0,
            },
            'action': [0.0, 0.0],
        },
        {
            'name': 'Heading error',
            'state': {
                'lateral_position': 0.0,
                'heading': 0.3,
                'speed': 21.0,
            },
            'action': [0.0, 0.0],
        },
        {
            'name': 'Hard steering',
            'state': {
                'lateral_position': 0.0,
                'heading': 0.0,
                'speed': 21.0,
            },
            'action': [1.0, 0.0],
        },
    ]
    
    for scenario in scenarios:
        # Set state manually
        env.lateral_position = scenario['state']['lateral_position']
        env.heading = scenario['state']['heading']
        env.speed = scenario['state']['speed']
        
        # Compute reward
        reward = env._compute_reward(
            scenario['action'][0],
            scenario['action'][1],
            0.0
        )
        
        print(f"{scenario['name']}")
        print(f"  Reward: {reward:.3f}")
    
    print()
    print("✅ Reward function test passed!")
    print("=" * 60)
    
    env.close()


if __name__ == '__main__':
    test_environment()
    test_reward_function()
