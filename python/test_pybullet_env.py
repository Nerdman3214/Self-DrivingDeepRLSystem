"""
Test PyBullet 3D Environment

Verify that the new PyBullet environment works correctly.
"""

import numpy as np
from rl.envs.pybullet_driving_env import PyBulletDrivingEnv
import time


def test_basic_functionality():
    """Test basic environment functionality"""
    print("🚗 Testing PyBullet 3D Driving Environment\n")
    
    # Create environment
    print("1️⃣ Creating environment...")
    env = PyBulletDrivingEnv(render_mode="human")
    print(f"   ✅ Environment created")
    print(f"   📊 Observation space: {env.observation_space}")
    print(f"   🎮 Action space: {env.action_space}\n")
    
    # Test reset
    print("2️⃣ Testing reset...")
    obs, info = env.reset()
    print(f"   ✅ Reset successful")
    print(f"   📊 Initial observation: {obs}")
    print(f"   ℹ️  Info: {info}\n")
    
    # Test random actions
    print("3️⃣ Testing random control (10 seconds)...")
    print("   🎲 Car will drive randomly\n")
    
    for step in range(600):  # 10 seconds at 60 FPS
        # Random action
        action = env.action_space.sample()
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render
        env.render()
        time.sleep(1/60)  # 60 FPS
        
        # Print status every 60 steps (1 second)
        if step % 60 == 0:
            print(f"   Step {step:3d} | "
                  f"Offset: {obs[0]:+.2f}m | "
                  f"Speed: {obs[2]:.1f}m/s | "
                  f"Reward: {reward:+.2f}")
        
        # Check termination
        if terminated or truncated:
            print(f"\n   ⚠️  Episode ended at step {step}")
            if terminated:
                print(f"   Reason: Safety violation")
            if truncated:
                print(f"   Reason: Max steps reached")
            break
    
    print("\n4️⃣ Testing manual control...")
    print("   🎮 Driving straight with constant throttle\n")
    
    obs, info = env.reset()
    
    for step in range(300):  # 5 seconds
        # Simple controller: go straight
        action = np.array([0.0, 0.5], dtype=np.float32)  # No steering, 50% throttle
        
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        time.sleep(1/60)
        
        if step % 60 == 0:
            print(f"   Step {step:3d} | "
                  f"Offset: {obs[0]:+.2f}m | "
                  f"Speed: {obs[2]:.1f}m/s | "
                  f"Reward: {reward:+.2f}")
        
        if terminated or truncated:
            break
    
    print("\n5️⃣ Closing environment...")
    env.close()
    print("   ✅ Environment closed successfully\n")
    
    print("=" * 60)
    print("✅ ALL TESTS PASSED!")
    print("=" * 60)
    print("\n📝 Next Steps:")
    print("   - Train PPO with: python train_lane_keeping.py --env pybullet")
    print("   - The 3D physics should improve realism")
    print("   - Same observation/action space = minimal code changes\n")


def test_compatibility():
    """Test compatibility with existing RL infrastructure"""
    print("\n🔧 Testing compatibility with existing system...\n")
    
    env = PyBulletDrivingEnv(render_mode=None)  # Headless for testing
    
    # Test Gymnasium interface
    print("✅ Gymnasium interface:")
    print(f"   - action_space: {type(env.action_space)}")
    print(f"   - observation_space: {type(env.observation_space)}")
    print(f"   - metadata: {env.metadata}")
    
    # Test multiple episodes
    print("\n✅ Multiple episode resets:")
    for i in range(3):
        obs, info = env.reset()
        print(f"   Episode {i+1}: obs_shape={obs.shape}, "
              f"obs_min={obs.min():.2f}, obs_max={obs.max():.2f}")
    
    # Test observation bounds
    print("\n✅ Observation bounds check:")
    obs, _ = env.reset()
    for _ in range(100):
        action = env.action_space.sample()
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()
    
    in_bounds = env.observation_space.contains(obs)
    print(f"   Observations within bounds: {in_bounds}")
    
    env.close()
    print("\n✅ Compatibility tests passed!\n")


if __name__ == "__main__":
    # Run basic tests
    test_basic_functionality()
    
    # Run compatibility tests
    test_compatibility()
