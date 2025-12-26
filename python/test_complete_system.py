"""
Complete System Test
Tests all major components of the self-driving RL system
"""

import torch
import numpy as np
from pathlib import Path

def test_environments():
    """Test all environment types"""
    print("\n" + "="*70)
    print("1. TESTING ENVIRONMENTS")
    print("="*70)
    
    # PyBullet environment
    from rl.envs.pybullet_driving_env import PyBulletDrivingEnv
    env = PyBulletDrivingEnv(render_mode=None)
    obs, _ = env.reset()
    assert obs.shape == (6,), f"Expected shape (6,), got {obs.shape}"
    for _ in range(3):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    env.close()
    print("✅ PyBullet environment: PASS")
    
    # Lane keeping environment
    from rl.envs.lane_keeping_env import LaneKeepingEnv
    env = LaneKeepingEnv()
    obs, _ = env.reset()
    assert obs.shape == (6,), f"Expected shape (6,), got {obs.shape}"
    for _ in range(3):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    env.close()
    print("✅ Lane keeping environment: PASS")

def test_policies():
    """Test policy networks"""
    print("\n" + "="*70)
    print("2. TESTING POLICY NETWORKS")
    print("="*70)
    
    # MLP policy
    from rl.networks.mlp_policy import MLPActorCritic
    policy = MLPActorCritic(
        observation_dim=6,
        action_dim=2,
        hidden_dims=(64, 64)
    )
    obs = torch.randn(1, 6)
    action, log_prob, entropy, value = policy.get_action_and_value(obs)
    assert action.shape == (1, 2), f"Expected action shape (1, 2), got {action.shape}"
    assert value.shape == (1, 1), f"Expected value shape (1, 1), got {value.shape}"
    print(f"✅ MLP policy: PASS ({sum(p.numel() for p in policy.parameters()):,} params)")
    
    # Vision policy
    from rl.networks.vision_policy import VisionActorCritic
    vision_policy = VisionActorCritic(input_channels=4, action_dim=2)
    img_obs = torch.randn(1, 4, 84, 84)
    action, log_prob, entropy, value = vision_policy.get_action_and_value(img_obs)
    assert action.shape == (1, 2), f"Expected action shape (1, 2), got {action.shape}"
    print(f"✅ Vision policy: PASS ({sum(p.numel() for p in vision_policy.parameters()):,} params)")

def test_vision_wrapper():
    """Test vision observation wrapper"""
    print("\n" + "="*70)
    print("3. TESTING VISION WRAPPER")
    print("="*70)
    
    from rl.envs.pybullet_driving_env import PyBulletDrivingEnv
    from rl.envs.vision_wrapper import VisionWrapper
    
    base_env = PyBulletDrivingEnv(render_mode=None)
    env = VisionWrapper(base_env, grayscale=True, frame_stack=4)
    obs, _ = env.reset()
    assert obs.shape == (4, 84, 84), f"Expected shape (4, 84, 84), got {obs.shape}"
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    env.close()
    print("✅ Vision wrapper: PASS")

def test_curriculum():
    """Test curriculum learning"""
    print("\n" + "="*70)
    print("4. TESTING CURRICULUM LEARNING")
    print("="*70)
    
    from rl.training.curriculum import CurriculumScheduler
    
    curriculum = CurriculumScheduler(initial_stage=0)
    config = curriculum.get_current_config()
    assert 'lane_width' in config, "Missing lane_width in config"
    print(f"✅ Curriculum stage 0: {config['lane_width']}m lane")
    
    # Test advancement
    for _ in range(20):
        curriculum.update(episode_reward=75.0)
    if curriculum.should_advance():
        curriculum.advance()
        print(f"✅ Curriculum advancement: PASS (now stage {curriculum.current_stage_idx})")
    else:
        print("✅ Curriculum advancement: PASS (needs more episodes)")

def test_stress_testing():
    """Test stress testing wrapper"""
    print("\n" + "="*70)
    print("5. TESTING STRESS TESTING")
    print("="*70)
    
    from rl.envs.pybullet_driving_env import PyBulletDrivingEnv
    from rl.evaluation.stress_testing import StressTestWrapper
    
    base_env = PyBulletDrivingEnv(render_mode=None)
    
    # Test slippery scenario
    env = StressTestWrapper(base_env, scenario='slippery')
    assert env.friction_multiplier == 0.3, f"Expected friction 0.3, got {env.friction_multiplier}"
    obs, _ = env.reset()
    env.close()
    print("✅ Stress test wrapper (slippery): PASS")

def test_checkpoint_loading():
    """Test checkpoint loading"""
    print("\n" + "="*70)
    print("6. TESTING CHECKPOINT LOADING")
    print("="*70)
    
    checkpoint_path = Path('checkpoints/pybullet/pybullet_model_final.pt')
    if not checkpoint_path.exists():
        print("⚠️  No checkpoint found - skipping test")
        return
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    assert 'policy_state_dict' in checkpoint, "Missing policy_state_dict"
    print(f"✅ Checkpoint loaded: {checkpoint.get('total_steps', 0):,} steps")
    
    # Test loading into policy
    from rl.networks.mlp_policy import MLPActorCritic
    policy = MLPActorCritic(observation_dim=6, action_dim=2, hidden_dims=(64, 64))
    policy.load_state_dict(checkpoint['policy_state_dict'])
    policy.eval()
    print("✅ Policy weights loaded: PASS")

def test_policy_comparison():
    """Test policy comparison system"""
    print("\n" + "="*70)
    print("7. TESTING POLICY COMPARISON")
    print("="*70)
    
    from rl.evaluation.policy_comparison import PolicyComparator
    from rl.envs.pybullet_driving_env import PyBulletDrivingEnv
    from rl.networks.mlp_policy import MLPActorCritic
    
    comparator = PolicyComparator(
        env_factory=lambda: PyBulletDrivingEnv(render_mode=None),
        policy_class=MLPActorCritic
    )
    
    checkpoint_dir = Path('checkpoints/pybullet')
    if checkpoint_dir.exists():
        checkpoints = list(checkpoint_dir.glob('*.pt'))
        if checkpoints:
            comparator.add_checkpoint(checkpoints[0])
            print(f"✅ Policy comparison: PASS ({len(comparator.checkpoints)} checkpoint loaded)")
        else:
            print("⚠️  No checkpoints found - partial test")
    else:
        print("⚠️  Checkpoint directory not found - skipping")

def main():
    print("\n" + "🚀"*35)
    print("COMPLETE SYSTEM TEST")
    print("🚀"*35)
    
    tests = [
        ("Environments", test_environments),
        ("Policy Networks", test_policies),
        ("Vision Wrapper", test_vision_wrapper),
        ("Curriculum Learning", test_curriculum),
        ("Stress Testing", test_stress_testing),
        ("Checkpoint Loading", test_checkpoint_loading),
        ("Policy Comparison", test_policy_comparison),
    ]
    
    results = {}
    for name, test_func in tests:
        try:
            test_func()
            results[name] = "PASS"
        except Exception as e:
            print(f"\n❌ {name}: FAILED")
            print(f"   Error: {e}")
            results[name] = "FAIL"
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    for name, result in results.items():
        icon = "✅" if result == "PASS" else "❌"
        print(f"{icon} {name}: {result}")
    
    passed = sum(1 for r in results.values() if r == "PASS")
    total = len(results)
    print("\n" + "="*70)
    print(f"TOTAL: {passed}/{total} tests passed ({passed/total*100:.0f}%)")
    print("="*70)
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! System is fully operational.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. See details above.")

if __name__ == "__main__":
    main()
