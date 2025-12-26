"""
Phase 3 Demo: Multi-Agent Traffic System

Train ego agent in traffic-aware environment.

Usage:
    # Test environment
    python phase3_demo.py --mode test
    
    # Train ego agent
    python phase3_demo.py --mode train --episodes 100
    
    # Record traffic episodes
    python phase3_demo.py --mode record --episodes 10
    
    # Evaluate with traffic metrics
    python phase3_demo.py --mode evaluate --checkpoint checkpoints/traffic_policy.pt
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from rl.envs.multi_agent_env import MultiAgentLaneKeepingEnv
from rl.safety.traffic_safety import TrafficSafetyShield, TrafficSafetyLimits
from rl.networks.mlp_policy import MLPActorCritic
from rl.utils.episode_recorder import EpisodeRecorder


def test_environment(args):
    """Test multi-agent environment."""
    print("\n" + "="*70)
    print("🚗 PHASE 3: MULTI-AGENT TRAFFIC SYSTEM TEST")
    print("="*70)
    
    # Create environment
    env = MultiAgentLaneKeepingEnv(
        scenario_type=args.scenario,
        max_episode_steps=200
    )
    
    # Create safety shield
    safety_limits = TrafficSafetyLimits()
    safety_shield = TrafficSafetyShield(limits=safety_limits)
    
    print(f"\n📊 Environment: {args.scenario}")
    print(f"Observation space: {env.observation_space.shape[0]}D")
    print(f"  [lane_offset, heading_error, speed, lead_distance,")
    print(f"   lead_relative_speed, left_lane_free, right_lane_free, ttc]")
    print(f"Action space: {env.action_space.shape[0]}D [steering, throttle]")
    print(f"\nSafety Shield:")
    print(f"  TTC Emergency: {safety_limits.ttc_emergency}s")
    print(f"  TTC Warning: {safety_limits.ttc_warning}s")
    print(f"  Min Safe Gap: {safety_limits.min_safe_gap}m")
    
    # Run test episode
    print(f"\n🎮 Running test episode...")
    obs, _ = env.reset()
    
    total_reward = 0.0
    collisions = 0
    near_misses = 0
    safety_interventions = 0
    
    for step in range(200):
        # Random policy for testing
        action = env.action_space.sample()
        
        # Apply safety shield
        safe_action, safety_info = safety_shield.check_and_fix(
            action,
            obs,
            {'step': step}
        )
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(safe_action)
        
        total_reward += reward
        
        if info.get('collision', False):
            collisions += 1
        if info.get('near_miss', False):
            near_misses += 1
        if len(safety_info['interventions']) > 0:
            safety_interventions += 1
        
        # Print key events
        if step % 20 == 0 or len(safety_info['interventions']) > 0:
            ttc = obs[7]
            lead_dist = obs[3]
            print(f"  Step {step:3d}: TTC={ttc:5.1f}s, Lead={lead_dist:5.1f}m, "
                  f"Speed={info['ego_speed']:4.1f}m/s, "
                  f"Safety={safety_info['interventions']}")
        
        if terminated or truncated:
            print(f"\n  Episode ended at step {step}")
            break
    
    # Print statistics
    stats = env.get_episode_stats()
    shield_stats = safety_shield.get_intervention_stats()
    
    print("\n" + "="*70)
    print("📊 EPISODE STATISTICS")
    print("="*70)
    print(f"Total Reward: {total_reward:.2f}")
    print(f"Episode Length: {stats['episode_length']}")
    print(f"\n🛡️  SAFETY METRICS")
    print(f"Collisions: {stats['collisions']}")
    print(f"Near Misses (TTC < 2s): {stats['near_misses']}")
    print(f"Average TTC: {stats['avg_ttc']:.2f}s")
    print(f"Minimum TTC: {stats['min_ttc']:.2f}s")
    print(f"Safety Interventions: {safety_interventions} timesteps")
    print(f"\n🔧 INTERVENTION BREAKDOWN")
    for intervention_type, count in shield_stats.items():
        if count > 0:
            print(f"  {intervention_type}: {count}")
    print("="*70)


def record_traffic_episodes(args):
    """Record episodes in traffic environment."""
    print("\n" + "="*70)
    print("📼 PHASE 3: RECORDING TRAFFIC EPISODES")
    print("="*70)
    
    # Create environment
    env = MultiAgentLaneKeepingEnv(
        scenario_type=args.scenario,
        max_episode_steps=500
    )
    
    # Create safety shield
    safety_shield = TrafficSafetyShield()
    
    # Create policy (random for now)
    policy = MLPActorCritic(
        observation_dim=8,  # Traffic-aware observations
        action_dim=2,
        hidden_dims=(256, 256)
    )
    policy.eval()
    
    # Create recorder
    recorder = EpisodeRecorder(output_dir="traffic_episodes")
    
    print(f"\n🎥 Recording {args.episodes} traffic episodes...")
    print(f"Scenario: {args.scenario}")
    
    for ep in range(args.episodes):
        obs, _ = env.reset()
        safety_shield.reset()
        
        episode_id = recorder.start_episode()
        
        print(f"\n📼 Recording Episode {ep}...")
        
        for step in range(500):
            # Policy inference (deterministic for replay)
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                action, _, _, _ = policy.get_action_and_value(
                    obs_tensor, deterministic=True
                )
                policy_action = action.cpu().numpy().flatten()
            
            # Safety shield
            safe_action, safety_info = safety_shield.check_and_fix(
                policy_action,
                obs,
                {'step': step}
            )
            
            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(safe_action)
            
            # Record timestep
            # Create 6D state for compatibility with EpisodeRecorder
            # (lane_offset, heading_error, speed, left_dist, right_dist, curvature)
            compatible_state = np.array([
                obs[0],  # lane_offset
                obs[1],  # heading_error
                obs[2],  # speed
                obs[3],  # lead_distance (use as left_distance for now)
                10.0,    # right_distance (placeholder)
                0.0      # curvature (placeholder)
            ], dtype=np.float32)
            
            recorder.record_timestep(
                timestep=step,
                state=compatible_state,  # 6D for compatibility
                policy_action=policy_action,
                safety_action=safe_action,
                safety_flags={
                    'ttc_emergency': 'ttc_emergency' in safety_info['interventions'],
                    'ttc_warning': 'ttc_warning' in safety_info['interventions'],
                    'unsafe_gap': 'unsafe_gap' in safety_info['interventions'],
                    'collision': bool(info.get('collision', False))
                },
                reward=reward,
                done=terminated or truncated,
                info={
                    'collision': bool(info.get('collision', False)),
                    'ttc': float(obs[7]),
                    'lead_distance': float(obs[3]),
                    'lead_relative_speed': float(obs[4]),
                    'near_miss': bool(info.get('near_miss', False)),
                    'ego_speed': float(info.get('ego_speed', 0.0)),
                    'ego_position': float(info.get('ego_position', 0.0))
                }
            )
            
            obs = next_obs
            
            if terminated or truncated:
                break
        
        # End episode
        episode_data = recorder.end_episode()
        stats = env.get_episode_stats()
        
        # The recorder already prints save message, just print stats
        print(f"   Collisions: {stats['collisions']}")
        print(f"   Near Misses: {stats['near_misses']}")
        print(f"   Avg TTC: {stats['avg_ttc']:.2f}s")
    
    print("\n✅ Recording complete!")
    print(f"Episodes saved to: traffic_episodes/")


def evaluate_traffic_policy(args):
    """Evaluate policy with traffic metrics."""
    print("\n" + "="*70)
    print("📊 PHASE 3: TRAFFIC-AWARE EVALUATION")
    print("="*70)
    
    # Create environment
    env = MultiAgentLaneKeepingEnv(
        scenario_type=args.scenario,
        max_episode_steps=500
    )
    
    # Load policy
    policy = MLPActorCritic(
        observation_dim=8,
        action_dim=2,
        hidden_dims=(256, 256)
    )
    
    if args.checkpoint:
        print(f"📦 Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        policy.load_state_dict(checkpoint['policy_state_dict'])
    else:
        print("🎲 Using random policy")
    
    policy.eval()
    
    # Create safety shield
    safety_shield = TrafficSafetyShield()
    
    # Run evaluation episodes
    num_episodes = args.episodes
    all_stats = []
    
    print(f"\n🎯 Evaluating {num_episodes} episodes...")
    
    for ep in range(num_episodes):
        obs, _ = env.reset()
        safety_shield.reset()
        
        for step in range(500):
            # Policy inference
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                action, _, _, _ = policy.get_action_and_value(
                    obs_tensor, deterministic=True
                )
                policy_action = action.cpu().numpy().flatten()
            
            # Safety shield
            safe_action, _ = safety_shield.check_and_fix(
                policy_action,
                obs,
                {'step': step}
            )
            
            # Step
            obs, _, terminated, truncated, _ = env.step(safe_action)
            
            if terminated or truncated:
                break
        
        stats = env.get_episode_stats()
        all_stats.append(stats)
        
        print(f"  Episode {ep}: Reward={stats['total_reward']:6.2f}, "
              f"Collisions={stats['collisions']}, TTC={stats['avg_ttc']:.2f}s")
    
    # Aggregate metrics
    print("\n" + "="*70)
    print("📊 AGGREGATED METRICS")
    print("="*70)
    
    total_collisions = sum(s['collisions'] for s in all_stats)
    total_near_misses = sum(s['near_misses'] for s in all_stats)
    avg_ttc = np.mean([s['avg_ttc'] for s in all_stats])
    min_ttc = min(s['min_ttc'] for s in all_stats)
    avg_reward = np.mean([s['total_reward'] for s in all_stats])
    
    shield_stats = safety_shield.get_intervention_stats()
    total_interventions = safety_shield.get_total_interventions()
    
    print(f"\n🏆 PERFORMANCE")
    print(f"Average Reward: {avg_reward:.2f}")
    
    print(f"\n🛡️  SAFETY")
    print(f"Collision Rate: {total_collisions / num_episodes:.2%}")
    print(f"Near-Miss Rate: {total_near_misses / num_episodes:.2f} per episode")
    print(f"Average TTC: {avg_ttc:.2f}s")
    print(f"Minimum TTC: {min_ttc:.2f}s")
    print(f"Safety Overrides: {total_interventions} total")
    
    print(f"\n🔧 INTERVENTION BREAKDOWN")
    for intervention_type, count in shield_stats.items():
        if count > 0:
            print(f"  {intervention_type}: {count}")
    
    # Pass/Fail
    print("\n" + "="*70)
    collision_free = total_collisions == 0
    safe_ttc = avg_ttc > 3.0
    
    if collision_free and safe_ttc:
        print("✅ TRAFFIC-AWARE SYSTEM: PASS")
        print("   - No collisions ✓")
        print("   - Safe TTC ✓")
        print("\n🎯 Your agent can coexist with traffic safely.")
    else:
        print("❌ TRAFFIC-AWARE SYSTEM: NEEDS IMPROVEMENT")
        if not collision_free:
            print(f"   - Collisions: {total_collisions}")
        if not safe_ttc:
            print(f"   - Average TTC too low: {avg_ttc:.2f}s (need > 3.0s)")
    
    print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description="Phase 3: Multi-Agent Traffic System"
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['test', 'record', 'evaluate', 'train'],
        default='test',
        help='Operation mode'
    )
    parser.add_argument(
        '--scenario',
        type=str,
        choices=['highway', 'dense', 'stop_and_go'],
        default='highway',
        help='Traffic scenario'
    )
    parser.add_argument(
        '--episodes',
        type=int,
        default=5,
        help='Number of episodes'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Policy checkpoint path'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'test':
        test_environment(args)
    elif args.mode == 'record':
        record_traffic_episodes(args)
    elif args.mode == 'evaluate':
        evaluate_traffic_policy(args)
    elif args.mode == 'train':
        print("⚠️  Training mode: Use train_traffic_agent.py for full PPO training")
    
    print("\n✨ Phase 3 demo complete!")


if __name__ == "__main__":
    main()
