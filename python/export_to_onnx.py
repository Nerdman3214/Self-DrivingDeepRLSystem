#!/usr/bin/env python3
"""
STEP 4: Export Trained Policy to ONNX

Inference-only deployment:
- No training code
- No gradients
- No exploration noise
- Deterministic
- Fast
- Safe

Exports only the ACTOR network (mean actions).
"""

import argparse
from pathlib import Path
import torch
import numpy as np

from rl.envs import LaneKeepingEnv
from rl.networks import MLPActorCritic


def export_to_onnx(
    checkpoint_path: str,
    output_path: str,
    opset_version: int = 14,
    test_input: bool = True
):
    """
    Export trained policy to ONNX format.
    
    Args:
        checkpoint_path: Path to trained model checkpoint
        output_path: Output path for ONNX file
        opset_version: ONNX opset version (14+ for better stability)
        test_input: Test exported model with dummy input
    """
    
    print("="*60)
    print("STEP 4: ONNX Export (Inference-Only)")
    print("="*60)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output: {output_path}")
    print()
    
    # Load environment for dimensions
    env = LaneKeepingEnv()
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    print(f"Input: {obs_dim}D state vector")
    print(f"Output: {action_dim}D action vector (steering, throttle)")
    print()
    
    # Load trained policy
    policy = MLPActorCritic(
        observation_dim=obs_dim,
        action_dim=action_dim,
        hidden_dims=[256, 256]
    )
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    policy.load_state_dict(checkpoint['policy_state_dict'])
    policy.eval()  # CRITICAL: Inference mode
    
    print(f"✅ Loaded checkpoint from step {checkpoint.get('global_step', 'unknown')}")
    print()
    
    # Create inference-only wrapper
    class PolicyInference(torch.nn.Module):
        """
        Inference-only wrapper.
        
        NO training components:
        - No log_prob
        - No entropy
        - No value function
        - No exploration noise
        
        Pure deterministic policy.
        """
        
        def __init__(self, actor_mean_net, shared_net):
            super().__init__()
            self.shared_net = shared_net
            self.actor_mean = actor_mean_net
        
        def forward(self, obs):
            """
            Deterministic inference.
            
            Args:
                obs: [batch, obs_dim] state tensor
            
            Returns:
                action: [batch, action_dim] in [-1, 1]
            """
            features = self.shared_net(obs)
            action_mean = self.actor_mean(features)
            action = torch.tanh(action_mean)  # Squash to [-1, 1]
            return action
    
    # Extract only actor components
    inference_model = PolicyInference(
        actor_mean_net=policy.actor_mean,
        shared_net=policy.shared_net
    )
    inference_model.eval()
    
    print("🔒 Inference-only model created:")
    print("   ✅ Deterministic (no sampling)")
    print("   ✅ No gradients")
    print("   ✅ No training code")
    print("   ✅ Actor only (no critic)")
    print()
    
    # Dummy input for tracing
    dummy_input = torch.randn(1, obs_dim)
    
    # Export to ONNX
    print("Exporting to ONNX...")
    torch.onnx.export(
        inference_model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,  # Optimize
        input_names=['state'],
        output_names=['action'],
        dynamic_axes={
            'state': {0: 'batch_size'},
            'action': {0: 'batch_size'}
        }
    )
    
    print(f"✅ ONNX export complete: {output_path}")
    print()
    
    # Test exported model
    if test_input:
        print("Testing exported model...")
        import onnxruntime as ort
        
        # Load ONNX model
        session = ort.InferenceSession(output_path)
        
        # Test with multiple scenarios
        test_cases = [
            # [lane_offset, heading_error, speed, left_dist, right_dist, curvature]
            np.array([[0.0, 0.0, 20.0, 1.75, 1.75, 0.0]], dtype=np.float32),  # Perfect center
            np.array([[0.5, 0.1, 20.0, 1.25, 2.25, 0.0]], dtype=np.float32),  # Right offset
            np.array([[-0.5, -0.1, 20.0, 2.25, 1.25, 0.0]], dtype=np.float32), # Left offset
            np.array([[0.0, 0.0, 10.0, 1.75, 1.75, 0.02]], dtype=np.float32),  # Curve
        ]
        
        print("\nTest Cases:")
        print("-" * 60)
        for i, test_input in enumerate(test_cases, 1):
            result = session.run(['action'], {'state': test_input})[0]
            steering, throttle = result[0]
            
            print(f"Case {i}:")
            print(f"  State: lane_offset={test_input[0,0]:.2f}, "
                  f"heading={test_input[0,1]:.2f}, speed={test_input[0,2]:.1f}")
            print(f"  Action: steering={steering:.3f}, throttle={throttle:.3f}")
            
            # Verify bounds
            assert -1.0 <= steering <= 1.0, "Steering out of bounds!"
            assert -1.0 <= throttle <= 1.0, "Throttle out of bounds!"
            assert not np.isnan(steering), "NaN detected!"
            assert not np.isnan(throttle), "NaN detected!"
        
        print("-" * 60)
        print("✅ All tests passed!")
        print()
    
    # Model info
    model_size = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"Model size: {model_size:.2f} MB")
    print()
    
    print("="*60)
    print("✅ DEPLOYMENT READY")
    print("="*60)
    print()
    print("Next steps:")
    print("  1. Load in C++: onnxruntime::Session(\"policy.onnx\")")
    print("  2. Inference: session.Run(state) -> action")
    print("  3. Apply safety shield (Step 4)")
    print("  4. Deploy via Java REST API")
    print()


def main():
    parser = argparse.ArgumentParser(description="Export Policy to ONNX")
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained checkpoint (.pt file)')
    parser.add_argument('--output', type=str, default='policy.onnx',
                       help='Output ONNX file path')
    parser.add_argument('--opset', type=int, default=14,
                       help='ONNX opset version')
    parser.add_argument('--no-test', action='store_true',
                       help='Skip test inference')
    
    args = parser.parse_args()
    
    export_to_onnx(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        opset_version=args.opset,
        test_input=not args.no_test
    )


if __name__ == '__main__':
    main()
