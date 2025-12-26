#!/usr/bin/env python3
"""
ONNX Export Utility for Self-Driving RL Models

Exports trained PyTorch policy networks to ONNX format for deployment
in C++ inference engine.

Usage:
    python export_onnx.py --checkpoint checkpoints/model_final.pt
    python export_onnx.py --checkpoint model.pt --output policy.onnx --opset 14
"""

import argparse
import os
import sys
from pathlib import Path
import torch
import torch.onnx
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from rl.networks.actor_critic import ActorCritic, ActorCriticForExport


def parse_args():
    parser = argparse.ArgumentParser(description="Export RL Policy to ONNX")
    
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to PyTorch checkpoint")
    parser.add_argument("--output", type=str, default=None,
                        help="Output ONNX file path")
    parser.add_argument("--output-dir", type=str, default="models",
                        help="Output directory for ONNX model")
    
    # Model architecture (must match checkpoint)
    parser.add_argument("--input-channels", type=int, default=3,
                        help="Number of input channels")
    parser.add_argument("--input-height", type=int, default=96,
                        help="Input image height")
    parser.add_argument("--input-width", type=int, default=96,
                        help="Input image width")
    parser.add_argument("--action-dim", type=int, default=3,
                        help="Action dimension")
    parser.add_argument("--feature-dim", type=int, default=512,
                        help="Feature dimension")
    
    # ONNX options
    parser.add_argument("--opset", type=int, default=14,
                        help="ONNX opset version")
    parser.add_argument("--dynamic-batch", action="store_true",
                        help="Enable dynamic batch size")
    parser.add_argument("--simplify", action="store_true",
                        help="Simplify ONNX model (requires onnx-simplifier)")
    parser.add_argument("--fp16", action="store_true",
                        help="Export in FP16 precision")
    
    # Verification
    parser.add_argument("--verify", action="store_true",
                        help="Verify exported model")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run inference benchmark")
    
    return parser.parse_args()


def load_checkpoint(checkpoint_path: str, args) -> ActorCritic:
    """Load trained model from checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")
    
    # Create model with same architecture
    model = ActorCritic(
        input_channels=args.input_channels,
        input_height=args.input_height,
        input_width=args.input_width,
        action_dim=args.action_dim,
        feature_dim=args.feature_dim
    )
    
    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if 'policy_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['policy_state_dict'])
        print(f"  Loaded from timestep: {checkpoint.get('num_timesteps', 'unknown')}")
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model


def export_to_onnx(
    model: ActorCritic,
    output_path: str,
    args
) -> str:
    """Export model to ONNX format."""
    print("\nExporting to ONNX...")
    print(f"  Output: {output_path}")
    print(f"  Opset version: {args.opset}")
    
    # Create export-friendly wrapper
    export_model = ActorCriticForExport(model)
    export_model.eval()
    
    # Create dummy input
    batch_size = 1
    dummy_input = torch.randn(
        batch_size,
        args.input_channels,
        args.input_height,
        args.input_width
    )
    
    # Input/output names
    input_names = ["state"]
    output_names = ["action"]
    
    # Dynamic axes for variable batch size
    dynamic_axes = None
    if args.dynamic_batch:
        dynamic_axes = {
            "state": {0: "batch_size"},
            "action": {0: "batch_size"}
        }
        print("  Dynamic batch size: enabled")
    
    # Export
    torch.onnx.export(
        export_model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        verbose=False
    )
    
    print("  Export complete!")
    
    # Simplify if requested
    if args.simplify:
        try:
            import onnx
            from onnxsim import simplify
            
            print("\nSimplifying ONNX model...")
            onnx_model = onnx.load(output_path)
            simplified_model, check = simplify(onnx_model)
            
            if check:
                onnx.save(simplified_model, output_path)
                print("  Simplification successful!")
            else:
                print("  Simplification failed, keeping original")
        except ImportError:
            print("  onnx-simplifier not installed, skipping simplification")
            print("  Install with: pip install onnx-simplifier")
    
    # Convert to FP16 if requested
    if args.fp16:
        try:
            from onnxconverter_common import float16
            import onnx
            
            print("\nConverting to FP16...")
            onnx_model = onnx.load(output_path)
            fp16_model = float16.convert_float_to_float16(onnx_model)
            
            fp16_path = output_path.replace('.onnx', '_fp16.onnx')
            onnx.save(fp16_model, fp16_path)
            print(f"  FP16 model saved to: {fp16_path}")
        except ImportError:
            print("  onnxconverter-common not installed, skipping FP16 conversion")
    
    return output_path


def verify_onnx(onnx_path: str, pytorch_model: ActorCritic, args):
    """Verify ONNX model matches PyTorch model."""
    print("\nVerifying ONNX model...")
    
    try:
        import onnx
        import onnxruntime as ort
    except ImportError:
        print("  onnxruntime not installed, skipping verification")
        print("  Install with: pip install onnxruntime")
        return
    
    # Check ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("  ONNX model check passed!")
    
    # Create ONNX Runtime session
    ort_session = ort.InferenceSession(onnx_path)
    
    # Create test input
    test_input = np.random.randn(
        1,
        args.input_channels,
        args.input_height,
        args.input_width
    ).astype(np.float32)
    
    # PyTorch inference
    export_model = ActorCriticForExport(pytorch_model)
    export_model.eval()
    
    with torch.no_grad():
        pytorch_output = export_model(torch.from_numpy(test_input)).numpy()
    
    # ONNX Runtime inference
    ort_inputs = {ort_session.get_inputs()[0].name: test_input}
    ort_output = ort_session.run(None, ort_inputs)[0]
    
    # Compare outputs
    max_diff = np.max(np.abs(pytorch_output - ort_output))
    mean_diff = np.mean(np.abs(pytorch_output - ort_output))
    
    print(f"  Max difference: {max_diff:.6e}")
    print(f"  Mean difference: {mean_diff:.6e}")
    
    if max_diff < 1e-5:
        print("  Verification PASSED!")
    else:
        print("  WARNING: Outputs differ significantly!")


def benchmark_onnx(onnx_path: str, args, n_iterations: int = 1000):
    """Benchmark ONNX inference speed."""
    print(f"\nBenchmarking ONNX inference ({n_iterations} iterations)...")
    
    try:
        import onnxruntime as ort
    except ImportError:
        print("  onnxruntime not installed, skipping benchmark")
        return
    
    import time
    
    # Create session with different providers
    providers = ort.get_available_providers()
    print(f"  Available providers: {providers}")
    
    for provider in ['CUDAExecutionProvider', 'CPUExecutionProvider']:
        if provider not in providers:
            continue
        
        print(f"\n  Testing {provider}...")
        
        try:
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            ort_session = ort.InferenceSession(
                onnx_path,
                session_options,
                providers=[provider]
            )
            
            # Prepare input
            test_input = np.random.randn(
                1,
                args.input_channels,
                args.input_height,
                args.input_width
            ).astype(np.float32)
            
            input_name = ort_session.get_inputs()[0].name
            
            # Warmup
            for _ in range(100):
                ort_session.run(None, {input_name: test_input})
            
            # Benchmark
            start_time = time.time()
            for _ in range(n_iterations):
                ort_session.run(None, {input_name: test_input})
            elapsed = time.time() - start_time
            
            avg_time_ms = (elapsed / n_iterations) * 1000
            fps = n_iterations / elapsed
            
            print(f"    Average inference time: {avg_time_ms:.3f} ms")
            print(f"    Throughput: {fps:.1f} FPS")
            
        except Exception as e:
            print(f"    Failed: {e}")


def print_model_info(onnx_path: str):
    """Print ONNX model information."""
    try:
        import onnx
    except ImportError:
        return
    
    model = onnx.load(onnx_path)
    
    print("\nModel Information:")
    print(f"  IR version: {model.ir_version}")
    print(f"  Producer: {model.producer_name} {model.producer_version}")
    print(f"  Opset version: {model.opset_import[0].version}")
    
    # File size
    file_size = os.path.getsize(onnx_path)
    if file_size > 1024 * 1024:
        print(f"  File size: {file_size / (1024*1024):.2f} MB")
    else:
        print(f"  File size: {file_size / 1024:.2f} KB")
    
    # Input/Output info
    print("\n  Inputs:")
    for input in model.graph.input:
        shape = [d.dim_value if d.dim_value else d.dim_param 
                 for d in input.type.tensor_type.shape.dim]
        print(f"    {input.name}: {shape}")
    
    print("\n  Outputs:")
    for output in model.graph.output:
        shape = [d.dim_value if d.dim_value else d.dim_param 
                 for d in output.type.tensor_type.shape.dim]
        print(f"    {output.name}: {shape}")


def main():
    args = parse_args()
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        checkpoint_name = Path(args.checkpoint).stem
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(output_dir / f"{checkpoint_name}.onnx")
    
    # Load checkpoint
    model = load_checkpoint(args.checkpoint, args)
    
    # Export to ONNX
    export_to_onnx(model, output_path, args)
    
    # Print model info
    print_model_info(output_path)
    
    # Verify if requested
    if args.verify:
        verify_onnx(output_path, model, args)
    
    # Benchmark if requested
    if args.benchmark:
        benchmark_onnx(output_path, args)
    
    print(f"\n{'='*60}")
    print(f"Export complete: {output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
