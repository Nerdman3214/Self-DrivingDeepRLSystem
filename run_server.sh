#!/bin/bash
# Start the Self-Driving REST API server

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/java"

# Set native library path
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$SCRIPT_DIR/cpp/build/lib"

# Check if native library exists
if [ ! -f "$SCRIPT_DIR/cpp/build/lib/libselfdriving_jni.so" ] && [ ! -f "$SCRIPT_DIR/cpp/build/lib/selfdriving_jni.so" ]; then
    echo "Warning: Native library not found at $SCRIPT_DIR/cpp/build/lib"
    echo "Build the C++ engine first: cd cpp && mkdir build && cd build && cmake .. && make"
    echo ""
fi

# Check if model exists
MODEL_PATH="$SCRIPT_DIR/models/self_driving_policy.onnx"
if [ ! -f "$MODEL_PATH" ]; then
    echo "Warning: Model not found at $MODEL_PATH"
    echo "Train and export a model first:"
    echo "  1. python python/train_self_driving.py"
    echo "  2. python python/export_onnx.py --checkpoint <checkpoint.pt>"
    echo ""
fi

echo "============================================================"
echo "Self-Driving REST API Server"
echo "============================================================"
echo "Native library path: $LD_LIBRARY_PATH"
echo "Model path: $MODEL_PATH"
echo "============================================================"
echo ""

# Run with Gradle
if [ -f "gradlew" ]; then
    ./gradlew bootRun
else
    gradle bootRun
fi
