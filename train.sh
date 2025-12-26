#!/bin/bash
# Quick training script for Self-Driving RL System

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/python"

# Default parameters
TOTAL_TIMESTEPS=${1:-500000}
N_ENVS=${2:-1}
EXP_NAME=${3:-"carracing_ppo"}

echo "============================================================"
echo "Self-Driving RL Training"
echo "============================================================"
echo "Timesteps: $TOTAL_TIMESTEPS"
echo "Environments: $N_ENVS"
echo "Experiment: $EXP_NAME"
echo "============================================================"

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Run training
python train_self_driving.py \
    --total-timesteps "$TOTAL_TIMESTEPS" \
    --n-envs "$N_ENVS" \
    --exp-name "$EXP_NAME" \
    --log-dir "logs" \
    --save-freq 50000 \
    --eval-freq 50000 \
    --n-eval-episodes 5

echo ""
echo "Training complete!"
echo "Checkpoints saved to: logs/$EXP_NAME/checkpoints/"
echo ""
echo "Next steps:"
echo "  1. Export to ONNX:"
echo "     python export_onnx.py --checkpoint logs/$EXP_NAME/checkpoints/model_final.pt"
echo ""
echo "  2. Evaluate:"
echo "     python evaluate.py --checkpoint logs/$EXP_NAME/checkpoints/model_final.pt --render"
