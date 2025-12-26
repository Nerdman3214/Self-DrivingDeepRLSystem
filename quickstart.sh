#!/bin/bash
# Quick start script - Python training only

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================================"
echo "Self-Driving RL Quick Start (Python Only)"
echo "============================================================"

# Step 1: Setup Python environment
if [ ! -d "$SCRIPT_DIR/python/venv" ]; then
    echo "Step 1: Creating Python virtual environment..."
    cd "$SCRIPT_DIR/python"
    python3 -m venv venv
    source venv/bin/activate
    
    echo "Step 2: Installing dependencies (this may take a while)..."
    pip install --upgrade pip
    pip install -r requirements.txt
else
    echo "Step 1: Activating existing virtual environment..."
    cd "$SCRIPT_DIR/python"
    source venv/bin/activate
fi

echo ""
echo "============================================================"
echo "Environment ready!"
echo "============================================================"
echo ""
echo "Available commands:"
echo ""
echo "1. Train a model (500K timesteps, ~30 minutes):"
echo "   python train_self_driving.py --total-timesteps 500000"
echo ""
echo "2. Train longer (1M timesteps, better performance):"
echo "   python train_self_driving.py --total-timesteps 1000000"
echo ""
echo "3. Train with multiple environments (faster):"
echo "   python train_self_driving.py --total-timesteps 1000000 --n-envs 4"
echo ""
echo "4. Export to ONNX after training:"
echo "   python export_onnx.py --checkpoint logs/carracing_ppo/checkpoints/model_final.pt"
echo ""
echo "5. Evaluate trained model:"
echo "   python evaluate.py --checkpoint logs/carracing_ppo/checkpoints/model_final.pt --render"
echo ""
echo "============================================================"
echo ""

# Ask if user wants to start training
read -p "Start training now with default settings? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Starting training with 500,000 timesteps..."
    python train_self_driving.py --total-timesteps 500000 --n-envs 1
fi
