# Getting Started - Quick Guide

## ✅ What's Working Now

Your Python environment is set up with all dependencies installed including:
- PyTorch 2.9.1 with CUDA support
- Gymnasium (CarRacing environment)
- ONNX and ONNX Runtime
- All training dependencies

## 🚀 Start Training Immediately

```bash
cd /home/steven/Self-DrivingDeepRLSystem/python
source venv/bin/activate
python train_self_driving.py --total-timesteps 500000 --n-envs 1
```

Or use the interactive quickstart:
```bash
cd /home/steven/Self-DrivingDeepRLSystem
./quickstart.sh
```

## 📋 Current Issues & Solutions

### ✅ FIXED: gym import error
- **Status:** Fixed automatically
- **What was wrong:** Code imported `gym` instead of `gymnasium`
- **Solution:** Updated [wrappers.py](python/rl/envs/wrappers.py) to use `gymnasium as gym`

### ⚠️ C++ Build (Optional - Only needed for inference server)
- **Issue:** ONNX Runtime C++ library not installed
- **Impact:** Can't build C++ inference engine or Java REST API
- **When you need it:** Only if you want to deploy the trained model in production
- **Solution:** See [INSTALL.md](INSTALL.md) for detailed ONNX Runtime installation

## 📚 Recommended Workflow

### Phase 1: Train a Model (You can do this now!)

```bash
cd python
source venv/bin/activate

# Quick test (5-10 minutes)
python train_self_driving.py --total-timesteps 100000

# Real training (2-4 hours, best results)
python train_self_driving.py \
    --total-timesteps 1000000 \
    --n-envs 4 \
    --exp-name my_first_model
```

### Phase 2: Evaluate & Export

```bash
# Watch your trained agent drive
python evaluate.py \
    --checkpoint logs/my_first_model/checkpoints/model_final.pt \
    --n-episodes 5 \
    --render

# Export to ONNX format
python export_onnx.py \
    --checkpoint logs/my_first_model/checkpoints/model_final.pt \
    --output ../models/self_driving_policy.onnx \
    --verify
```

### Phase 3: C++ & Java (Optional - Later)

Only needed if you want to:
- Deploy the model in a production environment
- Create a REST API for the driving agent
- Integrate with other applications via JNI

See [INSTALL.md](INSTALL.md) for installing ONNX Runtime C++ library.

## 🎯 Training Tips

### Quick Test Run
```bash
# Fast test to verify everything works (10 mins)
python train_self_driving.py --total-timesteps 100000 --n-envs 1
```

### Parallel Training (Recommended)
```bash
# Use multiple environments for faster training
python train_self_driving.py --total-timesteps 1000000 --n-envs 4
```

### Monitor Progress
```bash
# TensorBoard logging is automatic
tensorboard --logdir logs
# Open http://localhost:6006 in browser
```

### Key Hyperparameters
```bash
python train_self_driving.py \
    --total-timesteps 1000000 \    # Total training steps
    --n-envs 4 \                   # Parallel environments (faster)
    --learning-rate 3e-4 \         # Learning rate
    --batch-size 64 \              # Mini-batch size
    --n-steps 2048 \               # Rollout length
    --save-freq 50000 \            # Checkpoint frequency
    --eval-freq 50000              # Evaluation frequency
```

## 📊 What to Expect

### Training Progress
- **0-200K steps:** Agent learns basic control (steering, acceleration)
- **200K-500K steps:** Stays on track, improving speed
- **500K-1M steps:** Optimizes racing line, maximum score

### Performance Metrics
- **Target score:** 900+ (out of 1000)
- **Training time:** 2-4 hours for 1M steps (with GPU)
- **Episodes to solve:** Usually 500-1000 episodes

### Output Files
```
logs/
└── my_first_model/
    ├── checkpoints/
    │   ├── model_50000.pt
    │   ├── model_100000.pt
    │   └── model_final.pt
    ├── tensorboard/
    │   └── events.out.tfevents.*
    └── config.json
```

## 🔍 Troubleshooting

### Already Solved ✅
- ~~gym import error~~ → Fixed
- ~~Python dependencies~~ → All installed

### If training crashes
```bash
# Check GPU memory
nvidia-smi

# Reduce number of parallel environments
python train_self_driving.py --n-envs 1

# Or disable CUDA
python train_self_driving.py --device cpu
```

### If you see "No module named 'gym'"
```bash
# Make sure you're using the venv
cd python
source venv/bin/activate
```

### If you want GPU acceleration
```bash
# Check if CUDA is available
python -c "import torch; print(torch.cuda.is_available())"

# Should print: True
```

## 🎓 Next Steps

1. **Start with quickstart:**
   ```bash
   ./quickstart.sh
   ```

2. **Or go directly to training:**
   ```bash
   cd python && source venv/bin/activate
   python train_self_driving.py --total-timesteps 500000
   ```

3. **Monitor with TensorBoard (optional):**
   ```bash
   # In a new terminal
   cd /home/steven/Self-DrivingDeepRLSystem/python
   source venv/bin/activate
   tensorboard --logdir logs
   ```

4. **When ready for deployment:** See [INSTALL.md](INSTALL.md) for C++/Java setup

## 📖 Documentation

- [INSTALL.md](INSTALL.md) - Detailed installation guide (C++/Java)
- [README.md](README.md) - Full project documentation
- [python/](python/) - Python training code

## 💡 Pro Tips

- Use `--n-envs 4` for 4x faster training (if you have enough RAM)
- Save checkpoints frequently with `--save-freq 25000`
- Watch TensorBoard to see learning progress in real-time
- Start with a short run (100K steps) to verify everything works
- The agent typically needs 500K-1M steps to learn well
