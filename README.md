# Self-Driving Deep Reinforcement Learning System

A production-ready self-driving RL system featuring:

- **Python**: PPO training with PyTorch on CarRacing-v2
- **ONNX**: Model export for cross-platform deployment  
- **C++**: High-performance ONNX Runtime inference engine
- **Java**: JNI bridge + Spring Boot REST API

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    TRAINING (Python)                            │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐ │
│  │  CarRacing  │───▶│  PPO Agent  │───▶│  Export ONNX Model  │ │
│  │  Simulator  │    │  (PyTorch)  │    │                     │ │
│  └─────────────┘    └─────────────┘    └──────────┬──────────┘ │
└───────────────────────────────────────────────────┼─────────────┘
                                                    │
                                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                    INFERENCE (C++)                              │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │              ONNX Runtime Inference Engine                 │ │
│  │  • GPU acceleration (CUDA)                                 │ │
│  │  • Optimized for real-time (<10ms latency)                 │ │
│  │  • Image preprocessing                                     │ │
│  └──────────────────────────┬─────────────────────────────────┘ │
└─────────────────────────────┼───────────────────────────────────┘
                              │ JNI
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    API LAYER (Java)                             │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │              Spring Boot REST API                          │ │
│  │  • POST /api/v1/drive - Run inference                      │ │
│  │  • GET  /api/v1/health - Health check                      │ │
│  │  • POST /api/v1/benchmark - Performance test               │ │
│  └──────────────────────────┬─────────────────────────────────┘ │
└─────────────────────────────┼───────────────────────────────────┘
                              │ HTTP/REST
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    EXTERNAL CLIENTS                             │
│  • Simulator UI          • Dashboard          • Controllers    │
└─────────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
Self-DrivingDeepRLSystem/
├── python/                      # Training & Export
│   ├── rl/
│   │   ├── algorithms/          # PPO implementation
│   │   │   ├── ppo.py
│   │   │   └── rollout_buffer.py
│   │   ├── networks/            # Neural network architectures
│   │   │   ├── actor_critic.py
│   │   │   └── policy_networks.py
│   │   ├── envs/                # Environment wrappers
│   │   │   ├── wrappers.py
│   │   │   └── vec_env.py
│   │   └── utils/               # Utilities
│   │       ├── logger.py
│   │       ├── scheduler.py
│   │       └── helpers.py
│   ├── train_self_driving.py    # Main training script
│   ├── export_onnx.py           # ONNX export utility
│   ├── evaluate.py              # Evaluation script
│   └── requirements.txt
│
├── cpp/                         # C++ Inference Engine
│   ├── include/
│   │   ├── inference_engine.h
│   │   └── jni_bridge.h
│   ├── src/
│   │   ├── inference_engine.cpp
│   │   ├── jni_bridge.cpp
│   │   └── test_engine.cpp
│   └── CMakeLists.txt
│
├── java/                        # Java REST API
│   ├── src/main/java/com/selfdriving/
│   │   ├── SelfDrivingApplication.java
│   │   ├── engine/
│   │   │   ├── SelfDrivingEngine.java
│   │   │   └── DrivingAction.java
│   │   └── api/
│   │       ├── SelfDrivingController.java
│   │       └── InferenceService.java
│   ├── src/main/resources/
│   │   └── application.properties
│   └── build.gradle
│
├── models/                      # Saved models
└── README.md
```

## 🚀 Quick Start

**New to this project? Start here:**
- **[QUICKSTART.md](QUICKSTART.md)** - Get training in 2 minutes
- **[INSTALL.md](INSTALL.md)** - Detailed installation (C++/Java components)

### Fastest Path: Train Now (Python Only)

```bash
# One command to start training
./quickstart.sh

# Or manually:
cd python
source venv/bin/activate
python train_self_driving.py --total-timesteps 500000
```

### Full Stack Setup (All Components)

### Prerequisites

- Python 3.9+
- PyTorch 2.0+
- CMake 3.16+
- ONNX Runtime 1.16+
- Java 17+
- Gradle 8.0+

### 1. Training (Python)

```bash
# Install dependencies
cd python
pip install -r requirements.txt

# Train the agent
python train_self_driving.py \
    --total-timesteps 1000000 \
    --n-envs 4 \
    --log-dir logs/experiment1

# Evaluate trained model
python evaluate.py \
    --checkpoint logs/experiment1/checkpoints/model_final.pt \
    --n-episodes 10 \
    --render

# Export to ONNX
python export_onnx.py \
    --checkpoint logs/experiment1/checkpoints/model_final.pt \
    --output ../models/self_driving_policy.onnx \
    --verify
```

### 2. Build C++ Engine

```bash
# Download ONNX Runtime
# https://github.com/microsoft/onnxruntime/releases

# Build
cd cpp
mkdir build && cd build
cmake .. \
    -DONNXRUNTIME_ROOT=/path/to/onnxruntime \
    -DUSE_CUDA=ON \
    -DBUILD_JNI=ON
make -j$(nproc)

# Test
./bin/test_engine ../models/self_driving_policy.onnx --benchmark
```

### 3. Run Java Server

```bash
cd java

# Set native library path
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:../cpp/build/lib

# Run server
./gradlew bootRun

# Or build and run JAR
./gradlew bootJar
java -Djava.library.path=../cpp/build/lib -jar build/libs/selfdriving-server-1.0.0.jar
```

### 4. API Usage

```bash
# Health check
curl http://localhost:8080/api/v1/health

# Get stats
curl http://localhost:8080/api/v1/stats

# Run inference
curl -X POST http://localhost:8080/api/v1/drive \
    -H "Content-Type: application/json" \
    -d '{"state": [0.5, 0.3, ...(27648 floats)...]}'

# Run benchmark
curl -X POST http://localhost:8080/api/v1/benchmark \
    -H "Content-Type: application/json" \
    -d '{"iterations": 1000}'
```

## 🎮 API Reference

### POST /api/v1/drive

Run inference on a state observation.

**Request:**
```json
{
    "state": [0.1, 0.2, ...]  // Flattened [3, 96, 96] normalized to [0,1]
}
```

**Response:**
```json
{
    "action": {
        "steering": 0.35,
        "gas": 0.8,
        "brake": 0.0,
        "raw": [0.35, 0.8, 0.0]
    },
    "latencyMs": 5.2,
    "description": "Turn Right (35%) + Gas (80%)"
}
```

### POST /api/v1/drive/image

Run inference on raw image data.

**Request:**
```json
{
    "imageData": "base64_encoded_rgb_bytes",
    "height": 96,
    "width": 96
}
```

### GET /api/v1/health

Health check endpoint.

**Response:**
```json
{
    "status": "healthy",
    "ready": true,
    "timestamp": 1703500800000
}
```

## 🧠 RL Algorithm: PPO

This system uses Proximal Policy Optimization (PPO), ideal for continuous control:

### Key Features

- **Clipped Surrogate Objective**: Prevents destructive policy updates
- **Generalized Advantage Estimation (GAE)**: Reduces variance
- **Value Function Clipping**: Stabilizes training
- **Entropy Bonus**: Encourages exploration

### Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| learning_rate | 3e-4 | Adam learning rate |
| n_steps | 2048 | Steps per rollout |
| batch_size | 64 | Mini-batch size |
| n_epochs | 10 | Epochs per update |
| gamma | 0.99 | Discount factor |
| gae_lambda | 0.95 | GAE lambda |
| clip_range | 0.2 | PPO clip range |
| entropy_coef | 0.01 | Entropy coefficient |
| value_coef | 0.5 | Value loss coefficient |

### Network Architecture

```
Input: [batch, 3, 96, 96] RGB image

CNN Feature Extractor:
  Conv2d(3, 32, 8x8, stride=4) → ReLU
  Conv2d(32, 64, 4x4, stride=2) → ReLU
  Conv2d(64, 64, 3x3, stride=1) → ReLU
  Flatten → Linear(64*7*7, 512) → ReLU

Actor Head (Policy):
  Linear(512, 256) → ReLU → Linear(256, 3) → Tanh
  Output: μ (mean actions)
  Learnable log_std parameter

Critic Head (Value):
  Linear(512, 256) → ReLU → Linear(256, 1)
  Output: V(s)
```

## ⚡ Performance

### Inference Latency

| Platform | Latency (ms) | Throughput (FPS) |
|----------|-------------|------------------|
| CPU (i7-12700) | ~15ms | ~66 FPS |
| GPU (RTX 3080) | ~3ms | ~330 FPS |
| GPU (RTX 4090) | ~1.5ms | ~666 FPS |

### Training

| Metric | Value |
|--------|-------|
| Environment | CarRacing-v2 |
| Training time | ~2-4 hours (1M steps) |
| Episodes to solve | ~500-1000 |
| Target score | 900+ |

## 🔧 Configuration

### Training Configuration

```bash
python train_self_driving.py --help

Options:
  --total-timesteps INT   Total training timesteps [default: 1000000]
  --n-envs INT            Parallel environments [default: 1]
  --learning-rate FLOAT   Learning rate [default: 3e-4]
  --n-steps INT           Steps per rollout [default: 2048]
  --batch-size INT        Mini-batch size [default: 64]
  --gamma FLOAT           Discount factor [default: 0.99]
  --device STR            Device (cpu, cuda) [default: auto]
  --seed INT              Random seed [default: 42]
```

### Server Configuration

```properties
# application.properties
selfdriving.model.path=models/self_driving_policy.onnx
selfdriving.gpu.enabled=true
selfdriving.gpu.device=0
selfdriving.warmup.iterations=10
server.port=8080
```

## 📚 References

- [Proximal Policy Optimization (PPO)](https://arxiv.org/abs/1707.06347)
- [ONNX Runtime](https://onnxruntime.ai/)
- [OpenAI Gym](https://gymnasium.farama.org/)

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.
