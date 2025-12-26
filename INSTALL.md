# Installation Guide

## Quick Start (Python Only)

If you only want to train models without C++/Java components:

```bash
cd python
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Start training
python train_self_driving.py --total-timesteps 500000
```

## Full Installation (Python + C++ + Java)

### 1. Install ONNX Runtime (Required for C++)

#### Option A: Download Pre-built (Recommended)

```bash
# Download ONNX Runtime
cd ~
wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-linux-x64-1.16.3.tgz
tar -xzf onnxruntime-linux-x64-1.16.3.tgz

# For GPU support (if you have CUDA)
# wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-linux-x64-gpu-1.16.3.tgz

# Set environment variable
export ONNXRUNTIME_ROOT=~/onnxruntime-linux-x64-1.16.3
echo 'export ONNXRUNTIME_ROOT=~/onnxruntime-linux-x64-1.16.3' >> ~/.bashrc
```

#### Option B: Build from Source

```bash
git clone --recursive https://github.com/microsoft/onnxruntime.git
cd onnxruntime
./build.sh --config Release --build_shared_lib --parallel
export ONNXRUNTIME_ROOT=$PWD/build/Linux/Release
```

### 2. Build Everything

```bash
cd ~/Self-DrivingDeepRLSystem

# Build with ONNX Runtime path
ONNXRUNTIME_ROOT=~/onnxruntime-linux-x64-1.16.3 ./build.sh all

# Or if you want GPU support
ONNXRUNTIME_ROOT=~/onnxruntime-linux-x64-1.16.3 USE_CUDA=1 ./build.sh all
```

### 3. Create Gradle Wrapper

```bash
cd java
gradle wrapper
```

## Component-by-Component Installation

### Python Only

```bash
cd python
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### C++ Only

```bash
# Install ONNX Runtime first (see above)
export ONNXRUNTIME_ROOT=/path/to/onnxruntime

cd cpp
mkdir build && cd build
cmake .. -DONNXRUNTIME_ROOT=$ONNXRUNTIME_ROOT
make -j$(nproc)

# Test
./bin/test_engine
```

### Java Only

```bash
# Requires C++ library to be built first
cd java
./gradlew build
```

## Troubleshooting

### ONNX Runtime not found

**Error:**
```
CMake Warning: ONNX Runtime not found
fatal error: onnxruntime_cxx_api.h: No such file or directory
```

**Solution:**
```bash
# Download and set ONNXRUNTIME_ROOT
wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-linux-x64-1.16.3.tgz
tar -xzf onnxruntime-linux-x64-1.16.3.tgz
export ONNXRUNTIME_ROOT=$PWD/onnxruntime-linux-x64-1.16.3

# Rebuild C++
cd cpp/build
cmake .. -DONNXRUNTIME_ROOT=$ONNXRUNTIME_ROOT
make clean && make
```

### Module 'gym' not found

**Error:**
```
ModuleNotFoundError: No module named 'gym'
```

**Solution:**
This has been fixed. The code now uses `gymnasium` which is already installed.

### Virtual environment not activated

**Error:**
```
python: command not found (or using wrong Python)
```

**Solution:**
```bash
cd python
source venv/bin/activate
```

### CUDA not found (for GPU build)

**Error:**
```
CMake Error: Could not find CUDA
```

**Solution:**
Either:
- Install CUDA toolkit: `sudo apt install nvidia-cuda-toolkit`
- Or disable CUDA: Don't use `USE_CUDA=1` flag

### Gradle not found

**Solution:**
```bash
# Install Gradle
sudo apt install gradle

# Or use SDKMAN
curl -s "https://get.sdkman.io" | bash
sdk install gradle
```

## Verify Installation

### Python

```bash
cd python
source venv/bin/activate
python -c "import torch; import gymnasium; print('Python OK')"
```

### C++

```bash
cd cpp/build
./bin/test_engine --help
```

### Java

```bash
cd java
./gradlew test
```

## Platform-Specific Notes

### Ubuntu/Debian

```bash
# Install build dependencies
sudo apt update
sudo apt install -y build-essential cmake openjdk-17-jdk python3-venv python3-dev

# Install CUDA (optional, for GPU)
sudo apt install nvidia-cuda-toolkit
```

### macOS

```bash
# Install dependencies
brew install cmake openjdk python

# Note: CUDA is not available on macOS
# Use CPU-only ONNX Runtime
```

### Windows

Use WSL2 (Windows Subsystem for Linux) and follow Ubuntu instructions.

## Next Steps

After installation:

1. **Train a model:**
   ```bash
   ./train.sh 500000
   ```

2. **Export to ONNX:**
   ```bash
   cd python
   source venv/bin/activate
   python export_onnx.py --checkpoint logs/carracing_ppo/checkpoints/model_final.pt
   ```

3. **Test C++ inference:**
   ```bash
   cpp/build/bin/test_engine models/self_driving_policy.onnx --benchmark
   ```

4. **Run Java server:**
   ```bash
   ./run_server.sh
   ```
