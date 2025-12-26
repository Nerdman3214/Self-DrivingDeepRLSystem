# Lane-Keeping Self-Driving System 🚗

**State-Based Reinforcement Learning for Autonomous Lane Keeping**

This is a clean, production-ready implementation of a fundamental self-driving primitive using Deep RL.

## 🎯 What This System Teaches

The agent learns to:
- ✅ **Stay centered in the lane** (lane offset minimization)
- ✅ **Align with road heading** (heading error correction)
- ✅ **Maintain target speed** (velocity control)
- ✅ **Handle curves smoothly** (curvature adaptation)
- ✅ **Recover from disturbances** (robustness)

## 🧠 Why This Matters

This mirrors **real autonomous vehicle systems**:

| Component | Real AV Stack | This System |
|-----------|--------------|-------------|
| Perception | Camera/LiDAR → State Estimation | Direct state vector |
| Planning | Path planner | Learned policy |
| Control | MPC/PID controller | RL policy |
| Reward | Human-designed cost function | Reward function |

**Key Insight:** We skip the perception layer to focus on control policy learning.

## 📊 System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   TRAINING LOOP (Python)                     │
│                                                              │
│   ┌──────────────┐      ┌──────────────┐     ┌───────────┐ │
│   │ Lane-Keeping │──────▶│  PPO Agent   │────▶│  Policy   │ │
│   │ Environment  │ state │ (MLP-based)  │     │  Network  │ │
│   │              │◀──────│              │◀────│           │ │
│   └──────────────┘action └──────────────┘     └───────────┘ │
│         │                      │                             │
│         │ reward               │ optimize                    │
│         ▼                      ▼                             │
│   ┌──────────────┐      ┌──────────────┐                    │
│   │   Reward     │      │   Rollout    │                    │
│   │  Function    │      │   Buffer     │                    │
│   └──────────────┘      └──────────────┘                    │
└─────────────────────────────────────────────────────────────┘
```

## 🔬 Technical Specification

### State Space (6D Vector)

```python
state = [
    lane_offset,      # Distance from lane center [-1, 1]
    heading_error,    # Angle relative to lane [-π/4, π/4] 
    speed,            # Current speed normalized [0, 1]
    left_lane_dist,   # Distance to left boundary [0, 1]
    right_lane_dist,  # Distance to right boundary [0, 1]
    curvature,        # Road curvature ahead [-1, 1]
]
```

### Action Space (2D Continuous)

```python
action = [
    steering,   # Wheel angle [-1, 1]
    throttle,   # Acceleration/brake [-1, 1]
]
```

### Reward Function

$$
R = w_1 \cdot (1 - |lane\_offset|) - w_2 \cdot |heading\_error| - w_3 \cdot |steering| - w_4 \cdot collision
$$

**Default weights:**
- w₁ = 1.0 (lane centering)
- w₂ = 0.5 (heading alignment)
- w₃ = 0.1 (smooth steering)
- w₄ = 10.0 (collision penalty)

### Network Architecture

```
Input (6D state)
    ↓
MLP Feature Extractor
    Linear(6, 256) → ReLU
    Linear(256, 256) → ReLU
    ↓
Actor Head (Policy)                Critic Head (Value)
    Linear(256, 128) → ReLU            Linear(256, 128) → ReLU
    Linear(128, 2) → Tanh              Linear(128, 1)
    ↓                                  ↓
Action μ (mean)                    V(s)
    ↓
+ Learnable log_std
    ↓
Gaussian Policy: N(μ, σ²)
```

## 🚀 Quick Start

### 1. Test the Environment

```bash
cd python
source venv/bin/activate
python test_lane_keeping.py
```

**Expected output:**
```
Testing Lane-Keeping Environment...
Observation space: Box([-1.0, -0.785, 0.0, 0.0, 0.0, -1.0], [1.0, 0.785, 1.0, 1.0, 1.0, 1.0])
Action space: Box([-1.0, -1.0], [1.0, 1.0])
✅ Environment test passed!
```

### 2. Train the Agent

```bash
# Quick training (5-10 minutes)
python train_lane_keeping.py --total-timesteps 100000

# Full training (1-2 hours)
python train_lane_keeping.py --total-timesteps 500000 --n-epochs 10

# Advanced: Multi-CPU training
python train_lane_keeping.py --total-timesteps 1000000 --n-envs 4
```

### 3. Monitor Training

```bash
# In another terminal
tensorboard --logdir logs/lane_keeping_ppo
# Open http://localhost:6006
```

### 4. Evaluate Trained Agent

```bash
python evaluate_lane_keeping.py \
    --checkpoint logs/lane_keeping_ppo/checkpoints/model_final.pt \
    --n-episodes 10 \
    --render
```

## 📈 Training Progress

### What to Expect

| Timesteps | Behavior | Mean Reward |
|-----------|----------|-------------|
| 0-50K | Random exploration, learns basic steering | -10 to 0 |
| 50K-150K | Stays on road, corrects heading | 0 to 200 |
| 150K-300K | Smooth lane keeping, handles curves | 200 to 400 |
| 300K+ | Optimal control, minimal corrections | 400+ |

### Key Metrics

Monitor these in TensorBoard:

1. **Episode Reward** - Should increase steadily
2. **Lane Offset** - Should decrease toward 0
3. **Completion Rate** - % of episodes reaching end without crash
4. **Policy Loss** - Should stabilize
5. **KL Divergence** - Should stay < target_kl (0.01)

## 🔧 Hyperparameter Tuning

### Learning Rate

```bash
# Conservative (stable but slow)
--learning-rate 1e-4

# Default (good balance)
--learning-rate 3e-4

# Aggressive (faster but less stable)
--learning-rate 1e-3
```

### PPO Clip Range

```bash
# Tight clipping (conservative updates)
--clip-range 0.1

# Default
--clip-range 0.2

# Loose clipping (larger updates)
--clip-range 0.3
```

### Network Size

```bash
# Small (fast, less capacity)
--hidden-dims 128 128

# Default
--hidden-dims 256 256

# Large (slower, more capacity)
--hidden-dims 512 512 256
```

## 🎮 Files to Debug

When something goes wrong, debug in this order:

### 1. Environment Logic
**File:** [`rl/envs/lane_keeping_env.py`](python/rl/envs/lane_keeping_env.py)
- **Breakpoint:** `step()` method, line ~150
- **Check:** Vehicle dynamics, reward calculation, termination conditions

### 2. Policy Network
**File:** [`rl/networks/mlp_policy.py`](python/rl/networks/mlp_policy.py)
- **Breakpoint:** `get_action_and_value()`, line ~100
- **Check:** Action distribution, value estimates

### 3. PPO Algorithm
**File:** [`rl/algorithms/ppo.py`](python/rl/algorithms/ppo.py)
- **Breakpoint:** PPO update loop
- **Check:** Advantage computation, policy/value losses

### 4. Training Loop
**File:** [`train_lane_keeping.py`](python/train_lane_keeping.py)
- **Breakpoint:** Main training loop, line ~250
- **Check:** Data collection, buffer management, update frequency

### 5. Rollout Buffer
**File:** [`rl/algorithms/rollout_buffer.py`](python/rl/algorithms/rollout_buffer.py)
- **Breakpoint:** `compute_returns_and_advantages()`
- **Check:** GAE computation, return calculation

## 🐛 Common Issues

### Reward not increasing

**Cause:** Reward function mismatch or bad initialization

**Fix:**
```bash
# Check reward weights
python test_lane_keeping.py  # See reward values

# Adjust weights in training
python train_lane_keeping.py --reward-weights '{"lane_centering": 2.0}'
```

### Agent goes off-road immediately

**Cause:** Initial policy too random, log_std too high

**Fix:**
```bash
# Lower initial exploration
python train_lane_keeping.py --log-std-init -1.0
```

### Training unstable (reward oscillates)

**Cause:** Learning rate too high or PPO clip range too large

**Fix:**
```bash
# Reduce learning rate and clip range
python train_lane_keeping.py --learning-rate 1e-4 --clip-range 0.1
```

### Value function not learning

**Cause:** Value coefficient too low

**Fix:**
```bash
python train_lane_keeping.py --value-coef 1.0
```

## 📚 Design Patterns Used

### Strategy Pattern
- Reward function is swappable via config
- Different policies (MLP, CNN) can be plugged in

### Template Method
- `step()` always follows: apply action → update physics → compute reward → check termination

### Separation of Concerns
- Environment (lane_keeping_env.py)
- Policy (mlp_policy.py)
- Algorithm (ppo.py)
- Training (train_lane_keeping.py)

## 🔬 Extending the System

### Add New Observations

Edit [`lane_keeping_env.py`](python/rl/envs/lane_keeping_env.py):

```python
# In _get_observation()
observation = np.array([
    lane_offset,
    heading_error,
    speed_normalized,
    left_dist,
    right_dist,
    curvature,
    acceleration,  # NEW
    jerk,          # NEW
], dtype=np.float32)
```

Update `observation_space` in `__init__()`.

### Modify Reward Function

Edit reward weights:

```python
self.reward_weights = {
    'lane_centering': 2.0,      # Increase importance
    'heading_alignment': 0.5,
    'smooth_steering': 0.2,     # Penalize jerky control
    'collision': 20.0,          # Harsher penalty
    'speed_target': 0.5,        # Encourage speed
}
```

### Add Obstacles

```python
# In step()
if self._check_obstacle_collision():
    terminated = True
    reward -= 50.0
```

## 📖 References

- [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347)
- [OpenAI Spinning Up](https://spinningup.openai.com/)
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)

## ✅ Step 1 Complete

You now have:
- ✅ Custom lane-keeping environment
- ✅ State-action-reward design
- ✅ MLP-based policy network
- ✅ PPO training pipeline
- ✅ Testing & debugging tools

**Next:** Train your first agent and analyze its behavior!
