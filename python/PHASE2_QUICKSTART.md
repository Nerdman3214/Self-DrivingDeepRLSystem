# 🚀 PHASE 2 QUICKSTART

**Complete this in 2 minutes to verify Phase 2 works.**

---

## ✅ Prerequisites

Phase 1 complete (have recorded episodes in `episodes/` directory).

---

## 🧪 Test 1: Basic Offline Evaluation

```bash
cd python

# Evaluate random policy on recorded episodes
python offline_evaluation_demo.py --episodes episodes/episode_*.json.gz
```

**Expected output:**
```
📊 PHASE 2: OFFLINE EVALUATION
Policy: random_policy
Episodes: 1+
Mode: FROZEN (no training, no randomness)

✅ Deterministic: True
✅ Safety: PASS
✅ Numerical Stability: PASS

🎯 Your project is scientifically valid and deployable.
```

**Success criteria:**
- ✅ `is_deterministic: true`
- ✅ `is_safe: true` (or `false` if policy is actually unsafe)
- ✅ Report saved to `evaluation_reports/evaluation_report_*.json`

---

## 🧪 Test 2: Determinism Verification

The determinism hash verifies that **the same policy produces the same results** on the same episodes.

**Important**: Each run creates a new random policy (if no checkpoint provided), so hashes will differ between runs. To test determinism properly, use a **saved checkpoint**:

```bash
# First, save a policy checkpoint (or use existing one)
# Then run evaluation twice
python offline_evaluation_demo.py \
    --checkpoint checkpoints/policy.pt \
    --episodes episodes/episode_0000_*.json.gz

python offline_evaluation_demo.py \
    --checkpoint checkpoints/policy.pt \
    --episodes episodes/episode_0000_*.json.gz
```

**Compare the determinism hashes** - they MUST match:

```bash
grep "determinism_hash" evaluation_reports/*.json | tail -2
```

If hashes differ with **same checkpoint** → **hidden randomness** (CRITICAL BUG).

**Note**: Hash differences with random policy are expected (different random weights each run).

---

## 🧪 Test 3: Evaluate Trained Checkpoint

If you have a trained policy:

```bash
python offline_evaluation_demo.py \
    --checkpoint checkpoints/best_policy.pt \
    --episodes episodes/episode_*.json.gz
```

This evaluates your **actual trained policy** (not random).

---

## 🧪 Test 4: Safety Rejection

Test that unsafe policies are **rejected**:

```bash
python offline_evaluation_demo.py \
    --episodes episodes/*.json.gz \
    --max-deviation 0.1 \
    --max-intervention-rate 0.05
```

**Expected:** `❌ Safety: FAIL` (thresholds too strict)

This proves the safety validation works.

---

## 📊 Examine the Report

```bash
cat evaluation_reports/evaluation_report_*.json | jq
```

**Key fields:**
- `performance`: Mean/median reward, episode length
- `safety`: **MOST IMPORTANT** - max deviation, interventions, emergency brakes
- `comfort`: Jerk, oscillation frequency
- `is_safe`, `is_deterministic`, `has_numerical_issues`

---

## 🎯 What Makes Phase 2 "Engineering-Grade"

1. **Determinism Verification**
   - Same inputs → Same outputs → Same metrics
   - No hidden randomness
   
2. **Safety Validation**
   - Hard thresholds on worst-case behavior
   - Rejects unsafe policies
   
3. **Numerical Stability**
   - Fail-fast on NaN/Inf
   - No silent failures

4. **Scientific Rigor**
   - Reproducible evaluations
   - Formal evaluation artifacts (JSON reports)
   - Determinism hash for version tracking

---

## 🚨 Common Issues

**Issue:** `is_deterministic: false`  
**Cause:** Hidden randomness in policy or environment  
**Fix:** Ensure policy is frozen (`eval()` mode, `no_grad()`)

**Issue:** `Safety: FAIL`  
**Cause:** Policy violates safety constraints  
**Fix:** Retrain with better safety shield or tighter curriculum

**Issue:** Different determinism hashes  
**Cause:** Non-deterministic operations (e.g., `torch.backends.cudnn.deterministic = False`)  
**Fix:** Set all random seeds, disable CUDA non-determinism

---

## 📖 Usage Patterns

### Pattern 1: Development Cycle
```bash
# 1. Train policy
python train_lane_keeping.py

# 2. Record episodes
python record_and_playback.py --record --episodes 10

# 3. Offline evaluation
python offline_evaluation_demo.py \
    --checkpoint checkpoints/best.pt \
    --episodes episodes/*.json.gz

# 4. Check report → iterate
```

### Pattern 2: CI/CD Testing
```bash
# Fail build if policy is unsafe
python offline_evaluation_demo.py \
    --checkpoint $CHECKPOINT \
    --episodes $TEST_EPISODES \
    --max-deviation 1.2 || exit 1
```

### Pattern 3: Model Selection
```bash
# Evaluate multiple checkpoints
for ckpt in checkpoints/checkpoint_*.pt; do
    python offline_evaluation_demo.py \
        --checkpoint $ckpt \
        --episodes episodes/*.json.gz
done

# Pick best based on safety + performance
```

---

## 🏆 Phase 2 Complete When...

✅ Same episode replayed → **identical metrics**  
✅ Unsafe policies → **rejected by safety checks**  
✅ Evaluation runs → **no training code executed**  
✅ Reports saved → **JSON artifacts with all metrics**

**You now have a scientifically defensible autonomous system.**

---

*Phase 2 Quickstart — 2025-12-25*
