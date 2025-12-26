#pragma once

#include <vector>
#include <string>
#include <memory>
#include <stdexcept>

namespace selfdriving {

/**
 * STEP 4: Safety Limits (Hard Constraints)
 * 
 * Non-learned rules that guarantee safe operation.
 */
struct SafetyLimits {
    float max_steering_angle = 0.5f;   // ~28 degrees
    float max_steering_rate = 0.3f;    // Prevent oscillation
    float max_speed = 30.0f;           // m/s
    float max_lane_offset = 1.5f;      // Emergency brake threshold
    float action_min = -1.0f;
    float action_max = 1.0f;
};

/**
 * STEP 4: Safety Shield
 * 
 * Validates and constrains RL policy actions.
 * 
 * Design Pattern: Guarded Command Pattern
 * - Policy outputs = suggestions
 * - Safety layer = authority  
 * - Actuators = dumb executors
 */
class SafetyShield {
public:
    SafetyShield(const SafetyLimits& limits = SafetyLimits())
        : limits_(limits), prev_steering_(0.0f) {}
    
    /**
     * Apply safety rules to policy action.
     * 
     * Priority order:
     * 1. NaN/Inf check
     * 2. Action bounds
     * 3. Steering rate limiting
     * 4. Emergency brake
     * 5. Steering angle limiting
     * 
     * @param action [steering, throttle] from policy
     * @param state [lane_offset, heading_error, speed, ...]
     * @return Safe action, guaranteed within limits
     */
    std::vector<float> checkAndFix(
        const std::vector<float>& action,
        const std::vector<float>& state);
    
    void reset();
    bool isSafe(const std::vector<float>& state) const;
    
private:
    SafetyLimits limits_;
    float prev_steering_;
    
    bool isNanOrInf(float value) const;
};

/**
 * STEP 4: Inference Engine with Safety
 * 
 * Loads ONNX model and applies safety shield.
 * 
 * NO training code:
 * - No gradients
 * - No replay buffer
 * - No reward calculation
 * - Deterministic only
 */
class InferenceEngine {
public:
    InferenceEngine(const std::string& model_path);
    ~InferenceEngine();
    
    /**
     * Safe inference with safety shield.
     * 
     * @param state 6D state vector [lane_offset, heading_error, speed, left_dist, right_dist, curvature]
     * @return Safe action [steering, throttle]
     */
    std::vector<float> infer(const std::vector<float>& state);
    
    /**
     * Get raw policy output (before safety shield).
     * For debugging only.
     */
    std::vector<float> inferRaw(const std::vector<float>& state);
    
    void reset();
    
private:
    class Impl;  // PIMPL pattern
    std::unique_ptr<Impl> pimpl_;
    std::unique_ptr<SafetyShield> safety_shield_;
};

} // namespace selfdriving
