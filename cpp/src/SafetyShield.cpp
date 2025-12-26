#include "SafetyShield.h"
#include <cmath>
#include <algorithm>
#include <iostream>

namespace selfdriving {

bool SafetyShield::isNanOrInf(float value) const {
    return std::isnan(value) || std::isinf(value);
}

std::vector<float> SafetyShield::checkAndFix(
    const std::vector<float>& action,
    const std::vector<float>& state) {
    
    std::vector<float> safe_action = action;
    
    if (safe_action.size() != 2) {
        throw std::runtime_error("Action must be 2D [steering, throttle]");
    }
    
    if (state.size() < 3) {
        throw std::runtime_error("State must have at least 3 elements");
    }
    
    float lane_offset = state[0];
    float speed = state[2];
    
    // === RULE 1: NaN/Inf Check (CRITICAL) ===
    if (isNanOrInf(safe_action[0]) || isNanOrInf(safe_action[1])) {
        std::cerr << "⚠️  SAFETY: NaN/Inf detected! Emergency stop." << std::endl;
        safe_action[0] = 0.0f;   // Straight
        safe_action[1] = -1.0f;  // Full brake
        return safe_action;
    }
    
    // === RULE 2: Action Bounds ===
    safe_action[0] = std::clamp(safe_action[0], limits_.action_min, limits_.action_max);
    safe_action[1] = std::clamp(safe_action[1], limits_.action_min, limits_.action_max);
    
    // === RULE 3: Steering Rate Limiting ===
    float steering_change = std::abs(safe_action[0] - prev_steering_);
    if (steering_change > limits_.max_steering_rate) {
        float direction = (safe_action[0] > prev_steering_) ? 1.0f : -1.0f;
        safe_action[0] = prev_steering_ + direction * limits_.max_steering_rate;
    }
    
    // === RULE 4: Emergency Brake (Lane Loss) ===
    if (std::abs(lane_offset) > limits_.max_lane_offset) {
        std::cerr << "🚨 EMERGENCY BRAKE: Lane offset " << lane_offset 
                  << "m exceeds " << limits_.max_lane_offset << "m" << std::endl;
        safe_action[0] = 0.0f;   // Straighten
        safe_action[1] = -1.0f;  // Full brake
    }
    
    // === RULE 5: Steering Angle Limiting ===
    if (std::abs(safe_action[0]) > limits_.max_steering_angle) {
        safe_action[0] = (safe_action[0] > 0) ? limits_.max_steering_angle : -limits_.max_steering_angle;
    }
    
    // === RULE 6: Speed Limiting ===
    if (speed > limits_.max_speed) {
        safe_action[1] = std::min(safe_action[1], -0.5f);  // Force deceleration
    }
    
    // Update state
    prev_steering_ = safe_action[0];
    
    return safe_action;
}

void SafetyShield::reset() {
    prev_steering_ = 0.0f;
}

bool SafetyShield::isSafe(const std::vector<float>& state) const {
    if (state.size() < 3) return false;
    
    float lane_offset = state[0];
    float speed = state[2];
    
    if (std::abs(lane_offset) > limits_.max_lane_offset) return false;
    if (speed > limits_.max_speed) return false;
    
    return true;
}

} // namespace selfdriving
