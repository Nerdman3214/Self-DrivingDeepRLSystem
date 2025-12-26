#ifndef SELF_DRIVING_INFERENCE_ENGINE_H
#define SELF_DRIVING_INFERENCE_ENGINE_H

/**
 * STEP 4: Self-Driving RL Inference Engine with Safety Shield
 * 
 * Production-ready inference system:
 * - ONNX Runtime for policy execution
 * - Safety Shield for hard constraints
 * - NO training code (inference-only)
 * - Deterministic actions
 * - GPU acceleration (optional)
 * 
 * Design Pattern: Guarded Command
 * - Policy outputs = suggestions
 * - Safety layer = authority
 */

#include <string>
#include <vector>
#include <memory>
#include <chrono>
#include "SafetyShield.h"

namespace selfdriving {

/**
 * Configuration for the inference engine
 */
struct InferenceConfig {
    std::string model_path;
    bool use_gpu = true;
    int gpu_device_id = 0;
    int intra_op_threads = 1;
    int inter_op_threads = 1;
    bool enable_memory_arena = true;
    bool enable_profiling = false;
    
    // Input dimensions
    int batch_size = 1;
    int input_channels = 3;
    int input_height = 96;
    int input_width = 96;
    
    // Output dimensions
    int action_dim = 3;
};

/**
 * Result of inference
 */
struct InferenceResult {
    std::vector<float> actions;
    double inference_time_ms;
    bool success;
    std::string error_message;
};

/**
 * Inference Engine for Self-Driving RL Policy
 * 
 * Loads ONNX models and performs efficient inference using ONNX Runtime.
 */
class InferenceEngine {
public:
    /**
     * Create inference engine with configuration
     */
    explicit InferenceEngine(const InferenceConfig& config);
    
    /**
     * Destructor - cleans up ONNX Runtime resources
     */
    ~InferenceEngine();
    
    // Non-copyable
    InferenceEngine(const InferenceEngine&) = delete;
    InferenceEngine& operator=(const InferenceEngine&) = delete;
    
    // Movable
    InferenceEngine(InferenceEngine&&) noexcept;
    InferenceEngine& operator=(InferenceEngine&&) noexcept;
    
    /**
     * Initialize the engine and load the model
     * 
     * @return true if initialization successful
     */
    bool initialize();
    
    /**
     * Check if engine is ready for inference
     */
    bool isReady() const;
    
    /**
     * STEP 4: Run safe inference with safety shield
     * 
     * @param state 6D state vector [lane_offset, heading_error, speed, left_dist, right_dist, curvature]
     * @return Safe action [steering, throttle]
     */
    InferenceResult infer(const std::vector<float>& state);
    
    /**
     * Run inference without safety shield (for testing)
     * 
     * @param state 6D state vector
     * @return Raw policy output
     */
    InferenceResult inferRaw(const std::vector<float>& state);
    
    /**
     * Get the current configuration
     */
    const InferenceConfig& getConfig() const;
    
    /**
     * Get execution provider name (CUDA, CPU, etc.)
     */
    std::string getExecutionProvider() const;
    
    /**
     * Get model input shape
     */
    std::vector<int64_t> getInputShape() const;
    
    /**
     * Get model output shape
     */
    std::vector<int64_t> getOutputShape() const;
    
    /**
     * Warm up the engine with dummy inference
     * 
     * @param iterations Number of warmup iterations
     */
    void warmup(int iterations = 10);
    
    /**
     * Run benchmark and return statistics
     * 
     * @param iterations Number of iterations for benchmark
     * @return Average inference time in milliseconds
     */
    double benchmark(int iterations = 1000);

private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

/**
 * Utility functions for preprocessing
 */
namespace preprocess {

/**
 * Normalize image from [0,255] to [0,1]
 */
std::vector<float> normalizeImage(
    const unsigned char* data,
    int height,
    int width,
    int channels = 3
);

/**
 * Convert HWC to CHW format
 */
std::vector<float> hwcToChw(
    const std::vector<float>& hwc_data,
    int height,
    int width,
    int channels
);

/**
 * Resize image using bilinear interpolation
 */
std::vector<unsigned char> resizeImage(
    const unsigned char* data,
    int src_height,
    int src_width,
    int dst_height,
    int dst_width,
    int channels = 3
);

/**
 * Convert RGB to grayscale
 */
std::vector<unsigned char> rgbToGray(
    const unsigned char* data,
    int height,
    int width
);

} // namespace preprocess

} // namespace selfdriving

#endif // SELF_DRIVING_INFERENCE_ENGINE_H
