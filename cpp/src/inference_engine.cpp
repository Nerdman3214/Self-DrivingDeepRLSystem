/**
 * Self-Driving RL Inference Engine Implementation
 * 
 * Uses ONNX Runtime for high-performance policy inference.
 */

#include "inference_engine.h"

#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <numeric>

namespace selfdriving {

/**
 * Implementation class (PIMPL idiom)
 */
class InferenceEngine::Impl {
public:
    InferenceConfig config;
    
    // ONNX Runtime components
    Ort::Env env;
    Ort::SessionOptions session_options;
    std::unique_ptr<Ort::Session> session;
    Ort::AllocatorWithDefaultOptions allocator;
    
    // Model metadata
    std::vector<const char*> input_names;
    std::vector<const char*> output_names;
    std::vector<std::string> input_name_strings;
    std::vector<std::string> output_name_strings;
    std::vector<int64_t> input_shape;
    std::vector<int64_t> output_shape;
    
    // State
    bool ready = false;
    std::string execution_provider;
    
    // Reusable memory binding (for performance)
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault
    );
    
    Impl(const InferenceConfig& cfg) 
        : config(cfg)
        , env(ORT_LOGGING_LEVEL_WARNING, "SelfDrivingInference")
    {}
};

InferenceEngine::InferenceEngine(const InferenceConfig& config)
    : pImpl(std::make_unique<Impl>(config))
{
}

InferenceEngine::~InferenceEngine() = default;

InferenceEngine::InferenceEngine(InferenceEngine&&) noexcept = default;
InferenceEngine& InferenceEngine::operator=(InferenceEngine&&) noexcept = default;

bool InferenceEngine::initialize() {
    try {
        auto& impl = *pImpl;
        
        // Configure session options
        impl.session_options.SetIntraOpNumThreads(impl.config.intra_op_threads);
        impl.session_options.SetInterOpNumThreads(impl.config.inter_op_threads);
        impl.session_options.SetGraphOptimizationLevel(
            GraphOptimizationLevel::ORT_ENABLE_ALL
        );
        
        if (impl.config.enable_memory_arena) {
            impl.session_options.EnableMemPattern();
            impl.session_options.EnableCpuMemArena();
        }
        
        if (impl.config.enable_profiling) {
            impl.session_options.EnableProfiling("selfdriving_profile");
        }
        
        // Configure execution provider
        impl.execution_provider = "CPU";
        
#ifdef USE_CUDA
        if (impl.config.use_gpu) {
            OrtCUDAProviderOptions cuda_options;
            cuda_options.device_id = impl.config.gpu_device_id;
            cuda_options.arena_extend_strategy = 0;
            cuda_options.gpu_mem_limit = 0;  // No limit
            cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
            cuda_options.do_copy_in_default_stream = 1;
            
            impl.session_options.AppendExecutionProvider_CUDA(cuda_options);
            impl.execution_provider = "CUDA";
            std::cout << "[InferenceEngine] Using CUDA GPU " 
                      << impl.config.gpu_device_id << std::endl;
        }
#endif
        
        // Load model
        std::cout << "[InferenceEngine] Loading model: " 
                  << impl.config.model_path << std::endl;
        
        impl.session = std::make_unique<Ort::Session>(
            impl.env,
            impl.config.model_path.c_str(),
            impl.session_options
        );
        
        // Get input metadata
        size_t num_inputs = impl.session->GetInputCount();
        for (size_t i = 0; i < num_inputs; ++i) {
            auto name = impl.session->GetInputNameAllocated(i, impl.allocator);
            impl.input_name_strings.push_back(name.get());
            
            auto type_info = impl.session->GetInputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            impl.input_shape = tensor_info.GetShape();
        }
        
        // Store input names as const char*
        for (const auto& name : impl.input_name_strings) {
            impl.input_names.push_back(name.c_str());
        }
        
        // Get output metadata
        size_t num_outputs = impl.session->GetOutputCount();
        for (size_t i = 0; i < num_outputs; ++i) {
            auto name = impl.session->GetOutputNameAllocated(i, impl.allocator);
            impl.output_name_strings.push_back(name.get());
            
            auto type_info = impl.session->GetOutputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            impl.output_shape = tensor_info.GetShape();
        }
        
        // Store output names as const char*
        for (const auto& name : impl.output_name_strings) {
            impl.output_names.push_back(name.c_str());
        }
        
        // Print model info
        std::cout << "[InferenceEngine] Input shape: [";
        for (size_t i = 0; i < impl.input_shape.size(); ++i) {
            std::cout << impl.input_shape[i];
            if (i < impl.input_shape.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        
        std::cout << "[InferenceEngine] Output shape: [";
        for (size_t i = 0; i < impl.output_shape.size(); ++i) {
            std::cout << impl.output_shape[i];
            if (i < impl.output_shape.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        
        std::cout << "[InferenceEngine] Execution provider: " 
                  << impl.execution_provider << std::endl;
        
        impl.ready = true;
        std::cout << "[InferenceEngine] Initialization complete!" << std::endl;
        
        return true;
        
    } catch (const Ort::Exception& e) {
        std::cerr << "[InferenceEngine] ONNX Runtime error: " << e.what() << std::endl;
        return false;
    } catch (const std::exception& e) {
        std::cerr << "[InferenceEngine] Error: " << e.what() << std::endl;
        return false;
    }
}

bool InferenceEngine::isReady() const {
    return pImpl->ready;
}

InferenceResult InferenceEngine::infer(const std::vector<float>& state) {
    InferenceResult result;
    result.success = false;
    
    if (!pImpl->ready) {
        result.error_message = "Engine not initialized";
        return result;
    }
    
    auto& impl = *pImpl;
    
    try {
        auto start = std::chrono::high_resolution_clock::now();
        
        // Prepare input shape
        std::vector<int64_t> input_shape = {
            1,  // batch size
            impl.config.input_channels,
            impl.config.input_height,
            impl.config.input_width
        };
        
        size_t expected_size = 1;
        for (auto dim : input_shape) expected_size *= dim;
        
        if (state.size() != expected_size) {
            result.error_message = "Invalid input size. Expected " + 
                std::to_string(expected_size) + ", got " + 
                std::to_string(state.size());
            return result;
        }
        
        // Create input tensor
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            impl.memory_info,
            const_cast<float*>(state.data()),
            state.size(),
            input_shape.data(),
            input_shape.size()
        );
        
        // Run inference
        auto output_tensors = impl.session->Run(
            Ort::RunOptions{nullptr},
            impl.input_names.data(),
            &input_tensor,
            1,
            impl.output_names.data(),
            impl.output_names.size()
        );
        
        // Get output
        float* output_data = output_tensors[0].GetTensorMutableData<float>();
        auto output_info = output_tensors[0].GetTensorTypeAndShapeInfo();
        size_t output_size = output_info.GetElementCount();
        
        result.actions.assign(output_data, output_data + output_size);
        
        auto end = std::chrono::high_resolution_clock::now();
        result.inference_time_ms = std::chrono::duration<double, std::milli>(
            end - start
        ).count();
        
        result.success = true;
        
    } catch (const Ort::Exception& e) {
        result.error_message = std::string("ONNX Runtime error: ") + e.what();
    } catch (const std::exception& e) {
        result.error_message = std::string("Error: ") + e.what();
    }
    
    return result;
}

InferenceResult InferenceEngine::inferBatch(
    const std::vector<float>& states,
    int batch_size
) {
    InferenceResult result;
    result.success = false;
    
    if (!pImpl->ready) {
        result.error_message = "Engine not initialized";
        return result;
    }
    
    auto& impl = *pImpl;
    
    try {
        auto start = std::chrono::high_resolution_clock::now();
        
        // Prepare input shape
        std::vector<int64_t> input_shape = {
            batch_size,
            impl.config.input_channels,
            impl.config.input_height,
            impl.config.input_width
        };
        
        size_t expected_size = 1;
        for (auto dim : input_shape) expected_size *= dim;
        
        if (states.size() != expected_size) {
            result.error_message = "Invalid input size for batch";
            return result;
        }
        
        // Create input tensor
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            impl.memory_info,
            const_cast<float*>(states.data()),
            states.size(),
            input_shape.data(),
            input_shape.size()
        );
        
        // Run inference
        auto output_tensors = impl.session->Run(
            Ort::RunOptions{nullptr},
            impl.input_names.data(),
            &input_tensor,
            1,
            impl.output_names.data(),
            impl.output_names.size()
        );
        
        // Get output
        float* output_data = output_tensors[0].GetTensorMutableData<float>();
        auto output_info = output_tensors[0].GetTensorTypeAndShapeInfo();
        size_t output_size = output_info.GetElementCount();
        
        result.actions.assign(output_data, output_data + output_size);
        
        auto end = std::chrono::high_resolution_clock::now();
        result.inference_time_ms = std::chrono::duration<double, std::milli>(
            end - start
        ).count();
        
        result.success = true;
        
    } catch (const Ort::Exception& e) {
        result.error_message = std::string("ONNX Runtime error: ") + e.what();
    } catch (const std::exception& e) {
        result.error_message = std::string("Error: ") + e.what();
    }
    
    return result;
}

InferenceResult InferenceEngine::inferFromImage(
    const unsigned char* image_data,
    int height,
    int width
) {
    auto& impl = *pImpl;
    
    // Preprocess image
    // 1. Resize if needed
    std::vector<unsigned char> resized;
    const unsigned char* input_data = image_data;
    
    if (height != impl.config.input_height || width != impl.config.input_width) {
        resized = preprocess::resizeImage(
            image_data, height, width,
            impl.config.input_height, impl.config.input_width,
            3
        );
        input_data = resized.data();
        height = impl.config.input_height;
        width = impl.config.input_width;
    }
    
    // 2. Normalize to [0, 1]
    std::vector<float> normalized = preprocess::normalizeImage(
        input_data, height, width, 3
    );
    
    // 3. Convert HWC to CHW
    std::vector<float> chw = preprocess::hwcToChw(
        normalized, height, width, 3
    );
    
    // Run inference
    return infer(chw);
}

const InferenceConfig& InferenceEngine::getConfig() const {
    return pImpl->config;
}

std::string InferenceEngine::getExecutionProvider() const {
    return pImpl->execution_provider;
}

std::vector<int64_t> InferenceEngine::getInputShape() const {
    return pImpl->input_shape;
}

std::vector<int64_t> InferenceEngine::getOutputShape() const {
    return pImpl->output_shape;
}

void InferenceEngine::warmup(int iterations) {
    if (!pImpl->ready) return;
    
    auto& impl = *pImpl;
    size_t input_size = impl.config.input_channels * 
                        impl.config.input_height * 
                        impl.config.input_width;
    
    std::vector<float> dummy_input(input_size, 0.5f);
    
    std::cout << "[InferenceEngine] Warming up with " << iterations 
              << " iterations..." << std::endl;
    
    for (int i = 0; i < iterations; ++i) {
        infer(dummy_input);
    }
    
    std::cout << "[InferenceEngine] Warmup complete!" << std::endl;
}

double InferenceEngine::benchmark(int iterations) {
    if (!pImpl->ready) return -1.0;
    
    auto& impl = *pImpl;
    size_t input_size = impl.config.input_channels * 
                        impl.config.input_height * 
                        impl.config.input_width;
    
    std::vector<float> dummy_input(input_size, 0.5f);
    
    // Warmup
    warmup(100);
    
    // Benchmark
    std::cout << "[InferenceEngine] Running benchmark with " << iterations 
              << " iterations..." << std::endl;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; ++i) {
        infer(dummy_input);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double avg_ms = total_ms / iterations;
    
    std::cout << "[InferenceEngine] Benchmark results:" << std::endl;
    std::cout << "  Total time: " << total_ms << " ms" << std::endl;
    std::cout << "  Average time: " << avg_ms << " ms" << std::endl;
    std::cout << "  Throughput: " << (1000.0 / avg_ms) << " FPS" << std::endl;
    
    return avg_ms;
}

// =============================================================================
// Preprocessing utilities
// =============================================================================

namespace preprocess {

std::vector<float> normalizeImage(
    const unsigned char* data,
    int height,
    int width,
    int channels
) {
    size_t size = height * width * channels;
    std::vector<float> normalized(size);
    
    for (size_t i = 0; i < size; ++i) {
        normalized[i] = static_cast<float>(data[i]) / 255.0f;
    }
    
    return normalized;
}

std::vector<float> hwcToChw(
    const std::vector<float>& hwc_data,
    int height,
    int width,
    int channels
) {
    std::vector<float> chw_data(hwc_data.size());
    
    for (int c = 0; c < channels; ++c) {
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                int hwc_idx = h * width * channels + w * channels + c;
                int chw_idx = c * height * width + h * width + w;
                chw_data[chw_idx] = hwc_data[hwc_idx];
            }
        }
    }
    
    return chw_data;
}

std::vector<unsigned char> resizeImage(
    const unsigned char* data,
    int src_height,
    int src_width,
    int dst_height,
    int dst_width,
    int channels
) {
    std::vector<unsigned char> resized(dst_height * dst_width * channels);
    
    float scale_x = static_cast<float>(src_width) / dst_width;
    float scale_y = static_cast<float>(src_height) / dst_height;
    
    for (int y = 0; y < dst_height; ++y) {
        for (int x = 0; x < dst_width; ++x) {
            float src_x = x * scale_x;
            float src_y = y * scale_y;
            
            int x0 = static_cast<int>(src_x);
            int y0 = static_cast<int>(src_y);
            int x1 = std::min(x0 + 1, src_width - 1);
            int y1 = std::min(y0 + 1, src_height - 1);
            
            float dx = src_x - x0;
            float dy = src_y - y0;
            
            for (int c = 0; c < channels; ++c) {
                float v00 = data[y0 * src_width * channels + x0 * channels + c];
                float v01 = data[y0 * src_width * channels + x1 * channels + c];
                float v10 = data[y1 * src_width * channels + x0 * channels + c];
                float v11 = data[y1 * src_width * channels + x1 * channels + c];
                
                float value = (1 - dx) * (1 - dy) * v00 +
                              dx * (1 - dy) * v01 +
                              (1 - dx) * dy * v10 +
                              dx * dy * v11;
                
                resized[y * dst_width * channels + x * channels + c] = 
                    static_cast<unsigned char>(std::clamp(value, 0.0f, 255.0f));
            }
        }
    }
    
    return resized;
}

std::vector<unsigned char> rgbToGray(
    const unsigned char* data,
    int height,
    int width
) {
    std::vector<unsigned char> gray(height * width);
    
    for (int i = 0; i < height * width; ++i) {
        float r = data[i * 3 + 0];
        float g = data[i * 3 + 1];
        float b = data[i * 3 + 2];
        
        gray[i] = static_cast<unsigned char>(
            0.299f * r + 0.587f * g + 0.114f * b
        );
    }
    
    return gray;
}

} // namespace preprocess

} // namespace selfdriving
