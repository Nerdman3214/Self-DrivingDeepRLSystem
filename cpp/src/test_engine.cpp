/**
 * Test program for Self-Driving Inference Engine
 * 
 * Usage:
 *   ./test_engine <model.onnx> [--gpu]
 */

#include "inference_engine.h"
#include <iostream>
#include <random>
#include <cstring>

using namespace selfdriving;

void printUsage(const char* program) {
    std::cout << "Usage: " << program << " <model.onnx> [--gpu] [--benchmark]" << std::endl;
    std::cout << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --gpu        Use GPU acceleration (if available)" << std::endl;
    std::cout << "  --benchmark  Run inference benchmark" << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printUsage(argv[0]);
        return 1;
    }
    
    std::string model_path = argv[1];
    bool use_gpu = false;
    bool run_benchmark = false;
    
    for (int i = 2; i < argc; ++i) {
        if (strcmp(argv[i], "--gpu") == 0) {
            use_gpu = true;
        } else if (strcmp(argv[i], "--benchmark") == 0) {
            run_benchmark = true;
        }
    }
    
    std::cout << "================================================" << std::endl;
    std::cout << "Self-Driving Inference Engine Test" << std::endl;
    std::cout << "================================================" << std::endl;
    std::cout << "Model: " << model_path << std::endl;
    std::cout << "Use GPU: " << (use_gpu ? "Yes" : "No") << std::endl;
    std::cout << std::endl;
    
    // Configure engine
    InferenceConfig config;
    config.model_path = model_path;
    config.use_gpu = use_gpu;
    config.input_channels = 3;
    config.input_height = 96;
    config.input_width = 96;
    config.action_dim = 3;
    
    // Create and initialize engine
    InferenceEngine engine(config);
    
    std::cout << "Initializing engine..." << std::endl;
    if (!engine.initialize()) {
        std::cerr << "Failed to initialize engine!" << std::endl;
        return 1;
    }
    
    std::cout << "Engine ready: " << engine.isReady() << std::endl;
    std::cout << "Execution provider: " << engine.getExecutionProvider() << std::endl;
    std::cout << std::endl;
    
    // Generate random input
    std::cout << "Generating random input..." << std::endl;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    
    size_t input_size = config.input_channels * config.input_height * config.input_width;
    std::vector<float> input(input_size);
    for (auto& val : input) {
        val = dist(gen);
    }
    
    std::cout << "Input size: " << input_size << " floats" << std::endl;
    std::cout << std::endl;
    
    // Run inference
    std::cout << "Running inference..." << std::endl;
    InferenceResult result = engine.infer(input);
    
    if (result.success) {
        std::cout << "Inference successful!" << std::endl;
        std::cout << "Inference time: " << result.inference_time_ms << " ms" << std::endl;
        std::cout << "Actions (" << result.actions.size() << "): [";
        for (size_t i = 0; i < result.actions.size(); ++i) {
            std::cout << result.actions[i];
            if (i < result.actions.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        
        // Interpret actions
        if (result.actions.size() >= 3) {
            std::cout << std::endl;
            std::cout << "Interpreted actions:" << std::endl;
            std::cout << "  Steering: " << result.actions[0] << " (left=-1, right=+1)" << std::endl;
            std::cout << "  Gas: " << result.actions[1] << " (0-1)" << std::endl;
            std::cout << "  Brake: " << result.actions[2] << " (0-1)" << std::endl;
        }
    } else {
        std::cerr << "Inference failed: " << result.error_message << std::endl;
        return 1;
    }
    
    // Run benchmark if requested
    if (run_benchmark) {
        std::cout << std::endl;
        std::cout << "================================================" << std::endl;
        std::cout << "Running benchmark..." << std::endl;
        std::cout << "================================================" << std::endl;
        double avg_time = engine.benchmark(1000);
        std::cout << "Average inference time: " << avg_time << " ms" << std::endl;
    }
    
    std::cout << std::endl;
    std::cout << "Test complete!" << std::endl;
    
    return 0;
}
