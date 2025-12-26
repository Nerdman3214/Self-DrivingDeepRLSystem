package com.selfdriving.api;

import com.selfdriving.engine.DrivingAction;
import com.selfdriving.engine.SelfDrivingEngine;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import jakarta.annotation.PostConstruct;
import jakarta.annotation.PreDestroy;
import java.util.concurrent.locks.ReentrantLock;

/**
 * Self-Driving Inference Service
 * 
 * Provides thread-safe access to the inference engine.
 */
@Service
public class InferenceService {
    
    @Value("${selfdriving.model.path:models/self_driving_policy.onnx}")
    private String modelPath;
    
    @Value("${selfdriving.gpu.enabled:false}")
    private boolean gpuEnabled;
    
    @Value("${selfdriving.gpu.device:0}")
    private int gpuDevice;
    
    @Value("${selfdriving.warmup.iterations:10}")
    private int warmupIterations;
    
    private SelfDrivingEngine engine;
    private final ReentrantLock lock = new ReentrantLock();
    private boolean initialized = false;
    
    @PostConstruct
    public void initialize() {
        try {
            System.out.println("Initializing Self-Driving Inference Service...");
            System.out.println("Model path: " + modelPath);
            System.out.println("GPU enabled: " + gpuEnabled);
            
            engine = new SelfDrivingEngine(modelPath, gpuEnabled, gpuDevice);
            
            if (engine.initialize()) {
                // Warmup
                engine.warmup(warmupIterations);
                initialized = true;
                System.out.println("Inference service initialized successfully!");
                System.out.println("Execution provider: " + engine.getExecutionProvider());
            } else {
                System.err.println("Failed to initialize inference engine");
            }
        } catch (Exception e) {
            System.err.println("Error initializing inference service: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    @PreDestroy
    public void shutdown() {
        if (engine != null) {
            engine.close();
            System.out.println("Inference engine closed");
        }
    }
    
    /**
     * Check if service is ready.
     */
    public boolean isReady() {
        return initialized && engine != null && engine.isReady();
    }
    
    /**
     * Run inference on a state observation.
     * 
     * @param state Normalized observation [0,1]
     * @return Driving action
     */
    public InferenceResponse infer(float[] state) {
        if (!isReady()) {
            throw new IllegalStateException("Inference service not ready");
        }
        
        lock.lock();
        try {
            long startTime = System.nanoTime();
            
            float[] actions = engine.infer(state);
            DrivingAction action = new DrivingAction(actions);
            
            double latencyMs = (System.nanoTime() - startTime) / 1_000_000.0;
            
            return new InferenceResponse(action, latencyMs);
            
        } finally {
            lock.unlock();
        }
    }
    
    /**
     * Run inference from raw image data.
     */
    public InferenceResponse inferFromImage(byte[] imageData, int height, int width) {
        if (!isReady()) {
            throw new IllegalStateException("Inference service not ready");
        }
        
        lock.lock();
        try {
            long startTime = System.nanoTime();
            
            float[] actions = engine.inferFromImage(imageData, height, width);
            DrivingAction action = new DrivingAction(actions);
            
            double latencyMs = (System.nanoTime() - startTime) / 1_000_000.0;
            
            return new InferenceResponse(action, latencyMs);
            
        } finally {
            lock.unlock();
        }
    }
    
    /**
     * Get service statistics.
     */
    public ServiceStats getStats() {
        return new ServiceStats(
            initialized,
            engine != null ? engine.getExecutionProvider() : "N/A",
            engine != null ? engine.getLastInferenceTime() : 0.0,
            modelPath
        );
    }
    
    /**
     * Run benchmark.
     */
    public double benchmark(int iterations) {
        if (!isReady()) {
            throw new IllegalStateException("Inference service not ready");
        }
        
        lock.lock();
        try {
            return engine.benchmark(iterations);
        } finally {
            lock.unlock();
        }
    }
    
    /**
     * Inference response containing action and metadata.
     */
    public static class InferenceResponse {
        private final DrivingAction action;
        private final double latencyMs;
        
        public InferenceResponse(DrivingAction action, double latencyMs) {
            this.action = action;
            this.latencyMs = latencyMs;
        }
        
        public DrivingAction getAction() { return action; }
        public double getLatencyMs() { return latencyMs; }
    }
    
    /**
     * Service statistics.
     */
    public static class ServiceStats {
        private final boolean ready;
        private final String executionProvider;
        private final double lastInferenceTimeMs;
        private final String modelPath;
        
        public ServiceStats(boolean ready, String executionProvider, 
                          double lastInferenceTimeMs, String modelPath) {
            this.ready = ready;
            this.executionProvider = executionProvider;
            this.lastInferenceTimeMs = lastInferenceTimeMs;
            this.modelPath = modelPath;
        }
        
        public boolean isReady() { return ready; }
        public String getExecutionProvider() { return executionProvider; }
        public double getLastInferenceTimeMs() { return lastInferenceTimeMs; }
        public String getModelPath() { return modelPath; }
    }
}
