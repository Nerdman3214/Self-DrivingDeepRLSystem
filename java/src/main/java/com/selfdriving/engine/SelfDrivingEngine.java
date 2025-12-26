package com.selfdriving.engine;

/**
 * Self-Driving Inference Engine JNI Wrapper
 * 
 * Provides Java interface to the native C++ inference engine
 * for running trained RL policies.
 */
public class SelfDrivingEngine implements AutoCloseable {
    
    // Load native library
    static {
        try {
            System.loadLibrary("selfdriving_jni");
            System.out.println("[SelfDrivingEngine] Native library loaded successfully");
        } catch (UnsatisfiedLinkError e) {
            System.err.println("[SelfDrivingEngine] Failed to load native library: " + e.getMessage());
            System.err.println("Make sure libselfdriving_jni.so is in java.library.path");
            throw e;
        }
    }
    
    // Native pointer to C++ engine
    private long nativePtr;
    
    // Configuration
    private final String modelPath;
    private final boolean useGpu;
    private final int gpuDeviceId;
    
    // State
    private boolean initialized = false;
    
    /**
     * Create a new self-driving inference engine.
     * 
     * @param modelPath Path to ONNX model file
     * @param useGpu Whether to use GPU acceleration
     * @param gpuDeviceId GPU device ID (if using GPU)
     */
    public SelfDrivingEngine(String modelPath, boolean useGpu, int gpuDeviceId) {
        this.modelPath = modelPath;
        this.useGpu = useGpu;
        this.gpuDeviceId = gpuDeviceId;
        
        // Create native engine
        this.nativePtr = nativeCreate(modelPath, useGpu, gpuDeviceId);
        
        if (this.nativePtr == 0) {
            throw new RuntimeException("Failed to create native engine");
        }
    }
    
    /**
     * Create engine with default settings (CPU).
     */
    public SelfDrivingEngine(String modelPath) {
        this(modelPath, false, 0);
    }
    
    /**
     * Create engine with GPU acceleration.
     */
    public static SelfDrivingEngine withGpu(String modelPath, int deviceId) {
        return new SelfDrivingEngine(modelPath, true, deviceId);
    }
    
    /**
     * Initialize the engine and load the model.
     * 
     * @return true if initialization successful
     */
    public boolean initialize() {
        checkNotClosed();
        initialized = nativeInitialize(nativePtr);
        return initialized;
    }
    
    /**
     * Check if engine is ready for inference.
     */
    public boolean isReady() {
        return nativePtr != 0 && nativeIsReady(nativePtr);
    }
    
    /**
     * Run inference on a state observation.
     * 
     * @param state Flattened observation array (C*H*W values, normalized to [0,1])
     * @return Action array [steering, gas, brake]
     */
    public float[] infer(float[] state) {
        checkReady();
        return nativeInfer(nativePtr, state);
    }
    
    /**
     * Run inference and return a structured result.
     */
    public DrivingAction inferAction(float[] state) {
        float[] actions = infer(state);
        return new DrivingAction(actions);
    }
    
    /**
     * Run batch inference on multiple observations.
     * 
     * @param states Batch of flattened observations
     * @param batchSize Number of observations
     * @return Batched action array
     */
    public float[] inferBatch(float[] states, int batchSize) {
        checkReady();
        return nativeInferBatch(nativePtr, states, batchSize);
    }
    
    /**
     * Run inference from image data.
     * 
     * @param imageData RGB image bytes (H*W*3)
     * @param height Image height
     * @param width Image width
     * @return Action array
     */
    public float[] inferFromImage(byte[] imageData, int height, int width) {
        checkReady();
        return nativeInferFromImage(nativePtr, imageData, height, width);
    }
    
    /**
     * Get the execution provider name.
     */
    public String getExecutionProvider() {
        checkNotClosed();
        return nativeGetExecutionProvider(nativePtr);
    }
    
    /**
     * Get the last inference time in milliseconds.
     */
    public double getLastInferenceTime() {
        checkNotClosed();
        return nativeGetLastInferenceTime(nativePtr);
    }
    
    /**
     * Warm up the engine with dummy inference calls.
     */
    public void warmup(int iterations) {
        checkReady();
        nativeWarmup(nativePtr, iterations);
    }
    
    /**
     * Run benchmark and return average inference time.
     */
    public double benchmark(int iterations) {
        checkReady();
        return nativeBenchmark(nativePtr, iterations);
    }
    
    /**
     * Get model path.
     */
    public String getModelPath() {
        return modelPath;
    }
    
    /**
     * Check if using GPU.
     */
    public boolean isUsingGpu() {
        return useGpu;
    }
    
    private void checkNotClosed() {
        if (nativePtr == 0) {
            throw new IllegalStateException("Engine has been closed");
        }
    }
    
    private void checkReady() {
        checkNotClosed();
        if (!isReady()) {
            throw new IllegalStateException("Engine not ready. Call initialize() first.");
        }
    }
    
    @Override
    public void close() {
        if (nativePtr != 0) {
            nativeDestroy(nativePtr);
            nativePtr = 0;
            initialized = false;
        }
    }
    
    // Native methods
    private native long nativeCreate(String modelPath, boolean useGpu, int gpuDeviceId);
    private native void nativeDestroy(long enginePtr);
    private native boolean nativeInitialize(long enginePtr);
    private native boolean nativeIsReady(long enginePtr);
    private native float[] nativeInfer(long enginePtr, float[] state);
    private native float[] nativeInferBatch(long enginePtr, float[] states, int batchSize);
    private native float[] nativeInferFromImage(long enginePtr, byte[] imageData, int height, int width);
    private native String nativeGetExecutionProvider(long enginePtr);
    private native double nativeGetLastInferenceTime(long enginePtr);
    private native void nativeWarmup(long enginePtr, int iterations);
    private native double nativeBenchmark(long enginePtr, int iterations);
}
