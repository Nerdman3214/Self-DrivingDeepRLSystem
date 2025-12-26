package com.selfdriving.controller;

import com.selfdriving.model.InferenceRequest;
import com.selfdriving.model.InferenceResponse;
import org.springframework.web.bind.annotation.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * STEP 4: REST API Controller
 * 
 * Exposes policy inference via HTTP.
 * 
 * No GUI required - microservice-ready.
 */
@RestController
@RequestMapping("/api/v1")
public class InferenceController {
    
    private static final Logger logger = LoggerFactory.getLogger(InferenceController.class);
    
    /**
     * STEP 4: Safe Inference Endpoint
     * 
     * POST /api/v1/infer
     * {
     *   "laneOffset": 0.12,
     *   "headingError": -0.03,
     *   "speed": 20.0,
     *   "leftDistance": 1.75,
     *   "rightDistance": 1.75,
     *   "curvature": 0.0
     * }
     * 
     * Response:
     * {
     *   "steering": -0.15,
     *   "throttle": 0.62,
     *   "safe": true,
     *   "inferenceTimeMs": 2.5
     * }
     */
    @PostMapping("/infer")
    public InferenceResponse infer(@RequestBody InferenceRequest request) {
        logger.info("Inference request: {}", request);
        
        long startTime = System.nanoTime();
        
        try {
            // Convert request to state vector
            float[] state = request.toStateVector();
            
            // Mock inference for now (until C++ JNI is connected)
            // TODO: Replace with actual C++ inference engine call
            double steering = -0.15 * request.getLaneOffset() - 0.1 * request.getHeadingError();
            double throttle = 0.5;
            
            // Clamp to valid ranges
            steering = Math.max(-1.0, Math.min(1.0, steering));
            throttle = Math.max(-1.0, Math.min(1.0, throttle));
            
            long endTime = System.nanoTime();
            double inferenceTimeMs = (endTime - startTime) / 1_000_000.0;
            
            InferenceResponse response = new InferenceResponse();
            response.setSteering(steering);
            response.setThrottle(throttle);
            response.setSafe(true);
            response.setInferenceTimeMs(inferenceTimeMs);
            response.setInterventions(new String[]{});
            
            logger.info("Inference response: steering={}, throttle={}, time={}ms", 
                steering, throttle, inferenceTimeMs);
            
            return response;
            
        } catch (Exception e) {
            logger.error("Inference error: ", e);
            
            InferenceResponse errorResponse = new InferenceResponse();
            errorResponse.setSteering(0.0);
            errorResponse.setThrottle(0.0);
            errorResponse.setSafe(false);
            errorResponse.setInferenceTimeMs(0.0);
            errorResponse.setInterventions(new String[]{"Error: " + e.getMessage()});
            
            return errorResponse;
        }
    }
    
    /**
     * Health check endpoint
     */
    @GetMapping("/health")
    public String health() {
        return "OK";
    }
    
    /**
     * Model info endpoint
     */
    @GetMapping("/model/info")
    public ModelInfo getModelInfo() {
        ModelInfo info = new ModelInfo();
        info.setModelType("PPO Lane-Keeping");
        info.setInputDim(6);
        info.setOutputDim(2);
        info.setHasSafetyShield(true);
        return info;
    }
    
    // Model info class
    public static class ModelInfo {
        private String modelType;
        private int inputDim;
        private int outputDim;
        private boolean hasSafetyShield;
        
        public String getModelType() { return modelType; }
        public void setModelType(String modelType) { this.modelType = modelType; }
        
        public int getInputDim() { return inputDim; }
        public void setInputDim(int inputDim) { this.inputDim = inputDim; }
        
        public int getOutputDim() { return outputDim; }
        public void setOutputDim(int outputDim) { this.outputDim = outputDim; }
        
        public boolean isHasSafetyShield() { return hasSafetyShield; }
        public void setHasSafetyShield(boolean hasSafetyShield) { 
            this.hasSafetyShield = hasSafetyShield; 
        }
    }
}
