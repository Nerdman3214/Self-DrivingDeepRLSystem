package com.selfdriving.api;

import com.selfdriving.engine.DrivingAction;
import com.selfdriving.api.InferenceService.InferenceResponse;
import com.selfdriving.api.InferenceService.ServiceStats;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.Base64;
import java.util.HashMap;
import java.util.Map;

/**
 * REST Controller for Self-Driving Inference API
 * 
 * Endpoints:
 * - POST /drive - Run inference on state observation
 * - POST /drive/image - Run inference on image
 * - GET /health - Health check
 * - GET /stats - Service statistics
 * - POST /benchmark - Run performance benchmark
 */
@RestController
@RequestMapping("/api/v1")
@CrossOrigin(origins = "*")
public class SelfDrivingController {
    
    @Autowired
    private InferenceService inferenceService;
    
    /**
     * Health check endpoint.
     */
    @GetMapping("/health")
    public ResponseEntity<Map<String, Object>> healthCheck() {
        Map<String, Object> response = new HashMap<>();
        
        boolean ready = inferenceService.isReady();
        response.put("status", ready ? "healthy" : "unhealthy");
        response.put("ready", ready);
        response.put("timestamp", System.currentTimeMillis());
        
        return ResponseEntity.ok(response);
    }
    
    /**
     * Get service statistics.
     */
    @GetMapping("/stats")
    public ResponseEntity<Map<String, Object>> getStats() {
        ServiceStats stats = inferenceService.getStats();
        
        Map<String, Object> response = new HashMap<>();
        response.put("ready", stats.isReady());
        response.put("executionProvider", stats.getExecutionProvider());
        response.put("lastInferenceTimeMs", stats.getLastInferenceTimeMs());
        response.put("modelPath", stats.getModelPath());
        
        return ResponseEntity.ok(response);
    }
    
    /**
     * Run inference on state observation.
     * 
     * Request body:
     * {
     *   "state": [0.1, 0.2, ...],  // Flattened, normalized observation
     * }
     * 
     * Response:
     * {
     *   "action": {
     *     "steering": 0.5,
     *     "gas": 0.8,
     *     "brake": 0.0
     *   },
     *   "latencyMs": 5.2,
     *   "description": "Turn Right + Gas"
     * }
     */
    @PostMapping("/drive")
    public ResponseEntity<?> drive(@RequestBody DriveRequest request) {
        try {
            if (!inferenceService.isReady()) {
                return ResponseEntity.status(HttpStatus.SERVICE_UNAVAILABLE)
                    .body(Map.of("error", "Inference service not ready"));
            }
            
            if (request.getState() == null || request.getState().length == 0) {
                return ResponseEntity.badRequest()
                    .body(Map.of("error", "State observation is required"));
            }
            
            InferenceResponse result = inferenceService.infer(request.getState());
            
            return ResponseEntity.ok(buildDriveResponse(result));
            
        } catch (Exception e) {
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                .body(Map.of("error", e.getMessage()));
        }
    }
    
    /**
     * Run inference on image data.
     * 
     * Request body:
     * {
     *   "imageData": "base64_encoded_rgb_data",
     *   "height": 96,
     *   "width": 96
     * }
     */
    @PostMapping("/drive/image")
    public ResponseEntity<?> driveFromImage(@RequestBody ImageDriveRequest request) {
        try {
            if (!inferenceService.isReady()) {
                return ResponseEntity.status(HttpStatus.SERVICE_UNAVAILABLE)
                    .body(Map.of("error", "Inference service not ready"));
            }
            
            if (request.getImageData() == null || request.getImageData().isEmpty()) {
                return ResponseEntity.badRequest()
                    .body(Map.of("error", "Image data is required"));
            }
            
            byte[] imageBytes = Base64.getDecoder().decode(request.getImageData());
            
            InferenceResponse result = inferenceService.inferFromImage(
                imageBytes, request.getHeight(), request.getWidth()
            );
            
            return ResponseEntity.ok(buildDriveResponse(result));
            
        } catch (Exception e) {
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                .body(Map.of("error", e.getMessage()));
        }
    }
    
    /**
     * Run inference benchmark.
     * 
     * Request body:
     * {
     *   "iterations": 1000
     * }
     */
    @PostMapping("/benchmark")
    public ResponseEntity<?> benchmark(@RequestBody(required = false) BenchmarkRequest request) {
        try {
            if (!inferenceService.isReady()) {
                return ResponseEntity.status(HttpStatus.SERVICE_UNAVAILABLE)
                    .body(Map.of("error", "Inference service not ready"));
            }
            
            int iterations = (request != null && request.getIterations() > 0) 
                ? request.getIterations() : 1000;
            
            double avgTimeMs = inferenceService.benchmark(iterations);
            
            Map<String, Object> response = new HashMap<>();
            response.put("iterations", iterations);
            response.put("averageTimeMs", avgTimeMs);
            response.put("throughputFps", 1000.0 / avgTimeMs);
            
            return ResponseEntity.ok(response);
            
        } catch (Exception e) {
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                .body(Map.of("error", e.getMessage()));
        }
    }
    
    private Map<String, Object> buildDriveResponse(InferenceResponse result) {
        DrivingAction action = result.getAction();
        
        Map<String, Object> actionMap = new HashMap<>();
        actionMap.put("steering", action.getSteering());
        actionMap.put("gas", action.getGas());
        actionMap.put("brake", action.getBrake());
        actionMap.put("raw", action.getRawActions());
        
        Map<String, Object> response = new HashMap<>();
        response.put("action", actionMap);
        response.put("latencyMs", result.getLatencyMs());
        response.put("description", action.getActionDescription());
        
        return response;
    }
    
    // Request DTOs
    
    public static class DriveRequest {
        private float[] state;
        
        public float[] getState() { return state; }
        public void setState(float[] state) { this.state = state; }
    }
    
    public static class ImageDriveRequest {
        private String imageData;  // Base64 encoded
        private int height = 96;
        private int width = 96;
        
        public String getImageData() { return imageData; }
        public void setImageData(String imageData) { this.imageData = imageData; }
        public int getHeight() { return height; }
        public void setHeight(int height) { this.height = height; }
        public int getWidth() { return width; }
        public void setWidth(int width) { this.width = width; }
    }
    
    public static class BenchmarkRequest {
        private int iterations = 1000;
        
        public int getIterations() { return iterations; }
        public void setIterations(int iterations) { this.iterations = iterations; }
    }
}
