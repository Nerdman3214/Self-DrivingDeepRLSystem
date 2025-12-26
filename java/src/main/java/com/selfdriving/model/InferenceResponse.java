package com.selfdriving.model;

/**
 * STEP 4: Inference Response
 * 
 * Safe action from policy + safety shield.
 */
public class InferenceResponse {
    
    private double steering;    // Steering angle [-1, 1]
    private double throttle;    // Throttle/brake [-1, 1]
    private boolean safe;       // Safety status
    private double inferenceTimeMs;
    
    // Safety interventions
    private String[] interventions;
    
    // Constructors
    public InferenceResponse() {}
    
    public InferenceResponse(
        double steering,
        double throttle,
        boolean safe,
        double inferenceTimeMs
    ) {
        this.steering = steering;
        this.throttle = throttle;
        this.safe = safe;
        this.inferenceTimeMs = inferenceTimeMs;
        this.interventions = new String[0];
    }
    
    // Getters and Setters
    public double getSteering() {
        return steering;
    }
    
    public void setSteering(double steering) {
        this.steering = steering;
    }
    
    public double getThrottle() {
        return throttle;
    }
    
    public void setThrottle(double throttle) {
        this.throttle = throttle;
    }
    
    public boolean isSafe() {
        return safe;
    }
    
    public void setSafe(boolean safe) {
        this.safe = safe;
    }
    
    public double getInferenceTimeMs() {
        return inferenceTimeMs;
    }
    
    public void setInferenceTimeMs(double inferenceTimeMs) {
        this.inferenceTimeMs = inferenceTimeMs;
    }
    
    public String[] getInterventions() {
        return interventions;
    }
    
    public void setInterventions(String[] interventions) {
        this.interventions = interventions;
    }
    
    @Override
    public String toString() {
        return String.format(
            "Action[steering=%.3f, throttle=%.3f, safe=%s, time=%.2fms]",
            steering, throttle, safe, inferenceTimeMs
        );
    }
}
