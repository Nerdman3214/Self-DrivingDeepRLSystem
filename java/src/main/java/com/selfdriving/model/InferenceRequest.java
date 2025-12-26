package com.selfdriving.model;

/**
 * STEP 4: Inference Request
 * 
 * State vector for lane-keeping agent.
 */
public class InferenceRequest {
    
    // 6D state vector
    private double laneOffset;      // Distance from lane center (m)
    private double headingError;    // Heading misalignment (rad)
    private double speed;           // Current speed (m/s)
    private double leftDistance;    // Distance to left boundary (m)
    private double rightDistance;   // Distance to right boundary (m)
    private double curvature;       // Road curvature (1/m)
    
    // Constructors
    public InferenceRequest() {}
    
    public InferenceRequest(
        double laneOffset,
        double headingError,
        double speed,
        double leftDistance,
        double rightDistance,
        double curvature
    ) {
        this.laneOffset = laneOffset;
        this.headingError = headingError;
        this.speed = speed;
        this.leftDistance = leftDistance;
        this.rightDistance = rightDistance;
        this.curvature = curvature;
    }
    
    // Convert to float array for C++ inference
    public float[] toStateVector() {
        return new float[] {
            (float) laneOffset,
            (float) headingError,
            (float) speed,
            (float) leftDistance,
            (float) rightDistance,
            (float) curvature
        };
    }
    
    // Getters and Setters
    public double getLaneOffset() {
        return laneOffset;
    }
    
    public void setLaneOffset(double laneOffset) {
        this.laneOffset = laneOffset;
    }
    
    public double getHeadingError() {
        return headingError;
    }
    
    public void setHeadingError(double headingError) {
        this.headingError = headingError;
    }
    
    public double getSpeed() {
        return speed;
    }
    
    public void setSpeed(double speed) {
        this.speed = speed;
    }
    
    public double getLeftDistance() {
        return leftDistance;
    }
    
    public void setLeftDistance(double leftDistance) {
        this.leftDistance = leftDistance;
    }
    
    public double getRightDistance() {
        return rightDistance;
    }
    
    public void setRightDistance(double rightDistance) {
        this.rightDistance = rightDistance;
    }
    
    public double getCurvature() {
        return curvature;
    }
    
    public void setCurvature(double curvature) {
        this.curvature = curvature;
    }
    
    @Override
    public String toString() {
        return String.format(
            "State[offset=%.3f, heading=%.3f, speed=%.1f, left=%.2f, right=%.2f, curve=%.4f]",
            laneOffset, headingError, speed, leftDistance, rightDistance, curvature
        );
    }
}
