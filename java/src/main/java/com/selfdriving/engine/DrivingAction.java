package com.selfdriving.engine;

/**
 * Represents a driving action from the RL policy.
 * 
 * Actions for CarRacing environment:
 * - Steering: [-1, 1] where -1=full left, 1=full right
 * - Gas: [0, 1] where 0=no gas, 1=full throttle
 * - Brake: [0, 1] where 0=no brake, 1=full brake
 */
public class DrivingAction {
    
    private final float steering;
    private final float gas;
    private final float brake;
    private final float[] rawActions;
    
    /**
     * Create from raw action array.
     */
    public DrivingAction(float[] actions) {
        if (actions == null || actions.length < 3) {
            throw new IllegalArgumentException("Actions must have at least 3 elements");
        }
        this.rawActions = actions.clone();
        this.steering = actions[0];
        this.gas = actions[1];
        this.brake = actions[2];
    }
    
    /**
     * Create from individual values.
     */
    public DrivingAction(float steering, float gas, float brake) {
        this.steering = clamp(steering, -1.0f, 1.0f);
        this.gas = clamp(gas, 0.0f, 1.0f);
        this.brake = clamp(brake, 0.0f, 1.0f);
        this.rawActions = new float[]{this.steering, this.gas, this.brake};
    }
    
    /**
     * Get steering value.
     * @return Steering in range [-1, 1]
     */
    public float getSteering() {
        return steering;
    }
    
    /**
     * Get gas/throttle value.
     * @return Gas in range [0, 1]
     */
    public float getGas() {
        return gas;
    }
    
    /**
     * Get brake value.
     * @return Brake in range [0, 1]
     */
    public float getBrake() {
        return brake;
    }
    
    /**
     * Get raw action array.
     */
    public float[] getRawActions() {
        return rawActions.clone();
    }
    
    /**
     * Get steering angle in degrees.
     * @param maxAngle Maximum steering angle
     * @return Steering angle in degrees
     */
    public float getSteeringAngleDegrees(float maxAngle) {
        return steering * maxAngle;
    }
    
    /**
     * Check if accelerating (gas > brake).
     */
    public boolean isAccelerating() {
        return gas > brake;
    }
    
    /**
     * Check if braking (brake > gas).
     */
    public boolean isBraking() {
        return brake > gas;
    }
    
    /**
     * Check if turning left.
     */
    public boolean isTurningLeft() {
        return steering < -0.1f;
    }
    
    /**
     * Check if turning right.
     */
    public boolean isTurningRight() {
        return steering > 0.1f;
    }
    
    /**
     * Get action as string.
     */
    public String getActionDescription() {
        StringBuilder sb = new StringBuilder();
        
        // Direction
        if (isTurningLeft()) {
            sb.append("Turn Left (").append(String.format("%.1f", Math.abs(steering) * 100)).append("%) ");
        } else if (isTurningRight()) {
            sb.append("Turn Right (").append(String.format("%.1f", steering * 100)).append("%) ");
        } else {
            sb.append("Straight ");
        }
        
        // Acceleration
        if (isAccelerating()) {
            sb.append("+ Gas (").append(String.format("%.1f", gas * 100)).append("%)");
        } else if (isBraking()) {
            sb.append("+ Brake (").append(String.format("%.1f", brake * 100)).append("%)");
        } else {
            sb.append("+ Coast");
        }
        
        return sb.toString();
    }
    
    private static float clamp(float value, float min, float max) {
        return Math.max(min, Math.min(max, value));
    }
    
    @Override
    public String toString() {
        return String.format("DrivingAction[steering=%.3f, gas=%.3f, brake=%.3f]", 
                           steering, gas, brake);
    }
}
