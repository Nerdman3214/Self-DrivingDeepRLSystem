package com.selfdriving;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

/**
 * Self-Driving RL System - Spring Boot Application
 * 
 * REST API for running trained self-driving policies.
 */
@SpringBootApplication
public class SelfDrivingApplication {
    
    public static void main(String[] args) {
        System.out.println("=".repeat(60));
        System.out.println("Self-Driving RL Inference Server");
        System.out.println("=".repeat(60));
        
        SpringApplication.run(SelfDrivingApplication.class, args);
    }
}
