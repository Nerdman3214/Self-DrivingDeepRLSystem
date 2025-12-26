"""
Traffic Agents for Multi-Agent Environment

Rule-based agents using Intelligent Driver Model (IDM).
Predictable, debuggable, industry-standard.

Design Principle:
- Only EGO learns (Deep RL)
- Traffic agents are deterministic (IDM-based)
- Prevents non-stationary learning collapse
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class IDMParams:
    """Intelligent Driver Model parameters (calibrated for highway)."""
    
    # Desired speed (m/s)
    desired_speed: float = 25.0  # ~90 km/h
    
    # Time headway (seconds)
    time_headway: float = 1.5
    
    # Minimum gap (meters)
    min_gap: float = 2.0
    
    # Maximum acceleration (m/s²)
    max_accel: float = 2.0
    
    # Comfortable deceleration (m/s²)
    comfort_decel: float = 3.0
    
    # Acceleration exponent
    accel_exponent: float = 4.0


class TrafficAgent:
    """
    Rule-based traffic agent using IDM.
    
    IDM is the industry standard for realistic traffic simulation:
    - Used in SUMO, CARLA, OpenDRIVE
    - Based on real human driving behavior
    - Smooth acceleration/deceleration
    
    This is NOT learning - it's a predictable baseline.
    """
    
    def __init__(
        self,
        agent_id: int,
        initial_lane: int,
        initial_position: float,
        initial_speed: float,
        params: Optional[IDMParams] = None
    ):
        self.agent_id = agent_id
        self.lane = initial_lane
        self.position = initial_position  # Longitudinal position (meters)
        self.speed = initial_speed
        self.params = params or IDMParams()
        
        # State
        self.lateral_offset = 0.0  # Within-lane offset
        self.is_changing_lane = False
        self.lane_change_progress = 0.0
        
        # Statistics
        self.total_distance = 0.0
        self.total_time = 0.0
    
    def compute_idm_acceleration(
        self,
        lead_distance: Optional[float],
        lead_speed: Optional[float]
    ) -> float:
        """
        Compute acceleration using Intelligent Driver Model.
        
        IDM equation:
        a = a_max * [1 - (v/v_0)^δ - (s*/s)²]
        
        where:
        s* = s_0 + v*T + v*Δv / (2*sqrt(a*b))
        
        Args:
            lead_distance: Distance to vehicle ahead (None if no leader)
            lead_speed: Speed of vehicle ahead
            
        Returns:
            Acceleration (m/s²)
        """
        p = self.params
        
        # Free-flow acceleration (no leader)
        free_accel = p.max_accel * (
            1.0 - (self.speed / p.desired_speed) ** p.accel_exponent
        )
        
        if lead_distance is None or lead_distance > 100.0:
            # No leader in range - accelerate to desired speed
            return free_accel
        
        # Interaction term (car following)
        speed_diff = self.speed - (lead_speed or 0.0)
        
        # Desired gap
        desired_gap = (
            p.min_gap 
            + self.speed * p.time_headway
            + (self.speed * speed_diff) / (2.0 * np.sqrt(p.max_accel * p.comfort_decel))
        )
        
        # Avoid division by zero
        safe_distance = max(lead_distance, 0.1)
        
        # IDM acceleration
        accel = p.max_accel * (
            1.0 - (self.speed / p.desired_speed) ** p.accel_exponent
            - (desired_gap / safe_distance) ** 2
        )
        
        return accel
    
    def update(
        self,
        dt: float,
        lead_distance: Optional[float] = None,
        lead_speed: Optional[float] = None
    ):
        """
        Update agent state using IDM.
        
        Args:
            dt: Time step (seconds)
            lead_distance: Distance to leader
            lead_speed: Speed of leader
        """
        # Compute IDM acceleration
        accel = self.compute_idm_acceleration(lead_distance, lead_speed)
        
        # Clamp acceleration for safety
        p = self.params
        accel = np.clip(accel, -p.comfort_decel * 1.5, p.max_accel)
        
        # Update speed
        new_speed = self.speed + accel * dt
        new_speed = np.clip(new_speed, 0.0, p.desired_speed * 1.2)
        
        # Update position
        self.position += self.speed * dt + 0.5 * accel * dt * dt
        self.speed = new_speed
        
        # Statistics
        self.total_distance += self.speed * dt
        self.total_time += dt
        
        # Lane change logic (simple, smooth)
        if self.is_changing_lane:
            self.lane_change_progress += dt * 2.0  # 0.5s lane change
            if self.lane_change_progress >= 1.0:
                self.is_changing_lane = False
                self.lane_change_progress = 0.0
                self.lateral_offset = 0.0
            else:
                # Smooth lateral motion
                self.lateral_offset = np.sin(self.lane_change_progress * np.pi) * 3.5
    
    def get_state(self) -> dict:
        """Get agent state for observation."""
        return {
            'agent_id': self.agent_id,
            'lane': self.lane,
            'position': self.position,
            'speed': self.speed,
            'lateral_offset': self.lateral_offset,
            'is_changing_lane': self.is_changing_lane
        }


class TrafficScenario:
    """
    Manages multiple traffic agents.
    
    Scenarios:
    - Highway cruising
    - Stop-and-go traffic
    - Lane merging
    - Cut-in events
    """
    
    def __init__(self, scenario_type: str = 'highway'):
        self.scenario_type = scenario_type
        self.agents = []
        self.dt = 0.05  # 20 Hz
        
        self._initialize_scenario()
    
    def _initialize_scenario(self):
        """Initialize agents based on scenario type."""
        if self.scenario_type == 'highway':
            # Sparse highway traffic
            self.agents = [
                TrafficAgent(
                    agent_id=1,
                    initial_lane=0,
                    initial_position=50.0,  # 50m ahead
                    initial_speed=20.0
                ),
                TrafficAgent(
                    agent_id=2,
                    initial_lane=0,
                    initial_position=120.0,  # 120m ahead
                    initial_speed=25.0
                ),
            ]
        
        elif self.scenario_type == 'dense':
            # Dense traffic
            self.agents = [
                TrafficAgent(i, 0, 30.0 + i * 20.0, 15.0 + np.random.uniform(-3, 3))
                for i in range(1, 5)
            ]
        
        elif self.scenario_type == 'stop_and_go':
            # Stop-and-go traffic
            params = IDMParams(desired_speed=15.0, time_headway=1.0)
            self.agents = [
                TrafficAgent(i, 0, 20.0 + i * 15.0, 5.0, params)
                for i in range(1, 4)
            ]
    
    def step(self, ego_position: float):
        """
        Update all traffic agents.
        
        Args:
            ego_position: Ego vehicle position (for sorting)
        """
        # Sort agents by position
        self.agents.sort(key=lambda a: a.position)
        
        # Update each agent
        for i, agent in enumerate(self.agents):
            # Find leader (next agent in same lane)
            lead_distance = None
            lead_speed = None
            
            for j in range(i + 1, len(self.agents)):
                if self.agents[j].lane == agent.lane:
                    lead_distance = self.agents[j].position - agent.position
                    lead_speed = self.agents[j].speed
                    break
            
            # Update agent using IDM
            agent.update(self.dt, lead_distance, lead_speed)
    
    def get_observations_for_ego(
        self,
        ego_lane: int,
        ego_position: float
    ) -> dict:
        """
        Get traffic observations relevant to ego vehicle.
        
        Returns:
            lead_distance: Distance to vehicle ahead
            lead_speed: Speed of vehicle ahead
            left_lane_free: Can change to left lane
            right_lane_free: Can change to right lane
        """
        lead_distance = 100.0  # Default: far away
        lead_speed = 0.0
        left_lane_free = True
        right_lane_free = True
        
        # Find lead vehicle in same lane
        min_lead_dist = float('inf')
        for agent in self.agents:
            if agent.lane == ego_lane and agent.position > ego_position:
                dist = agent.position - ego_position
                if dist < min_lead_dist:
                    min_lead_dist = dist
                    lead_distance = dist
                    lead_speed = agent.speed
        
        # Check adjacent lanes
        for agent in self.agents:
            lateral_dist = abs(agent.position - ego_position)
            
            # Left lane check (lane 1)
            if ego_lane == 0 and agent.lane == 1 and lateral_dist < 20.0:
                left_lane_free = False
            
            # Right lane check (lane -1 doesn't exist in simple case)
            # For now, assume only 2 lanes
        
        return {
            'lead_distance': lead_distance,
            'lead_speed': lead_speed,
            'left_lane_free': left_lane_free,
            'right_lane_free': right_lane_free
        }
    
    def reset(self):
        """Reset scenario."""
        self.agents = []
        self._initialize_scenario()
    
    def check_collision(self, ego_position: float, ego_lane: int) -> bool:
        """Check if ego collided with any traffic agent."""
        for agent in self.agents:
            if agent.lane == ego_lane:
                distance = abs(agent.position - ego_position)
                if distance < 4.0:  # Vehicle length ~4m
                    return True
        return False
