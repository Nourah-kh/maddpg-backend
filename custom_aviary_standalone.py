"""
custom_aviary_standalone.py — MADDPG Environment without Ray dependency
=========================================================================
Standalone version for deployment that doesn't require Ray/RLlib
"""

import numpy as np
import pybullet as p
import pybullet_data
from gymnasium import spaces


class CustomAviaryMADDPG:
    """MADDPG UAV environment - standalone version for deployment"""
    
    def __init__(self, num_drones=4, obs_radius=0.3, act_radius=0.3, num_obstacles=4, gui=False, record=False):
        self.num_drones = num_drones
        self.obs_radius = obs_radius
        self.act_radius = act_radius
        self.num_obstacles = num_obstacles
        self.gui = gui
        
        # PyBullet setup
        if gui:
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        
        # Load plane
        self.plane_id = p.loadURDF("plane.urdf")
        
        # Observation and action spaces (per drone)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )
        
        # State
        self.drone_ids = []
        self.obstacle_ids = []
        self.goal_position = None
        self.step_count = 0
        
    def reset(self, seed=None):
        """Reset environment"""
        # Remove old drones/obstacles
        for drone_id in self.drone_ids:
            p.removeBody(drone_id)
        for obs_id in self.obstacle_ids:
            p.removeBody(obs_id)
        
        self.drone_ids = []
        self.obstacle_ids = []
        self.step_count = 0
        
        # Spawn drones near origin
        for i in range(self.num_drones):
            x = np.random.uniform(-0.5, 0.5)
            y = np.random.uniform(-0.5, 0.5)
            z = 0.5
            
            # Create simple sphere as drone
            collision_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=0.1)
            visual_shape = p.createVisualShape(p.GEOM_SPHERE, radius=0.1, rgbaColor=[0, 1, 0, 1])
            
            drone_id = p.createMultiBody(
                baseMass=0.5,
                baseCollisionShapeIndex=collision_shape,
                baseVisualShapeIndex=visual_shape,
                basePosition=[x, y, z]
            )
            self.drone_ids.append(drone_id)
        
        # Spawn obstacles
        if self.num_obstacles == 2:
            obs_positions = [
                (2.5, 0.0, 0.4),
                (-2.5, 0.0, 0.4),
            ]
        elif self.num_obstacles == 3:
            obs_positions = [
                (2.5, 0.0, 0.4),
                (-2.5, 0.0, 0.4),
                (0.0, 1.5, 0.4),
            ]
        else:  # 4 obstacles
            obs_positions = [
                (2.5, 0.0, 0.4),
                (-2.5, 0.0, 0.4),
                (0.0, 1.5, 0.4),
                (1.5, 2.5, 0.4),
            ]
        
        for ox, oy, oz in obs_positions:
            collision_shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=self.obs_radius, height=0.5)
            visual_shape = p.createVisualShape(p.GEOM_CYLINDER, radius=self.obs_radius, length=0.5, rgbaColor=[1, 0, 0, 1])
            
            obs_id = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=collision_shape,
                baseVisualShapeIndex=visual_shape,
                basePosition=[ox, oy, oz]
            )
            self.obstacle_ids.append(obs_id)
        
        # Random goal
        self.goal_position = np.array([
            np.random.uniform(-2.0, 2.0),
            np.random.uniform(-2.0, 2.0),
            0.5
        ])
        
        # Create goal marker
        goal_visual = p.createVisualShape(p.GEOM_SPHERE, radius=0.2, rgbaColor=[1, 0.8, 0, 1])
        p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=goal_visual,
            basePosition=self.goal_position
        )
        
        # Get initial observations
        obs = self._get_observations()
        
        return obs, {}
    
    def _get_observations(self):
        """Get observations for all drones"""
        observations = {}
        
        for i, drone_id in enumerate(self.drone_ids):
            pos, orn = p.getBasePositionAndOrientation(drone_id)
            vel, ang_vel = p.getBaseVelocity(drone_id)
            
            # 13D observation
            obs = np.zeros(13, dtype=np.float32)
            obs[0:3] = pos  # position
            obs[3:6] = vel  # velocity
            obs[6:9] = p.getEulerFromQuaternion(orn)  # orientation
            obs[9] = 0.0  # proximity (simplified)
            obs[10:13] = self.goal_position - pos  # relative goal direction
            
            observations[f"drone_{i}"] = obs
        
        return observations
    
    def step(self, actions):
        """Step environment with actions for all drones"""
        # Apply actions
        for i, drone_id in enumerate(self.drone_ids):
            action = actions[i]
            
            # Simple velocity control
            vx, vy, vz, yaw_rate = action * 2.0  # Scale actions
            
            pos, orn = p.getBasePositionAndOrientation(drone_id)
            new_pos = [
                pos[0] + vx * 0.01,
                pos[1] + vy * 0.01,
                pos[2] + vz * 0.01
            ]
            
            p.resetBasePositionAndOrientation(drone_id, new_pos, orn)
        
        # Step simulation
        p.stepSimulation()
        self.step_count += 1
        
        # Get observations
        obs = self._get_observations()
        
        # Calculate rewards
        rewards = {}
        terminated = {}
        truncated = {}
        
        for i in range(self.num_drones):
            drone_key = f"drone_{i}"
            pos = obs[drone_key][0:3]
            
            # Distance to goal
            dist_to_goal = np.linalg.norm(pos - self.goal_position)
            
            # Reward
            reward = 0.01  # survival
            if dist_to_goal < 0.3:
                reward += 10.0  # reached goal
            
            rewards[drone_key] = reward
            terminated[drone_key] = False
            truncated[drone_key] = self.step_count >= 500
        
        return obs, rewards, terminated, truncated, {}
    
    def close(self):
        """Close PyBullet"""
        p.disconnect(self.client)
