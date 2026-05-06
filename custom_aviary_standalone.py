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
        self.goal_id = None  # Track goal body for cleanup
        self.goal_position = None
        self.step_count = 0
        self.crashed = [False] * num_drones  # collision flags per drone
        self.goal_reached = False
        
    def reset(self, seed=None):
        """Reset environment"""
        # Remove old drones/obstacles/goal
        for drone_id in self.drone_ids:
            p.removeBody(drone_id)
        for obs_id in self.obstacle_ids:
            p.removeBody(obs_id)
        if self.goal_id is not None:
            p.removeBody(self.goal_id)
        
        self.drone_ids = []
        self.obstacle_ids = []
        self.goal_id = None
        self.step_count = 0
        self.crashed = [False] * self.num_drones
        self.goal_reached = False
        
        # Spawn drones clustered near origin
        for i in range(self.num_drones):
            x = np.random.uniform(-0.3, 0.3)
            y = np.random.uniform(-0.3, 0.3)
            z = 0.5
            
            collision_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=0.1)
            visual_shape = p.createVisualShape(p.GEOM_SPHERE, radius=0.1, rgbaColor=[0, 1, 0, 1])
            
            drone_id = p.createMultiBody(
                baseMass=0.5,
                baseCollisionShapeIndex=collision_shape,
                baseVisualShapeIndex=visual_shape,
                basePosition=[x, y, z]
            )
            self.drone_ids.append(drone_id)
        
        # Spawn obstacles at fixed positions
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
        
        # Place goal in a safe zone: fixed candidate positions away from obstacles and origin
        safe_goal_candidates = [
            np.array([ 0.0,  2.5, 0.5]),
            np.array([ 0.0, -2.5, 0.5]),
            np.array([ 2.0,  2.0, 0.5]),
            np.array([-2.0,  2.0, 0.5]),
            np.array([-2.0, -2.0, 0.5]),
            np.array([ 2.0, -2.0, 0.5]),
        ]
        obs_pos_array = [np.array([ox, oy, oz]) for ox, oy, oz in obs_positions]
        
        # Pick a goal candidate that is at least 1.0m from every obstacle
        np.random.shuffle(safe_goal_candidates)
        self.goal_position = safe_goal_candidates[0]  # fallback
        for candidate in safe_goal_candidates:
            too_close = any(
                np.linalg.norm(candidate[:2] - op[:2]) < 1.0
                for op in obs_pos_array
            )
            if not too_close:
                self.goal_position = candidate
                break
        
        # Create goal marker
        goal_visual = p.createVisualShape(p.GEOM_SPHERE, radius=0.2, rgbaColor=[1, 0.8, 0, 1])
        self.goal_id = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=goal_visual,
            basePosition=self.goal_position
        )
        
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
            
            # Proximity: distance to nearest obstacle
            min_dist = float('inf')
            for obs_id in self.obstacle_ids:
                obs_pos, _ = p.getBasePositionAndOrientation(obs_id)
                d = np.linalg.norm(np.array(pos) - np.array(obs_pos))
                min_dist = min(min_dist, d)
            obs[9] = min_dist if min_dist < float('inf') else 0.0
            
            obs[10:13] = self.goal_position - np.array(pos)  # relative goal direction
            
            observations[f"drone_{i}"] = obs
        
        return observations
    
    def step(self, actions):
        """Step environment with actions for all drones.
        
        actions: dict {"drone_0": array(4,), ...} OR list/array indexed by int
        """
        self.crashed = [False] * self.num_drones

        # Apply actions
        for i, drone_id in enumerate(self.drone_ids):
            drone_key = f"drone_{i}"
            if isinstance(actions, dict):
                action = actions.get(drone_key, actions.get(i, np.zeros(4)))
            else:
                action = actions[i]
            
            # Scale: action in [-1,1], move up to 0.05m per step (was 0.02 — too slow)
            vx, vy, vz, _ = action * 0.05
            
            pos, orn = p.getBasePositionAndOrientation(drone_id)
            new_pos = [
                np.clip(pos[0] + vx, -3.5, 3.5),
                np.clip(pos[1] + vy, -3.5, 3.5),
                max(0.2, pos[2] + vz),
            ]
            p.resetBasePositionAndOrientation(drone_id, new_pos, orn)
        
        p.stepSimulation()
        self.step_count += 1
        
        # Check collisions
        for i, drone_id in enumerate(self.drone_ids):
            for obs_id in self.obstacle_ids:
                if p.getContactPoints(drone_id, obs_id):
                    self.crashed[i] = True
        
        obs = self._get_observations()
        
        rewards = {}
        terminated = {}
        truncated = {}
        
        # Episode ends when the MAJORITY of drones reach the goal (or all crash)
        drones_at_goal = 0
        for i in range(self.num_drones):
            drone_key = f"drone_{i}"
            pos = obs[drone_key][0:3]
            dist_to_goal = np.linalg.norm(pos - self.goal_position)
            
            reward = -0.01  # small penalty per step to encourage speed
            if dist_to_goal < 0.5:
                reward += 10.0
                drones_at_goal += 1
            elif dist_to_goal < 1.5:
                reward += 1.0  # shaped reward: closer = better
            
            if self.crashed[i]:
                reward -= 5.0
            
            rewards[drone_key] = reward
            terminated[drone_key] = False
            truncated[drone_key] = self.step_count >= 400
        
        # Episode terminates when majority (>=2 of 4) reach the goal
        self.goal_reached = drones_at_goal >= max(1, self.num_drones // 2)
        if self.goal_reached:
            for key in terminated:
                terminated[key] = True
        
        return obs, rewards, terminated, truncated, {"goal_reached": self.goal_reached}
    
    def close(self):
        """Close PyBullet"""
        p.disconnect(self.client)
