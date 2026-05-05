"""
custom_aviary_standalone.py — MADDPG Environment without Ray dependency
=========================================================================
Standalone version for deployment that EXACTLY matches training environment:
  - cf2x.urdf drones (not spheres)
  - Normalized observations (pos/5.0, vel/MAX_SPEED, rpy/pi)
  - 5 physics steps per control (exact match to training)
  - resetBaseVelocity for actions (exact match)
  - ACTION_SMOOTHING=0.5 (exact match)
  - Real contact-point crash detection
  - FLEXIBLE: dynamically adapt num_drones & num_obstacles per reset() call
"""

import numpy as np
from importlib.resources import files

import os


class CustomAviaryMADDPG:
    """MADDPG UAV environment - exact match to training environment"""
    
    def __init__(self, num_drones=4, num_obstacles=4, gui=False, **kwargs):
        """Initialize environment with configurable drone/obstacle counts"""
        self.num_drones = num_drones
        self.num_obstacles = num_obstacles
        self.gui = gui
        
        # ══════════════════════════════════════════════════════════════
        # Physics parameters (EXACT MATCH to training)
        # ══════════════════════════════════════════════════════════════
        self.PYB_FREQ = 240
        self.CTRL_FREQ = 48
        self.PYB_STEPS_PER_CTRL = self.PYB_FREQ // self.CTRL_FREQ  # 5 steps
        self.CTRL_TIMESTEP = 1.0 / self.CTRL_FREQ
        self.GRAVITY = 9.8
        
        # Safety limits (EXACT MATCH to training)
        self.MAX_SPEED = 1.0
        self.MAX_YAW_RATE = 1.0
        self.MAX_BOUND_XY = 5.0
        self.MAX_HEIGHT = 3.0
        self.MIN_HEIGHT = 0.2
        
        # Action smoothing (EXACT MATCH to training)
        self.MAX_ACCEL = 2.0
        self.ACTION_SMOOTHING = 0.5
        
        # Goal settings (EXACT MATCH to training)
        self.GOAL_XY_RANGE = 3.0
        self.GOAL_Z_RANGE = [0.8, 1.5]
        self.GOAL_RADIUS = 3.0
        
        # PyBullet setup
        if gui:
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -self.GRAVITY)
        p.setPhysicsEngineParameter(fixedTimeStep=1.0/self.PYB_FREQ, numSubSteps=1)
        
        # Load plane (once, never removed)
        self.plane_id = p.loadURDF("plane.urdf")
        
        # Observation and action spaces (per drone)
        self.observation_space = spaces.Box(
            low=np.array([-1.]*13), high=np.array([1.]*13), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=np.array([-self.MAX_SPEED]*3 + [-self.MAX_YAW_RATE]),
            high=np.array([self.MAX_SPEED]*3 + [self.MAX_YAW_RATE]),
            dtype=np.float32
        )
        
        # State (will be reinitialized in reset())
        self.drone_ids = []
        self.obstacle_ids = []
        self.goal_position = None
        self.step_count = 0
        self.crashed = np.zeros(num_drones, dtype=bool)
        self.crash_type = [None] * num_drones
        self.prev_cmd = np.zeros((num_drones, 4), dtype=np.float32)
        self._prev_dist = {f"drone_{i}": float("inf") for i in range(num_drones)}
        
    def reset(self, seed=None, num_drones=None, num_obstacles=None):
        """
        Reset environment.
        
        FLEXIBLE: Can change num_drones and num_obstacles per reset call!
        Example:
            env.reset(num_drones=2, num_obstacles=3)  # Changes config
            env.reset(num_drones=4, num_obstacles=2)  # Different config next time
        """
        
        # ═══ ALLOW DYNAMIC RECONFIGURATION ═══
        if num_drones is not None:
            self.num_drones = num_drones
        if num_obstacles is not None:
            self.num_obstacles = num_obstacles
        
        # ═══ COMPLETE CLEANUP OF OLD BODIES ═══
        # Remove drones
        for drone_id in self.drone_ids:
            try:
                p.removeBody(drone_id, physicsClientId=self.client)
            except:
                pass
        
        # Remove obstacles
        for obs_id in self.obstacle_ids:
            try:
                p.removeBody(obs_id, physicsClientId=self.client)
            except:
                pass
        
        # ═══ REINITIALIZE STATE FOR NEW CONFIGURATION ═══
        self.drone_ids = []
        self.obstacle_ids = []
        self.step_count = 0
        self.crashed = np.zeros(self.num_drones, dtype=bool)
        self.crash_type = [None] * self.num_drones
        self.prev_cmd = np.zeros((self.num_drones, 4), dtype=np.float32)
        self._prev_dist = {f"drone_{i}": float("inf") for i in range(self.num_drones)}
        
        # ══════════════════════════════════════════════════════════════
        # Load cf2x.urdf drones (EXACT MATCH to training)
        # ══════════════════════════════════════════════════════════════
        drone_path = os.path.join(os.path.dirname(__file__), "cf2x.urdf")
        for i in range(self.num_drones):
            angle = 2 * np.pi * i / self.num_drones
            drone_id = p.loadURDF(
                drone_path,
                [np.cos(angle), np.sin(angle), 1.0],
                p.getQuaternionFromEuler([0, 0, 0]),
                physicsClientId=self.client,
            )
            self.drone_ids.append(drone_id)
        
        # ══════════════════════════════════════════════════════════════
        # Spawn obstacles (EXACT MATCH to training positions)
        # ══════════════════════════════════════════════════════════════
        all_obs_positions = [
            [2.5, 0., .4],
            [-2.5, 0., .4],
            [0., 1.5, .4],
            [1.5, 2.5, .4],
        ]
        
        assets = pybullet_data.getDataPath()
        # Only spawn the number requested
        for pos in all_obs_positions[:self.num_obstacles]:
            cube = p.loadURDF(
                os.path.join(assets, "cube_no_rotation.urdf"),
                pos,
                globalScaling=0.8,
                physicsClientId=self.client,
            )
            p.changeDynamics(cube, -1, mass=0.0, physicsClientId=self.client)
            self.obstacle_ids.append(cube)
        
        # ══════════════════════════════════════════════════════════════
        # Random goal (EXACT MATCH to training)
        # ══════════════════════════════════════════════════════════════
        self.goal_position = np.array([
            np.random.uniform(-self.GOAL_XY_RANGE, self.GOAL_XY_RANGE),
            np.random.uniform(-self.GOAL_XY_RANGE, self.GOAL_XY_RANGE),
            np.random.uniform(*self.GOAL_Z_RANGE),
        ], dtype=np.float32)
        
        # Initialize prev_dist for current drone count
        for i in range(self.num_drones):
            pos, _ = p.getBasePositionAndOrientation(self.drone_ids[i], physicsClientId=self.client)
            self._prev_dist[f"drone_{i}"] = float(np.linalg.norm(np.array(pos) - self.goal_position))
        
        # Get initial observations
        obs = self._get_observations()
        
        return obs, {}
    
    def _get_observations(self):
        """Get normalized observations for all drones (EXACT MATCH to training)"""
        observations = {}
        PROX_RANGE = 5.0
        
        for i in range(self.num_drones):  # ✅ Use self.num_drones, not len(self.drone_ids)
            drone_id = self.drone_ids[i]
            pos, quat = p.getBasePositionAndOrientation(drone_id, physicsClientId=self.client)
            vel, ang_vel = p.getBaseVelocity(drone_id, physicsClientId=self.client)
            rpy = p.getEulerFromQuaternion(quat)
            
            # Calculate proximity to obstacles and other drones
            min_dist = float("inf")
            
            # Distance to obstacles
            for obs_id in self.obstacle_ids:  # ✅ Dynamically uses current obstacles
                obs_pos, _ = p.getBasePositionAndOrientation(obs_id, physicsClientId=self.client)
                d = np.linalg.norm(np.array(pos) - np.array(obs_pos))
                min_dist = min(min_dist, d)
            
            # Distance to other drones
            for j in range(self.num_drones):  # ✅ Use self.num_drones
                if j != i:
                    other_drone_id = self.drone_ids[j]
                    other_pos, _ = p.getBasePositionAndOrientation(other_drone_id, physicsClientId=self.client)
                    d = np.linalg.norm(np.array(pos) - np.array(other_pos))
                    min_dist = min(min_dist, d)
            
            proximity = float(np.clip(min_dist / PROX_RANGE, 0.0, 1.0))
            rel_goal = np.clip((self.goal_position - np.array(pos)) / 5.0, -1., 1.)
            
            # ══════════════════════════════════════════════════════════
            # NORMALIZED observation (EXACT MATCH to training)
            # ══════════════════════════════════════════════════════════
            own_obs = np.concatenate([
                np.clip(np.array(pos) / 5.0, -1., 1.),                    # [0:3]   pos / 5.0
                np.clip(np.array(vel) / self.MAX_SPEED, -1., 1.),         # [3:6]   vel / MAX_SPEED
                np.clip(np.array(rpy) / np.pi, -1., 1.),                  # [6:9]   rpy / pi
                np.array([proximity]),                                     # [9]     proximity
                rel_goal,                                                  # [10:13] rel_goal
            ]).astype(np.float32)
            
            observations[f"drone_{i}"] = own_obs
        
        return observations
    
    def step(self, actions):
        """
        Step environment with actions.
        
        FLEXIBLE INPUT: Accepts both dict and list formats
        PHYSICS: 5 steps per control (EXACT MATCH)
        VELOCITY: resetBaseVelocity (EXACT MATCH)
        SMOOTHING: ACTION_SMOOTHING=0.5 (EXACT MATCH)
        """
        
        # ═══ CONVERT LIST → DICT IF NEEDED ═══
        if isinstance(actions, (list, tuple)):
            # ✅ Validate action count matches drone count
            if len(actions) != self.num_drones:
                raise ValueError(
                    f"Expected {self.num_drones} actions, got {len(actions)}"
                )
            actions = {f"drone_{i}": np.array(actions[i]) for i in range(len(actions))}
        
        # ═══ APPLY ACTIONS WITH SMOOTHING ═══
        for i in range(self.num_drones):  # ✅ Use self.num_drones
            if self.crashed[i]:
                continue
            
            drone_id = self.drone_ids[i]
            agent_key = f"drone_{i}"
            
            if agent_key not in actions:
                continue
            
            action = np.array(actions[agent_key], dtype=np.float32)
            action = np.clip(action, [-self.MAX_SPEED]*3 + [-self.MAX_YAW_RATE],
                                      [self.MAX_SPEED]*3 + [self.MAX_YAW_RATE])
            
            # ═══ ACTION SMOOTHING (EXACT MATCH) ═══
            prev = self.prev_cmd[i].copy()
            max_dv = self.MAX_ACCEL * self.CTRL_TIMESTEP
            
            dv = np.clip(action[:3] - prev[:3], -max_dv, max_dv)
            limited = prev.copy()
            limited[:3] = prev[:3] + dv
            limited[3] = action[3]
            
            smoothed = (self.ACTION_SMOOTHING * prev +
                       (1.0 - self.ACTION_SMOOTHING) * limited)
            smoothed = np.nan_to_num(smoothed, nan=0., posinf=1., neginf=-1.)
            
            # ═══ APPLY VELOCITY (EXACT MATCH) ═══
            p.resetBaseVelocity(
                drone_id,
                linearVelocity=smoothed[:3].tolist(),
                angularVelocity=[0., 0., float(smoothed[3])],
                physicsClientId=self.client,
            )
            self.prev_cmd[i] = smoothed
        
        # ═══ PHYSICS STEPS (EXACT MATCH: 5 steps per control) ═══
        for _ in range(self.PYB_STEPS_PER_CTRL):
            p.stepSimulation(physicsClientId=self.client)
        
        # ═══ CRASH DETECTION (REAL CONTACT POINTS) ═══
        self._check_crashes()
        
        self.step_count += 1
        
        # ═══ GET OBSERVATIONS ═══
        obs = self._get_observations()
        
        # ═══ COMPUTE REWARDS ═══
        rewards = {}
        for i in range(self.num_drones):  # ✅ Use self.num_drones
            agent_key = f"drone_{i}"
            pos, _ = p.getBasePositionAndOrientation(self.drone_ids[i], physicsClientId=self.client)
            pos = np.array(pos)
            dist = float(np.linalg.norm(pos - self.goal_position))
            
            r_crash = -50.0 if self.crashed[i] else 0.0
            r_survive = 0.0 if self.crashed[i] else 0.001
            
            if not self.crashed[i]:
                prev_d = self._prev_dist.get(agent_key, dist)
                r_shape = 4.0 * (prev_d - dist)
            else:
                r_shape = 0.0
            
            r_goal = 200.0 if dist <= self.GOAL_RADIUS else 0.0
            
            reward = float(r_crash + r_survive + r_shape + r_goal)
            rewards[agent_key] = reward
            
            self._prev_dist[agent_key] = dist
        
        # ═══ COMPUTE TERMINATION/TRUNCATION ═══
        terminated = {f"drone_{i}": False for i in range(self.num_drones)}
        truncated = {f"drone_{i}": self.step_count >= 300 for i in range(self.num_drones)}
        
        # Any crash terminates all
        if any(self.crashed[:self.num_drones]):  # ✅ Check only active drones
            terminated = {f"drone_{i}": True for i in range(self.num_drones)}
        
        # Check if all reached goal
        all_at_goal = all(
            np.linalg.norm(
                np.array(p.getBasePositionAndOrientation(
                    self.drone_ids[i], physicsClientId=self.client
                )[0]) - self.goal_position
            ) <= self.GOAL_RADIUS
            for i in range(self.num_drones)  # ✅ Check only active drones
        )
        
        if all_at_goal and not any(self.crashed[:self.num_drones]):
            terminated = {f"drone_{i}": True for i in range(self.num_drones)}
        
        # RLlib compatibility
        terminated["__all__"] = any(terminated.values())
        truncated["__all__"] = self.step_count >= 300
        
        return obs, rewards, terminated, truncated, {}
    
    def _check_crashes(self):
        """Real contact-point crash detection (EXACT MATCH to training)"""
        for i in range(self.num_drones):  # ✅ Use self.num_drones
            if self.crashed[i]:
                continue
            
            drone_id = self.drone_ids[i]
            
            # Drone-drone collisions
            for j in range(i + 1, self.num_drones):
                if p.getContactPoints(
                    drone_id, self.drone_ids[j],
                    physicsClientId=self.client,
                ):
                    self.crashed[i] = self.crashed[j] = True
                    self.crash_type[i] = "drone_collision"
                    self.crash_type[j] = "drone_collision"
            
            # Drone-obstacle collisions
            for obs_id in self.obstacle_ids:  # ✅ Dynamically uses current obstacles
                if p.getContactPoints(
                    drone_id, obs_id,
                    physicsClientId=self.client,
                ):
                    self.crashed[i] = True
                    self.crash_type[i] = "obstacle_collision"
    
    def close(self):
        """Close PyBullet"""
        if self.client >= 0:
            p.disconnect(self.client)