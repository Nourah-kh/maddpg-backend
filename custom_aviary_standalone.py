"""
custom_aviary_standalone.py — MADDPG Deployment Environment
============================================================
Matches the training environment (custom_aviary_maddpg.py) exactly:
  - Observations normalized identically (pos/5, vel/MAX_SPEED, rpy/pi, proximity/5, rel_goal/5)
  - Actions applied via resetBaseVelocity (not position delta)
  - 5 physics substeps per control step (PYB_FREQ=240, CTRL_FREQ=48)
  - Action smoothing (0.5 blend with previous command)
  - Goal radius = 3.0m, all drones within it = success
  - Termination on crash, out-of-bounds, height violation, or success
"""

import os
import numpy as np
import pybullet as p
import pybullet_data
from gymnasium import spaces


class CustomAviaryMADDPG:
    """Standalone MADDPG UAV environment — matches training env exactly."""

    PYB_FREQ           = 240
    CTRL_FREQ          = 48
    PYB_STEPS_PER_CTRL = PYB_FREQ // CTRL_FREQ   # = 5
    CTRL_TIMESTEP      = 1.0 / CTRL_FREQ

    MAX_SPEED    = 1.0
    MAX_YAW_RATE = 1.0
    MAX_BOUND_XY = 5.0
    MAX_HEIGHT   = 3.0
    MIN_HEIGHT   = 0.2
    GOAL_RADIUS  = 3.0
    MAX_STEPS    = 300
    PROX_RANGE   = 5.0
    ACTION_SMOOTHING = 0.5
    MAX_ACCEL    = 2.0

    def __init__(self, num_drones=4, num_obstacles=4, gui=False, **kwargs):
        self.num_drones    = num_drones
        self.num_obstacles = num_obstacles
        self.gui           = gui

        self.client = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client)
        p.setGravity(0, 0, -9.8, physicsClientId=self.client)
        p.setTimeStep(1.0 / self.PYB_FREQ, physicsClientId=self.client)

        lo = np.array([-1.]*3 + [-1.]*3 + [-1.]*3 + [0.] + [-1.]*3, dtype=np.float32)
        hi = np.ones(13, dtype=np.float32)
        self.observation_space = spaces.Box(low=lo, high=hi, dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.array([-self.MAX_SPEED]*3 + [-self.MAX_YAW_RATE], dtype=np.float32),
            high=np.array([+self.MAX_SPEED]*3 + [+self.MAX_YAW_RATE], dtype=np.float32),
        )

        self.DRONE_IDS       = []
        self.obstacle_ids    = []
        self.goal_id         = None
        self.goal_pos        = np.zeros(3, dtype=np.float32)
        self.step_counter    = 0
        self.crashed         = np.zeros(num_drones, dtype=bool)
        self.crash_type      = [None] * num_drones
        self.prev_cmd        = np.zeros((num_drones, 4), dtype=np.float32)
        self.is_success      = False
        self.is_collision    = False
        self.mission_time    = 0
        self._prev_dist      = {}
        self._last_valid_obs = {}
        self.goal_reached    = False

    # ──────────────────────────────────────────────────────────────
    # reset
    # ──────────────────────────────────────────────────────────────

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

        p.resetSimulation(physicsClientId=self.client)
        p.setGravity(0, 0, -9.8, physicsClientId=self.client)
        p.setTimeStep(1.0 / self.PYB_FREQ, physicsClientId=self.client)
        p.loadURDF("plane.urdf", physicsClientId=self.client)

        self.DRONE_IDS       = []
        self.obstacle_ids    = []
        self.goal_id         = None
        self.step_counter    = 0
        self.crashed         = np.zeros(self.num_drones, dtype=bool)
        self.crash_type      = [None] * self.num_drones
        self.prev_cmd        = np.zeros((self.num_drones, 4), dtype=np.float32)
        self.is_success      = False
        self.is_collision    = False
        self.mission_time    = 0
        self._last_valid_obs = {}
        self.goal_reached    = False

        self._loadDrones()
        self._addObstacles()

        # Safe goal candidates — shuffled each episode
        candidates = [
            np.array([ 0.0,  2.5, 1.0], dtype=np.float32),
            np.array([ 0.0, -2.5, 1.0], dtype=np.float32),
            np.array([ 2.0,  2.0, 1.0], dtype=np.float32),
            np.array([-2.0,  2.0, 1.0], dtype=np.float32),
            np.array([-2.0, -2.0, 1.0], dtype=np.float32),
            np.array([ 2.0, -2.0, 1.0], dtype=np.float32),
        ]
        obs_positions = []
        for obs_id in self.obstacle_ids:
            op, _ = p.getBasePositionAndOrientation(obs_id, physicsClientId=self.client)
            obs_positions.append(np.array(op))

        np.random.shuffle(candidates)
        self.goal_pos = candidates[0]
        for c in candidates:
            if all(np.linalg.norm(c[:2] - op[:2]) >= 1.2 for op in obs_positions):
                self.goal_pos = c
                break

        goal_vis = p.createVisualShape(
            p.GEOM_SPHERE, radius=0.3, rgbaColor=[1.0, 0.8, 0.0, 0.8],
            physicsClientId=self.client
        )
        self.goal_id = p.createMultiBody(
            baseMass=0, baseVisualShapeIndex=goal_vis,
            basePosition=self.goal_pos.tolist(), physicsClientId=self.client
        )

        for i in range(self.num_drones):
            pos, _ = p.getBasePositionAndOrientation(self.DRONE_IDS[i], physicsClientId=self.client)
            self._prev_dist[f"drone_{i}"] = float(np.linalg.norm(np.array(pos) - self.goal_pos))

        return self._computeObs(), {}

    # ──────────────────────────────────────────────────────────────
    # step
    # ──────────────────────────────────────────────────────────────

    def step(self, actions):
        for i in range(self.num_drones):
            if not self.crashed[i]:
                key = f"drone_{i}"
                act = actions.get(key, np.zeros(4)) if isinstance(actions, dict) else actions[i]
                self._applyAction(np.array(act, dtype=np.float32), i)

        for _ in range(self.PYB_STEPS_PER_CTRL):
            p.stepSimulation(physicsClientId=self.client)

        self._check_crashes()
        self.step_counter += 1
        self.mission_time  = self.step_counter

        obs        = self._computeObs()
        terminated = self._computeTerminated()
        truncated  = self._computeTruncated()
        rewards    = self._computeRewards()
        info       = {"goal_reached": self.is_success, "is_collision": self.is_collision}

        for i in range(self.num_drones):
            pos, _ = p.getBasePositionAndOrientation(self.DRONE_IDS[i], physicsClientId=self.client)
            self._prev_dist[f"drone_{i}"] = float(np.linalg.norm(np.array(pos) - self.goal_pos))

        return obs, rewards, terminated, truncated, info

    # ──────────────────────────────────────────────────────────────
    # Internal — match training env exactly
    # ──────────────────────────────────────────────────────────────

    def _applyAction(self, action: np.ndarray, idx: int):
        action[0:3] = np.clip(action[0:3], -self.MAX_SPEED,    self.MAX_SPEED)
        action[3]   = np.clip(action[3],   -self.MAX_YAW_RATE, self.MAX_YAW_RATE)

        prev    = self.prev_cmd[idx].copy()
        max_dv  = self.MAX_ACCEL * self.CTRL_TIMESTEP
        dv      = np.clip(action[0:3] - prev[0:3], -max_dv, max_dv)

        limited        = prev.copy()
        limited[0:3]   = prev[0:3] + dv
        limited[3]     = action[3]

        smoothed = self.ACTION_SMOOTHING * prev + (1.0 - self.ACTION_SMOOTHING) * limited
        smoothed = np.nan_to_num(smoothed, nan=0., posinf=1., neginf=-1.)

        p.resetBaseVelocity(
            self.DRONE_IDS[idx],
            linearVelocity=smoothed[0:3].tolist(),
            angularVelocity=[0., 0., float(smoothed[3])],
            physicsClientId=self.client
        )
        self.prev_cmd[idx] = smoothed

    def _computeObs(self) -> dict:
        obs = {}
        for i in range(self.num_drones):
            aid       = f"drone_{i}"
            pos, quat = p.getBasePositionAndOrientation(self.DRONE_IDS[i], physicsClientId=self.client)
            vel, _    = p.getBaseVelocity(self.DRONE_IDS[i], physicsClientId=self.client)
            rpy       = p.getEulerFromQuaternion(quat)

            min_d     = self._get_min_dist(i)
            proximity = float(np.clip(min_d / self.PROX_RANGE, 0.0, 1.0))
            rel_goal  = np.clip((self.goal_pos - np.array(pos)) / 5.0, -1., 1.)

            own_obs = np.concatenate([
                np.clip(np.array(pos) / 5.0,            -1., 1.),
                np.clip(np.array(vel) / self.MAX_SPEED, -1., 1.),
                np.clip(np.array(rpy) / np.pi,          -1., 1.),
                np.array([proximity]),
                rel_goal,
            ]).astype(np.float32)

            if np.any(np.isnan(own_obs)) or np.any(np.isinf(own_obs)):
                own_obs = self._last_valid_obs.get(aid, np.zeros(13, dtype=np.float32))
            else:
                self._last_valid_obs[aid] = own_obs.copy()

            obs[aid] = own_obs
        return obs

    def _get_min_dist(self, drone_idx: int) -> float:
        pos, _ = p.getBasePositionAndOrientation(self.DRONE_IDS[drone_idx], physicsClientId=self.client)
        pos    = np.array(pos)
        min_d  = float("inf")
        for j in range(self.num_drones):
            if j != drone_idx:
                op, _ = p.getBasePositionAndOrientation(self.DRONE_IDS[j], physicsClientId=self.client)
                min_d = min(min_d, np.linalg.norm(pos - np.array(op)))
        for obs_id in self.obstacle_ids:
            op, _ = p.getBasePositionAndOrientation(obs_id, physicsClientId=self.client)
            min_d = min(min_d, np.linalg.norm(pos - np.array(op)))
        return min_d if min_d < float("inf") else 0.0

    def _check_crashes(self):
        for i in range(self.num_drones):
            if self.crashed[i]:
                continue
            for j in range(i + 1, self.num_drones):
                if p.getContactPoints(self.DRONE_IDS[i], self.DRONE_IDS[j], physicsClientId=self.client):
                    self.crashed[i] = self.crashed[j] = True
                    self.crash_type[i] = self.crash_type[j] = "drone_collision"
            for obs_id in self.obstacle_ids:
                if p.getContactPoints(self.DRONE_IDS[i], obs_id, physicsClientId=self.client):
                    self.crashed[i]    = True
                    self.crash_type[i] = "obstacle_collision"

    def _computeTerminated(self) -> dict:
        all_keys = {f"drone_{i}": False for i in range(self.num_drones)}

        if any(self.crashed):
            self.is_collision = True
            return {k: True for k in all_keys}

        for i in range(self.num_drones):
            pos, _ = p.getBasePositionAndOrientation(self.DRONE_IDS[i], physicsClientId=self.client)
            x, y, z = pos
            if np.linalg.norm([x, y]) > self.MAX_BOUND_XY or z > self.MAX_HEIGHT or z < self.MIN_HEIGHT:
                return {k: True for k in all_keys}

        if all(
            np.linalg.norm(
                np.array(p.getBasePositionAndOrientation(self.DRONE_IDS[i], physicsClientId=self.client)[0])
                - self.goal_pos
            ) <= self.GOAL_RADIUS
            for i in range(self.num_drones)
        ):
            self.is_success   = True
            self.goal_reached = True
            return {k: True for k in all_keys}

        return all_keys

    def _computeTruncated(self) -> dict:
        done = self.step_counter >= self.MAX_STEPS
        return {f"drone_{i}": done for i in range(self.num_drones)}

    def _computeRewards(self) -> dict:
        rewards = {}
        for i in range(self.num_drones):
            aid    = f"drone_{i}"
            pos, _ = p.getBasePositionAndOrientation(self.DRONE_IDS[i], physicsClientId=self.client)
            dist   = float(np.linalg.norm(np.array(pos) - self.goal_pos))
            prev_d = self._prev_dist.get(aid, dist)

            r = 0.001
            if self.crashed[i]:
                r -= 50.0
            else:
                r += 4.0 * (prev_d - dist)
            if dist <= self.GOAL_RADIUS:
                r += 200.0
            rewards[aid] = float(r)
        return rewards

    def _loadDrones(self):
        self.DRONE_IDS = []
        for i in range(self.num_drones):
            angle = 2 * np.pi * i / self.num_drones
            x, y  = np.cos(angle), np.sin(angle)
            col   = p.createCollisionShape(p.GEOM_SPHERE, radius=0.05, physicsClientId=self.client)
            vis   = p.createVisualShape(p.GEOM_SPHERE, radius=0.1,
                                         rgbaColor=[0.2, 0.8, 0.2, 1.0],
                                         physicsClientId=self.client)
            drone_id = p.createMultiBody(
                baseMass=0.5,
                baseCollisionShapeIndex=col,
                baseVisualShapeIndex=vis,
                basePosition=[x, y, 1.0],
                physicsClientId=self.client
            )
            self.DRONE_IDS.append(drone_id)

    def _addObstacles(self):
        self.obstacle_ids = []
        assets    = pybullet_data.getDataPath()
        cube_urdf = os.path.join(assets, "cube_no_rotation.urdf")

        if self.num_obstacles == 2:
            positions = [[2.5, 0., .4], [-2.5, 0., .4]]
        elif self.num_obstacles == 3:
            positions = [[2.5, 0., .4], [-2.5, 0., .4], [0., 1.5, .4]]
        else:
            positions = [[2.5, 0., .4], [-2.5, 0., .4], [0., 1.5, .4], [1.5, 2.5, .4]]

        for pos in positions:
            if os.path.exists(cube_urdf):
                obs_id = p.loadURDF(cube_urdf, pos, globalScaling=0.8, physicsClientId=self.client)
                p.changeDynamics(obs_id, -1, mass=0.0, physicsClientId=self.client)
            else:
                col    = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.3, height=0.8,
                                                 physicsClientId=self.client)
                vis    = p.createVisualShape(p.GEOM_CYLINDER, radius=0.3, length=0.8,
                                              rgbaColor=[0.8, 0.1, 0.1, 1.0],
                                              physicsClientId=self.client)
                obs_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col,
                                            baseVisualShapeIndex=vis, basePosition=pos,
                                            physicsClientId=self.client)
            self.obstacle_ids.append(obs_id)

    def close(self):
        if self.client >= 0:
            p.disconnect(physicsClientId=self.client)
            self.client = -1
