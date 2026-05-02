"""
custom_aviary_standalone.py
===========================
Standalone deployment version of CustomAviaryMADDPG
Matches training environment EXACTLY - no Ray dependency
"""

import os
import random
import numpy as np
import pybullet as p
import pybullet_data
from gymnasium import spaces


class CustomAviaryMADDPG:
    """
    Standalone MADDPG environment matching training exactly.
    Removes Ray/RLlib dependency for deployment.
    """

    def __init__(self, num_drones=4, num_obstacles=4, gui=False, **kwargs):
        self.NUM_DRONES    = num_drones
        self.NUM_OBSTACLES = num_obstacles
        self.GUI           = gui
        self.CLIENT        = -1

        # Agent IDs - EXACT match to training
        self._agent_ids      = set(f"drone_{i}" for i in range(num_drones))
        self.possible_agents = sorted(self._agent_ids)
        self._agent_to_idx   = {f"drone_{i}": i for i in range(num_drones)}
        self.agents          = list(self.possible_agents)

        # Physics - EXACT match to training
        self.PYB_FREQ           = 240
        self.CTRL_FREQ          = 48
        self.PYB_STEPS_PER_CTRL = self.PYB_FREQ // self.CTRL_FREQ  # = 5
        self.CTRL_TIMESTEP      = 1.0 / self.CTRL_FREQ
        self.GRAVITY            = 9.8

        # Safety limits - EXACT match
        self.MAX_SPEED    = 1.0
        self.MAX_YAW_RATE = 1.0
        self.MAX_BOUND_XY = 5.0
        self.MAX_HEIGHT   = 3.0
        self.MIN_HEIGHT   = 0.2

        # Goal - EXACT match
        self.GOAL_XY_RANGE = 3.0
        self.GOAL_Z_RANGE  = [0.8, 1.5]
        self.goal_pos      = np.zeros(3, dtype=np.float32)
        self.GOAL_RADIUS   = 3.0

        # Episode control - EXACT match
        self.MAX_STEPS    = 300
        self.step_counter = 0

        # Action smoothing - EXACT match
        self.MAX_ACCEL        = 2.0
        self.ACTION_SMOOTHING = 0.5
        self.prev_cmd         = np.zeros((num_drones, 4), dtype=np.float32)

        # Runtime state
        self.crashed           = np.zeros(num_drones, dtype=bool)
        self.crash_type        = [None] * num_drones
        self._prev_dist        = {aid: float("inf") for aid in self._agent_ids}
        self._last_valid_obs   = {}
        self.DRONE_IDS         = []
        self.obstacle_ids      = []
        self._cached_min_dists = {}

        self.termination_reason = None
        self.is_success         = False
        self.is_collision       = False
        self.mission_time       = 0
        self.avg_dist_to_goal   = 0.0
        self.avg_drone_dist     = 0.0

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        if self.CLIENT == -1:
            self.CLIENT = p.connect(p.GUI if self.GUI else p.DIRECT)

        p.resetSimulation(physicsClientId=self.CLIENT)
        p.setGravity(0, 0, -self.GRAVITY, physicsClientId=self.CLIENT)
        p.setTimeStep(1.0 / self.PYB_FREQ, physicsClientId=self.CLIENT)
        p.loadURDF(
            os.path.join(pybullet_data.getDataPath(), "plane.urdf"),
            physicsClientId=self.CLIENT
        )

        self._addObstacles()
        self._loadDrones()

        # Random goal - EXACT match
        self.goal_pos = np.array([
            np.random.uniform(-self.GOAL_XY_RANGE, self.GOAL_XY_RANGE),
            np.random.uniform(-self.GOAL_XY_RANGE, self.GOAL_XY_RANGE),
            np.random.uniform(*self.GOAL_Z_RANGE),
        ], dtype=np.float32)

        self.step_counter       = 0
        self.crashed            = np.zeros(self.NUM_DRONES, dtype=bool)
        self.crash_type         = [None] * self.NUM_DRONES
        self.prev_cmd           = np.zeros((self.NUM_DRONES, 4), dtype=np.float32)
        self.termination_reason = None
        self.is_success         = False
        self.is_collision       = False
        self.mission_time       = 0
        self.agents             = list(self.possible_agents)
        self._last_valid_obs    = {}
        self._cached_min_dists  = {}

        for i, aid in enumerate(self.possible_agents):
            pos, _ = p.getBasePositionAndOrientation(
                self.DRONE_IDS[i], physicsClientId=self.CLIENT
            )
            self._prev_dist[aid] = float(np.linalg.norm(np.array(pos) - self.goal_pos))

        obs  = self._computeObs()
        info = self._computeInfo()
        return obs, info

    def step(self, action_dict):
        """action_dict: {drone_0: np.array(4), drone_1: ...} - EXACT match"""

        for agent_id, act in action_dict.items():
            idx = self._agent_to_idx[agent_id]
            if not self.crashed[idx]:
                self._applyAction(act, idx)

        # 5 physics steps per control step - EXACT match
        for _ in range(self.PYB_STEPS_PER_CTRL):
            p.stepSimulation(physicsClientId=self.CLIENT)

        self._check_crashes()

        self._cached_min_dists = {
            i: self._get_min_dist(i) for i in range(self.NUM_DRONES)
        }

        obs        = self._computeObs()
        reward     = self._computeReward()
        terminated = self._computeTerminated()
        truncated  = self._computeTruncated()
        info       = self._computeInfo()

        for i, aid in enumerate(self.possible_agents):
            pos, _ = p.getBasePositionAndOrientation(
                self.DRONE_IDS[i], physicsClientId=self.CLIENT
            )
            self._prev_dist[aid] = float(np.linalg.norm(np.array(pos) - self.goal_pos))

        terminated["__all__"] = any(terminated[aid] for aid in self._agent_ids)
        truncated["__all__"]  = any(truncated[aid]  for aid in self._agent_ids)

        self.step_counter += 1
        self.mission_time  = self.step_counter

        return obs, reward, terminated, truncated, info

    def _computeObs(self) -> dict:
        """EXACT observation format from training"""
        obs        = {}
        PROX_RANGE = 5.0

        for i, aid in enumerate(self.possible_agents):
            pos, quat = p.getBasePositionAndOrientation(
                self.DRONE_IDS[i], physicsClientId=self.CLIENT
            )
            vel, _ = p.getBaseVelocity(
                self.DRONE_IDS[i], physicsClientId=self.CLIENT
            )
            rpy = p.getEulerFromQuaternion(quat)

            min_d     = self._cached_min_dists.get(i, self._get_min_dist(i))
            proximity = float(np.clip(min_d / PROX_RANGE, 0.0, 1.0))
            rel_goal  = np.clip((self.goal_pos - np.array(pos)) / 5.0, -1., 1.)

            own_obs = np.concatenate([
                np.clip(np.array(pos) / 5.0,             -1., 1.),  # [0:3]
                np.clip(np.array(vel) / self.MAX_SPEED,  -1., 1.),  # [3:6]
                np.clip(np.array(rpy) / np.pi,           -1., 1.),  # [6:9]
                np.array([proximity]),                                # [9]
                rel_goal,                                             # [10:13]
            ]).astype(np.float32)

            if np.any(np.isnan(own_obs)) or np.any(np.isinf(own_obs)):
                own_obs = self._last_valid_obs.get(aid, np.zeros(13, dtype=np.float32))
            else:
                self._last_valid_obs[aid] = own_obs.copy()

            obs[aid] = own_obs

        return obs

    def _computeReward(self) -> dict:
        CRASH_PENALTY = -50.0
        GOAL_REWARD   = +200.0
        SURVIVAL      = +0.001
        SHAPING_W     = +4.0
        PROX_W        = -3.0
        SAFE_DIST     = 1.0
        TEAM_BONUS    = +100.0
        SEP_W         = +1.0

        reward      = {}
        all_at_goal = True

        for i, aid in enumerate(self.possible_agents):
            pos, _ = p.getBasePositionAndOrientation(
                self.DRONE_IDS[i], physicsClientId=self.CLIENT
            )
            pos  = np.array(pos)
            dist = float(np.linalg.norm(pos - self.goal_pos))

            r_crash   = CRASH_PENALTY if self.crashed[i] else 0.0
            r_survive = 0.0 if self.crashed[i] else SURVIVAL

            if not self.crashed[i]:
                prev_d  = self._prev_dist.get(aid, dist)
                r_shape = SHAPING_W * (prev_d - dist)
            else:
                r_shape = 0.0

            r_goal = GOAL_REWARD if dist <= self.GOAL_RADIUS else 0.0
            min_d  = self._cached_min_dists.get(i, self._get_min_dist(i))
            r_prox = 0.0

            if min_d < SAFE_DIST:
                ratio  = 1.0 - min_d / SAFE_DIST
                r_prox = PROX_W * (ratio ** 2)

            r_sep = SEP_W if min_d >= SAFE_DIST else 0.0

            if dist > self.GOAL_RADIUS:
                all_at_goal = False

            reward[aid] = float(r_crash + r_survive + r_shape + r_goal + r_prox + r_sep)

        if all_at_goal and not any(self.crashed):
            for k in reward:
                reward[k] += TEAM_BONUS

        return reward

    def _computeTerminated(self) -> dict:
        terminated = {aid: False for aid in self.possible_agents}

        for i, aid in enumerate(self.possible_agents):
            if self.crashed[i]:
                self.termination_reason = self.crash_type[i]
                self.is_collision = True
                return {aid: True for aid in self.possible_agents}

            pos, _ = p.getBasePositionAndOrientation(
                self.DRONE_IDS[i], physicsClientId=self.CLIENT
            )
            x, y, z = pos

            if np.linalg.norm([x, y]) > self.MAX_BOUND_XY:
                self.termination_reason = "out_of_bounds"
                return {aid: True for aid in self.possible_agents}

            if z > self.MAX_HEIGHT or z < self.MIN_HEIGHT:
                self.termination_reason = "height_violation"
                return {aid: True for aid in self.possible_agents}

        if all(
            np.linalg.norm(
                np.array(p.getBasePositionAndOrientation(
                    self.DRONE_IDS[i], physicsClientId=self.CLIENT
                )[0]) - self.goal_pos
            ) <= self.GOAL_RADIUS
            for i in range(self.NUM_DRONES)
        ):
            self.termination_reason = "success"
            self.is_success = True
            return {aid: True for aid in self.possible_agents}

        return terminated

    def _computeTruncated(self) -> dict:
        if self.step_counter >= self.MAX_STEPS:
            self.termination_reason = "timeout"
            return {aid: True for aid in self.possible_agents}
        return {aid: False for aid in self.possible_agents}

    def _computeInfo(self) -> dict:
        dists_to_goal = []
        for i in range(self.NUM_DRONES):
            pos, _ = p.getBasePositionAndOrientation(
                self.DRONE_IDS[i], physicsClientId=self.CLIENT
            )
            dists_to_goal.append(np.linalg.norm(np.array(pos) - self.goal_pos))
        self.avg_dist_to_goal = float(np.mean(dists_to_goal))

        return {
            aid: {
                "crashed":            bool(self.crashed[i]),
                "termination_reason": self.termination_reason,
                "dist_to_goal":       self._prev_dist.get(aid, 0.0),
                "is_success":         self.is_success,
                "is_collision":       self.is_collision,
            }
            for i, aid in enumerate(self.possible_agents)
        }

    def _get_min_dist(self, drone_idx: int) -> float:
        pos, _ = p.getBasePositionAndOrientation(
            self.DRONE_IDS[drone_idx], physicsClientId=self.CLIENT
        )
        pos      = np.array(pos)
        min_dist = float("inf")

        for j in range(self.NUM_DRONES):
            if j != drone_idx:
                op, _ = p.getBasePositionAndOrientation(
                    self.DRONE_IDS[j], physicsClientId=self.CLIENT
                )
                min_dist = min(min_dist, np.linalg.norm(pos - np.array(op)))

        for obs_id in self.obstacle_ids:
            op, _ = p.getBasePositionAndOrientation(
                obs_id, physicsClientId=self.CLIENT
            )
            min_dist = min(min_dist, np.linalg.norm(pos - np.array(op)))

        return min_dist

    def _loadDrones(self) -> None:
        """Load real cf2x drone URDFs - EXACT match to training"""
        self.DRONE_IDS = []

        drone_path = None
        try:
            from importlib.resources import files
            import gym_pybullet_drones
            drone_path = str(files(gym_pybullet_drones) / "assets" / "cf2x.urdf")
        except Exception:
            pass

        for i in range(self.NUM_DRONES):
            angle = 2 * np.pi * i / self.NUM_DRONES
            pos   = [np.cos(angle), np.sin(angle), 1.0]

            if drone_path and os.path.exists(drone_path):
                drone_id = p.loadURDF(
                    drone_path,
                    pos,
                    p.getQuaternionFromEuler([0, 0, 0]),
                    physicsClientId=self.CLIENT,
                )
            else:
                col = p.createCollisionShape(p.GEOM_SPHERE, radius=0.1)
                vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.1, rgbaColor=[0, 1, 0, 1])
                drone_id = p.createMultiBody(
                    baseMass=0.5,
                    baseCollisionShapeIndex=col,
                    baseVisualShapeIndex=vis,
                    basePosition=pos,
                    physicsClientId=self.CLIENT
                )

            self.DRONE_IDS.append(drone_id)

    def _addObstacles(self) -> None:
        """Load obstacles with exact positions from training"""
        self.obstacle_ids = []
        assets = pybullet_data.getDataPath()

        if self.NUM_OBSTACLES == 2:
            positions = [[2.5, 0., .4], [-2.5, 0., .4]]
        elif self.NUM_OBSTACLES == 3:
            positions = [[2.5, 0., .4], [-2.5, 0., .4], [0., 1.5, .4]]
        else:
            positions = [[2.5, 0., .4], [-2.5, 0., .4], [0., 1.5, .4], [1.5, 2.5, .4]]

        for pos in positions:
            try:
                cube = p.loadURDF(
                    os.path.join(assets, "cube_no_rotation.urdf"),
                    pos,
                    globalScaling=0.8,
                    physicsClientId=self.CLIENT,
                )
                p.changeDynamics(cube, -1, mass=0.0, physicsClientId=self.CLIENT)
            except Exception:
                col  = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.4, 0.4, 0.4])
                vis  = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.4, 0.4, 0.4], rgbaColor=[1, 0, 0, 1])
                cube = p.createMultiBody(
                    baseMass=0,
                    baseCollisionShapeIndex=col,
                    baseVisualShapeIndex=vis,
                    basePosition=pos,
                    physicsClientId=self.CLIENT
                )
            self.obstacle_ids.append(cube)

    def _applyAction(self, action: np.ndarray, idx: int) -> None:
        """EXACT match to training action application"""
        action = np.array(action, dtype=np.float32)
        action[:3] = np.clip(action[:3], -self.MAX_SPEED,    self.MAX_SPEED)
        action[3]  = np.clip(action[3],  -self.MAX_YAW_RATE, self.MAX_YAW_RATE)

        prev   = self.prev_cmd[idx].copy()
        max_dv = self.MAX_ACCEL * self.CTRL_TIMESTEP

        dv          = np.clip(action[:3] - prev[:3], -max_dv, max_dv)
        limited     = prev.copy()
        limited[:3] = prev[:3] + dv
        limited[3]  = action[3]

        smoothed    = (self.ACTION_SMOOTHING * prev
                       + (1.0 - self.ACTION_SMOOTHING) * limited)
        smoothed    = np.nan_to_num(smoothed, nan=0., posinf=1., neginf=-1.)

        p.resetBaseVelocity(
            self.DRONE_IDS[idx],
            linearVelocity=smoothed[:3].tolist(),
            angularVelocity=[0., 0., float(smoothed[3])],
            physicsClientId=self.CLIENT,
        )
        self.prev_cmd[idx] = smoothed

    def _check_crashes(self) -> None:
        for i in range(self.NUM_DRONES):
            if self.crashed[i]:
                continue
            for j in range(i + 1, self.NUM_DRONES):
                if p.getContactPoints(
                    self.DRONE_IDS[i], self.DRONE_IDS[j],
                    physicsClientId=self.CLIENT,
                ):
                    self.crashed[i] = self.crashed[j] = True
                    self.crash_type[i] = "drone_collision"
                    self.crash_type[j] = "drone_collision"

            for obs_id in self.obstacle_ids:
                if p.getContactPoints(
                    self.DRONE_IDS[i], obs_id,
                    physicsClientId=self.CLIENT,
                ):
                    self.crashed[i]    = True
                    self.crash_type[i] = "obstacle_collision"

    def close(self) -> None:
        if self.CLIENT >= 0:
            p.disconnect(physicsClientId=self.CLIENT)
            self.CLIENT = -1