"""
custom_aviary_maddpg.py — CustomAviary (MADDPG Edition)
===============================================================
UAV Swarm Environment adapted to be compatible with MADDPG

Differences from MAPPO version:
  - observation_space : simple Box (13D) instead of Dict
    → global_state is computed by the Policy, not the environment
  - action_space      : continuous Box [vx, vy, vz, yaw_rate]  (unchanged)
  - added get_global_state() as a separate method called by the Trainer
  - remaining logic (reward, termination, ...) is identical
"""

import os
import random
import numpy as np
import pybullet as p
import pybullet_data
from gymnasium import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from importlib.resources import files
import gym_pybullet_drones


class CustomAviaryMADDPG(MultiAgentEnv):
    """
    Multi-Agent UAV Swarm — MADDPG Edition

    Control mode : velocity-level  [vx, vy, vz, yaw_rate]
    Observation  : Box(13,) per agent  <- Actor input (local only)
    Global State : Box(N,)             <- Critic input (computed externally)
    Reward       : cooperative
    Framework    : RLlib MultiAgentEnv
    """

    # ═══════════════════════════════════════════════════════════
    # __init__
    # ═══════════════════════════════════════════════════════════

    def __init__(self, config: dict | None = None) -> None:
        super().__init__()

        config         = config or {}
        num_drones     = config.get("num_drones", 4)
        gui            = config.get("gui", False)

        self.NUM_DRONES = num_drones
        self.GUI        = gui
        self.CLIENT     = -1

        # -- Agent IDs ---------------------------------------------------
        self._agent_ids      = set(f"drone_{i}" for i in range(num_drones))
        self.possible_agents = sorted(self._agent_ids)
        self._agent_to_idx   = {f"drone_{i}": i for i in range(num_drones)}
        self.agents          = list(self.possible_agents)

        # -- Physics -----------------------------------------------------
        self.PYB_FREQ           = 240
        self.CTRL_FREQ          = 48
        self.PYB_STEPS_PER_CTRL = self.PYB_FREQ // self.CTRL_FREQ
        self.CTRL_TIMESTEP      = 1.0 / self.CTRL_FREQ
        self.GRAVITY            = 9.8

        # -- Safety limits -----------------------------------------------
        self.MAX_SPEED    = 1.0
        self.MAX_YAW_RATE = 1.0
        self.MAX_BOUND_XY = 5.0
        self.MAX_HEIGHT   = 3.0
        self.MIN_HEIGHT   = 0.2

        # -- Obstacles ---------------------------------------------------
        self.NUM_OBSTACLES = 3

        # -- Spaces ------------------------------------------------------
        # MADDPG: each agent sees only own_obs (13D Box)
        # global_state is computed separately for the Critic
        self.observation_space = self._build_obs_space()
        self.action_space      = self._build_act_space()

        # -- Global state dimension (for Critic) -------------------------
        # num_drones*6 (pos+vel) + num_obstacles*3 (pos) + 3 (goal)
        self.global_state_dim = num_drones * 6 + self.NUM_OBSTACLES * 3 + 3

        # -- Goal --------------------------------------------------------
        self.GOAL_XY_RANGE = 3.0    # reduced from 5.0 -> closer goals -> easier to reach
        self.GOAL_Z_RANGE  = [0.8, 1.5]  # tighter Z range -> more reachable height
        self.goal_pos      = np.zeros(3, dtype=np.float32)
        # -- GOAL_RADIUS ثابت عند 3.0m - منطقي لـ 4 درونات بمسافة أمان 1.0m --
        self._curriculum_episode  = 0
        self._curriculum_schedule = [
            (0, 3.0),   # ثابت للكل
        ]
        self.GOAL_RADIUS = 3.0

        # -- Episode control ---------------------------------------------
        self.MAX_STEPS          = 300   # reverted back - 500 was worse
        self.step_counter       = 0
        self.termination_reason = None
        self.is_success         = False
        self.is_collision       = False
        self.mission_time       = 0
        self.avg_dist_to_goal   = 0.0
        self.avg_drone_dist     = 0.0

        # -- Action smoothing --------------------------------------------
        self.MAX_ACCEL        = 2.0
        self.ACTION_SMOOTHING = 0.5
        self.prev_cmd         = np.zeros((num_drones, 4), dtype=np.float32)

        # -- Runtime state -----------------------------------------------
        self.crashed         = np.zeros(num_drones, dtype=bool)
        self.crash_type      = [None] * num_drones
        self._prev_dist      = {aid: float("inf") for aid in self._agent_ids}
        self._last_valid_obs = {}
        self.DRONE_IDS       = []
        self.obstacle_ids    = []
        self._cached_min_dists = {}

    # ═══════════════════════════════════════════════════════════
    # Spaces
    # ═══════════════════════════════════════════════════════════

    def _build_obs_space(self) -> spaces.Dict:
        """
        MADDPG: simple 13D observation per agent (Box only, not Dict)

        own_obs layout (13D):
            [0:3]   pos / 5.0          in [-1, 1]
            [3:6]   vel / MAX_SPEED    in [-1, 1]
            [6:9]   rpy / pi           in [-1, 1]
            [9]     proximity          in [0,  1]
            [10:13] rel_goal / 5.0     in [-1, 1]
        """
        lo = np.array([-1.]*3 + [-1.]*3 + [-1.]*3 + [0.] + [-1.]*3,
                      dtype=np.float32)
        hi = np.ones(13, dtype=np.float32)
        obs_box = spaces.Box(low=lo, high=hi, dtype=np.float32)
        return spaces.Dict({aid: obs_box for aid in self._agent_ids})

    def _build_act_space(self) -> spaces.Dict:
        """Action: [vx, vy, vz, yaw_rate]"""
        lo = np.array([-self.MAX_SPEED]*3 + [-self.MAX_YAW_RATE],
                      dtype=np.float32)
        hi = np.array([+self.MAX_SPEED]*3 + [+self.MAX_YAW_RATE],
                      dtype=np.float32)
        box = spaces.Box(low=lo, high=hi, dtype=np.float32)
        return spaces.Dict({aid: box for aid in self._agent_ids})

    # ═══════════════════════════════════════════════════════════
    # reset
    # ═══════════════════════════════════════════════════════════

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)

        if seed is not None:
            random.seed(seed)

        # -- Curriculum update -------------------------------------------
        self._curriculum_episode += 1
        for ep_thresh, radius in reversed(self._curriculum_schedule):
            if self._curriculum_episode >= ep_thresh:
                self.GOAL_RADIUS = radius
                break

        # -- PyBullet ----------------------------------------------------
        if self.CLIENT == -1:
            self.CLIENT = p.connect(p.GUI if self.GUI else p.DIRECT)

        p.resetSimulation(physicsClientId=self.CLIENT)
        p.setGravity(0, 0, -self.GRAVITY, physicsClientId=self.CLIENT)
        p.setTimeStep(1.0 / self.PYB_FREQ, physicsClientId=self.CLIENT)
        p.loadURDF(
            os.path.join(pybullet_data.getDataPath(), "plane.urdf"),
            physicsClientId=self.CLIENT,
        )

        self._addObstacles()
        self._loadDrones()

        # -- Random goal -------------------------------------------------
        self.goal_pos = np.array([
            self.np_random.uniform(-self.GOAL_XY_RANGE, self.GOAL_XY_RANGE),
            self.np_random.uniform(-self.GOAL_XY_RANGE, self.GOAL_XY_RANGE),
            self.np_random.uniform(*self.GOAL_Z_RANGE),
        ], dtype=np.float32)

        # -- Reset state -------------------------------------------------
        self.step_counter       = 0
        self.crashed            = np.zeros(self.NUM_DRONES, dtype=bool)
        self.crash_type         = [None] * self.NUM_DRONES
        self.prev_cmd           = np.zeros((self.NUM_DRONES, 4), dtype=np.float32)
        self.termination_reason = None
        self.is_success         = False
        self.is_collision       = False
        self.mission_time       = 0
        self.avg_dist_to_goal   = 0.0
        self.avg_drone_dist     = 0.0
        self.agents             = list(self.possible_agents)
        self._last_valid_obs    = {}
        self._cached_min_dists  = {}

        # -- Initialize _prev_dist ----------------------------------------
        for i, aid in enumerate(self.possible_agents):
            pos, _ = p.getBasePositionAndOrientation(
                self.DRONE_IDS[i], physicsClientId=self.CLIENT
            )
            self._prev_dist[aid] = float(
                np.linalg.norm(np.array(pos) - self.goal_pos)
            )

        obs  = self._computeObs()
        info = self._computeInfo()
        return obs, info

    # ═══════════════════════════════════════════════════════════
    # step
    # ═══════════════════════════════════════════════════════════

    def step(self, action_dict):
        # -- Apply actions -----------------------------------------------
        for agent_id, act in action_dict.items():
            idx = self._agent_to_idx[agent_id]
            if not self.crashed[idx]:
                self._applyAction(act, idx)

        # -- Physics steps -----------------------------------------------
        for _ in range(self.PYB_STEPS_PER_CTRL):
            p.stepSimulation(physicsClientId=self.CLIENT)

        # -- Crash detection ---------------------------------------------
        self._check_crashes()

        # -- Cache min distances -----------------------------------------
        self._cached_min_dists = {
            i: self._get_min_dist(i) for i in range(self.NUM_DRONES)
        }

        # -- Outputs -----------------------------------------------------
        obs        = self._computeObs()
        reward     = self._computeReward()
        terminated = self._computeTerminated()
        truncated  = self._computeTruncated()
        info       = self._computeInfo()

        # -- Update _prev_dist --------------------------------------------
        for i, aid in enumerate(self.possible_agents):
            pos, _ = p.getBasePositionAndOrientation(
                self.DRONE_IDS[i], physicsClientId=self.CLIENT
            )
            self._prev_dist[aid] = float(
                np.linalg.norm(np.array(pos) - self.goal_pos)
            )

        # -- __all__ key required by RLlib --------------------------------
        terminated["__all__"] = any(terminated[aid] for aid in self._agent_ids)
        truncated["__all__"]  = any(truncated[aid]  for aid in self._agent_ids)

        self.step_counter += 1
        self.mission_time  = self.step_counter
        return obs, reward, terminated, truncated, info

    # ═══════════════════════════════════════════════════════════
    # get_global_state  <- key addition for MADDPG
    # ═══════════════════════════════════════════════════════════

    def get_global_state(self) -> np.ndarray:
        """
        Called by the MADDPG Trainer to gather info for the Critic.

        In MAPPO: global_state was part of the observation (Dict).
        In MADDPG: the Critic receives:
            - observations of all agents
            - actions of all agents
        This function returns the global state for the Critic (optional).

        Layout:
            Per drone (6D each): [pos/5 (3D), vel/MAX_SPEED (3D)]
            Per obstacle (3D each): [pos/5 (3D)]
            Goal (3D): [goal_pos/5]
        Total = num_drones*6 + num_obstacles*3 + 3
        """
        parts = []

        for i in range(self.NUM_DRONES):
            pos, _ = p.getBasePositionAndOrientation(
                self.DRONE_IDS[i], physicsClientId=self.CLIENT
            )
            vel, _ = p.getBaseVelocity(
                self.DRONE_IDS[i], physicsClientId=self.CLIENT
            )
            parts.extend(np.clip(np.array(pos) / 5.0, -1., 1.))
            parts.extend(np.clip(np.array(vel) / self.MAX_SPEED, -1., 1.))

        for obs_id in self.obstacle_ids:
            obs_pos, _ = p.getBasePositionAndOrientation(
                obs_id, physicsClientId=self.CLIENT
            )
            parts.extend(np.clip(np.array(obs_pos) / 5.0, -1., 1.))

        parts.extend(np.clip(self.goal_pos / 5.0, -1., 1.))

        return np.array(parts, dtype=np.float32)

    # ═══════════════════════════════════════════════════════════
    # _computeObs  -- simple Box (not Dict)
    # ═══════════════════════════════════════════════════════════

    def _computeObs(self) -> dict:
        """
        MADDPG: each agent sees only own_obs (13D Box).
        The Critic takes concat(all obs, all actions) -- not from here.
        """
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
                np.clip(np.array(pos) / 5.0,            -1., 1.),
                np.clip(np.array(vel) / self.MAX_SPEED, -1., 1.),
                np.clip(np.array(rpy) / np.pi,          -1., 1.),
                np.array([proximity]),
                rel_goal,
            ]).astype(np.float32)

            # NaN/Inf safety
            if np.any(np.isnan(own_obs)) or np.any(np.isinf(own_obs)):
                own_obs = self._last_valid_obs.get(
                    aid, np.zeros(13, dtype=np.float32)
                )
            else:
                self._last_valid_obs[aid] = own_obs.copy()

            obs[aid] = own_obs  # <- direct Box, not Dict

        return obs

    # ═══════════════════════════════════════════════════════════
    # _computeReward
    # ═══════════════════════════════════════════════════════════

    def _computeReward(self) -> dict:
        CRASH_PENALTY = -50.0   # رُفع من -20 لتقليل التصادم
        GOAL_REWARD   = +200.0
        SURVIVAL      = +0.001
        SHAPING_W     = +4.0
        PROX_W        = -3.0    # قوي لضمان 0% collision
        SAFE_DIST     = 1.0     # رُفع من 0.8 لمنطقة أمان أوسع
        TEAM_BONUS    = +100.0
        SEP_W         = +1.0    # جديد: مكافأة التباعد الآمن بين الدرونات

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

            # مكافأة التباعد الآمن: إذا الدرونات بعيدة بما يكفي عن بعض
            r_sep = SEP_W if min_d >= SAFE_DIST else 0.0

            if dist > self.GOAL_RADIUS:
                all_at_goal = False

            reward[aid] = float(r_crash + r_survive + r_shape + r_goal + r_prox + r_sep)

        if all_at_goal and not any(self.crashed):
            for k in reward:
                reward[k] += TEAM_BONUS

        return reward

    # ═══════════════════════════════════════════════════════════
    # _computeTerminated
    # ═══════════════════════════════════════════════════════════

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

        # Success check
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

    # ═══════════════════════════════════════════════════════════
    # _computeTruncated
    # ═══════════════════════════════════════════════════════════

    def _computeTruncated(self) -> dict:
        if self.step_counter >= self.MAX_STEPS:
            self.termination_reason = "timeout"
            return {aid: True for aid in self.possible_agents}
        return {aid: False for aid in self.possible_agents}

    # ═══════════════════════════════════════════════════════════
    # _computeInfo
    # ═══════════════════════════════════════════════════════════

    def _computeInfo(self) -> dict:
        dists_to_goal = []
        for i in range(self.NUM_DRONES):
            pos, _ = p.getBasePositionAndOrientation(
                self.DRONE_IDS[i], physicsClientId=self.CLIENT
            )
            dists_to_goal.append(np.linalg.norm(np.array(pos) - self.goal_pos))
        self.avg_dist_to_goal = float(np.mean(dists_to_goal))

        drone_dists = []
        for i in range(self.NUM_DRONES):
            pos_i, _ = p.getBasePositionAndOrientation(
                self.DRONE_IDS[i], physicsClientId=self.CLIENT
            )
            for j in range(i + 1, self.NUM_DRONES):
                pos_j, _ = p.getBasePositionAndOrientation(
                    self.DRONE_IDS[j], physicsClientId=self.CLIENT
                )
                drone_dists.append(
                    np.linalg.norm(np.array(pos_i) - np.array(pos_j))
                )
        self.avg_drone_dist = float(np.mean(drone_dists)) if drone_dists else 0.0

        return {
            aid: {
                "crashed":            bool(self.crashed[i]),
                "termination_reason": self.termination_reason,
                "dist_to_goal":       self._prev_dist.get(aid, 0.0),
                "is_success":         self.is_success,
                "is_collision":       self.is_collision,
                "crash_type":         self.crash_type[i] if self.crashed[i] else "none",
                "mission_time":       self.mission_time,
                "avg_dist_to_goal":   self.avg_dist_to_goal,
                "avg_drone_dist":     self.avg_drone_dist,
            }
            for i, aid in enumerate(self.possible_agents)
        }

    # ═══════════════════════════════════════════════════════════
    # Helpers
    # ═══════════════════════════════════════════════════════════

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
        self.DRONE_IDS = []
        drone_path     = str(files(gym_pybullet_drones) / "assets" / "cf2x.urdf")
        for i in range(self.NUM_DRONES):
            angle    = 2 * np.pi * i / self.NUM_DRONES
            drone_id = p.loadURDF(
                drone_path,
                [np.cos(angle), np.sin(angle), 1.0],
                p.getQuaternionFromEuler([0, 0, 0]),
                physicsClientId=self.CLIENT,
            )
            self.DRONE_IDS.append(drone_id)

    def _addObstacles(self) -> None:
        self.obstacle_ids = []
        assets = pybullet_data.getDataPath()
        for pos in [[2.5, 0., .4], [-2.5, 0., .4], [0., 1.5, .4]]:
            cube = p.loadURDF(
                os.path.join(assets, "cube_no_rotation.urdf"),
                pos,
                globalScaling=0.8,
                physicsClientId=self.CLIENT,
            )
            p.changeDynamics(cube, -1, mass=0.0, physicsClientId=self.CLIENT)
            self.obstacle_ids.append(cube)

    def _applyAction(self, action: np.ndarray, idx: int) -> None:
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