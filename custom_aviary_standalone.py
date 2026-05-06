import os
import numpy as np
import pybullet as p
import pybullet_data
from gymnasium import spaces


class CustomAviaryMADDPG:

    PYB_FREQ = 240
    CTRL_FREQ = 48
    PYB_STEPS_PER_CTRL = 5
    CTRL_TIMESTEP = 1.0 / CTRL_FREQ

    MAX_SPEED = 2.0   # 🔥 increased for visible movement
    MAX_YAW_RATE = 1.0

    MAX_BOUND_XY = 5.0
    MAX_HEIGHT = 3.0
    MIN_HEIGHT = 0.2

    GOAL_RADIUS = 1.0   # 🔥 fixed (was too big)
    DRONE_RADIUS = 0.15

    MAX_STEPS = 300
    PROX_RANGE = 5.0

    ACTION_SMOOTHING = 0.5
    MAX_ACCEL = 2.0

    OBS_RADIUS = 0.55

    def __init__(self, num_drones=4, num_obstacles=4, gui=False):

        self.num_drones = num_drones
        self.num_obstacles = num_obstacles

        self.client = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        p.setTimeStep(1.0 / self.PYB_FREQ)

        self.DRONE_IDS = []
        self.obstacle_ids = []

        self.goal_pos = np.zeros(3)

        self.step_counter = 0
        self.crashed = np.zeros(num_drones, dtype=bool)

        self.prev_cmd = np.zeros((num_drones, 4))
        self._prev_dist = {}

    # ================= RESET =================
    def reset(self):
        p.resetSimulation()

        p.setGravity(0, 0, -9.8)
        p.loadURDF("plane.urdf")

        self.step_counter = 0
        self.crashed[:] = False
        self.prev_cmd[:] = 0

        self._load_drones()
        self._add_obstacles()

        # 🔥 Better goal placement (not too close)
        while True:
            self.goal_pos = np.array([
                np.random.uniform(-3, 3),
                np.random.uniform(-3, 3),
                np.random.uniform(0.8, 1.5)
            ])
            if np.linalg.norm(self.goal_pos[:2]) > 2.0:
                break

        for i in range(self.num_drones):
            pos, _ = p.getBasePositionAndOrientation(self.DRONE_IDS[i])
            self._prev_dist[i] = np.linalg.norm(np.array(pos) - self.goal_pos)

        return self._obs(), {}

    # ================= STEP =================
    def step(self, actions):

        for i in range(self.num_drones):
            if self.crashed[i]:
                continue

            act = actions[f"drone_{i}"]
            self._apply_action(act, i)

        for _ in range(self.PYB_STEPS_PER_CTRL):
            p.stepSimulation()

        self._check_crash()

        self.step_counter += 1

        obs = self._obs()
        rewards = self._reward()
        terminated = self._terminated()
        truncated = {f"drone_{i}": self.step_counter >= self.MAX_STEPS for i in range(self.num_drones)}

        info = {
            "goal_reached": all(terminated.values()) and not any(self.crashed)
        }

        return obs, rewards, terminated, truncated, info

    # ================= ACTION =================
    def _apply_action(self, action, i):

        action = np.clip(action, -1, 1)

        prev = self.prev_cmd[i]

        smoothed = self.ACTION_SMOOTHING * prev + (1 - self.ACTION_SMOOTHING) * action

        p.resetBaseVelocity(
            self.DRONE_IDS[i],
            linearVelocity=(smoothed[:3] * self.MAX_SPEED).tolist(),
            angularVelocity=[0, 0, smoothed[3]]
        )

        self.prev_cmd[i] = smoothed

    # ================= OBS =================
    def _obs(self):
        obs = {}

        for i in range(self.num_drones):
            pos, _ = p.getBasePositionAndOrientation(self.DRONE_IDS[i])
            vel, _ = p.getBaseVelocity(self.DRONE_IDS[i])

            rel_goal = (self.goal_pos - np.array(pos)) / 5.0

            obs[f"drone_{i}"] = np.concatenate([
                np.array(pos) / 5,
                np.array(vel),
                rel_goal
            ])

        return obs

    # ================= CRASH =================
    def _check_crash(self):

        for i in range(self.num_drones):
            pos_i, _ = p.getBasePositionAndOrientation(self.DRONE_IDS[i])

            for j in range(i + 1, self.num_drones):
                pos_j, _ = p.getBasePositionAndOrientation(self.DRONE_IDS[j])

                if np.linalg.norm(np.array(pos_i) - np.array(pos_j)) < 0.3:
                    self.crashed[i] = True
                    self.crashed[j] = True

            for obs_id in self.obstacle_ids:
                pos_o, _ = p.getBasePositionAndOrientation(obs_id)

                if np.linalg.norm(np.array(pos_i[:2]) - np.array(pos_o[:2])) < self.OBS_RADIUS:
                    self.crashed[i] = True

    # ================= TERMINATION =================
    def _terminated(self):

        result = {}

        for i in range(self.num_drones):
            pos, _ = p.getBasePositionAndOrientation(self.DRONE_IDS[i])

            dist = np.linalg.norm(np.array(pos) - self.goal_pos)

            # 🔥 FIXED goal logic (touch counts)
            reached = dist <= (self.GOAL_RADIUS + self.DRONE_RADIUS)

            out = (
                abs(pos[0]) > self.MAX_BOUND_XY or
                abs(pos[1]) > self.MAX_BOUND_XY or
                pos[2] < self.MIN_HEIGHT or
                pos[2] > self.MAX_HEIGHT
            )

            result[f"drone_{i}"] = reached or out or self.crashed[i]

        return result

    # ================= REWARD =================
    def _reward(self):
        rewards = {}

        for i in range(self.num_drones):
            pos, _ = p.getBasePositionAndOrientation(self.DRONE_IDS[i])

            dist = np.linalg.norm(np.array(pos) - self.goal_pos)
            prev = self._prev_dist[i]

            r = (prev - dist) * 5.0

            if self.crashed[i]:
                r -= 50

            if dist <= self.GOAL_RADIUS:
                r += 200

            self._prev_dist[i] = dist
            rewards[f"drone_{i}"] = r

        return rewards

    # ================= SPAWN =================
    def _load_drones(self):
        self.DRONE_IDS = []

        for i in range(self.num_drones):
            angle = 2 * np.pi * i / self.num_drones

            x = 1.5 * np.cos(angle)
            y = 1.5 * np.sin(angle)

            drone = p.createMultiBody(
                baseMass=0.5,
                baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_SPHERE, radius=0.1),
                baseVisualShapeIndex=p.createVisualShape(p.GEOM_SPHERE, radius=0.15),
                basePosition=[x, y, 1.0]
            )

            self.DRONE_IDS.append(drone)

    def _add_obstacles(self):
        self.obstacle_ids = []

        positions = [
            [2.5, 0, 0.4],
            [-2.5, 0, 0.4],
            [0, 1.5, 0.4],
            [1.5, 2.5, 0.4],
        ][:self.num_obstacles]

        for pos in positions:
            obs = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_CYLINDER, radius=0.3, height=0.8),
                baseVisualShapeIndex=p.createVisualShape(p.GEOM_CYLINDER, radius=0.3, length=0.8),
                basePosition=pos
            )
            self.obstacle_ids.append(obs)

    def close(self):
        p.disconnect()
