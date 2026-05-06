import os
import numpy as np
import pybullet as p
import pybullet_data
from gymnasium import spaces


class CustomAviaryMADDPG:

    PYB_FREQ = 240
    CTRL_FREQ = 48
    PYB_STEPS_PER_CTRL = PYB_FREQ // CTRL_FREQ
    CTRL_TIMESTEP = 1.0 / CTRL_FREQ

    MAX_SPEED = 1.0
    MAX_YAW_RATE = 1.0
    MAX_BOUND_XY = 5.0
    MAX_HEIGHT = 3.0
    MIN_HEIGHT = 0.2
    GOAL_RADIUS = 3.0
    MAX_STEPS = 300
    PROX_RANGE = 5.0
    ACTION_SMOOTHING = 0.5
    MAX_ACCEL = 2.0

    DRONE_RADIUS = 0.15
    OBS_RADIUS = 0.55

    def __init__(self, num_drones=4, num_obstacles=4, gui=False):

        self.num_drones = num_drones
        self.num_obstacles = num_obstacles
        self.gui = gui

        self.client = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client)
        p.setGravity(0, 0, -9.8, physicsClientId=self.client)
        p.setTimeStep(1.0 / self.PYB_FREQ, physicsClientId=self.client)

        lo = np.array([-1]*3 + [-1]*3 + [-1]*3 + [0] + [-1]*3, dtype=np.float32)
        hi = np.ones(13, dtype=np.float32)

        self.observation_space = spaces.Box(low=lo, high=hi, dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.array([-self.MAX_SPEED]*3 + [-self.MAX_YAW_RATE], dtype=np.float32),
            high=np.array([self.MAX_SPEED]*3 + [self.MAX_YAW_RATE], dtype=np.float32)
        )

        self.DRONE_IDS = []
        self.obstacle_ids = []
        self.goal_pos = None

        self.step_counter = 0
        self.crashed = np.zeros(self.num_drones, dtype=bool)

    def reset(self):
        p.resetSimulation(physicsClientId=self.client)
        p.setGravity(0, 0, -9.8, physicsClientId=self.client)

        p.loadURDF("plane.urdf", physicsClientId=self.client)

        self.DRONE_IDS = []
        self.obstacle_ids = []

        self._loadDrones()
        self._addObstacles()

        self.goal_pos = np.array([
            np.random.uniform(-3, 3),
            np.random.uniform(-3, 3),
            np.random.uniform(0.8, 1.5)
        ])

        self.step_counter = 0
        self.crashed = np.zeros(self.num_drones, dtype=bool)

        return self._computeObs(), {}

    def step(self, actions):

        for i in range(self.num_drones):
            if not self.crashed[i]:
                act = actions.get(f"drone_{i}", np.zeros(4))
                self._applyAction(act, i)

        for _ in range(self.PYB_STEPS_PER_CTRL):
            p.stepSimulation(physicsClientId=self.client)

        self._check_crashes()
        self.step_counter += 1

        obs = self._computeObs()
        rewards = self._computeRewards()
        terminated = self._computeTerminated()
        truncated = self._computeTruncated()

        return obs, rewards, terminated, truncated, {}

    def _applyAction(self, action, idx):
        action = np.clip(action, -1, 1)

        p.resetBaseVelocity(
            self.DRONE_IDS[idx],
            linearVelocity=action[:3].tolist(),
            angularVelocity=[0, 0, action[3]],
            physicsClientId=self.client
        )

    def _computeObs(self):
        obs = {}
        for i in range(self.num_drones):
            pos, _ = p.getBasePositionAndOrientation(self.DRONE_IDS[i], physicsClientId=self.client)
            vel, _ = p.getBaseVelocity(self.DRONE_IDS[i], physicsClientId=self.client)

            rel_goal = (self.goal_pos - np.array(pos)) / 5.0

            obs[f"drone_{i}"] = np.concatenate([
                np.array(pos)/5.0,
                np.array(vel),
                np.zeros(3),
                np.array([0.0]),
                rel_goal
            ]).astype(np.float32)

        return obs

    def _computeRewards(self):
        return {f"drone_{i}": 0.0 for i in range(self.num_drones)}

    def _computeTerminated(self):
        done = False

        # SUCCESS
        if all(
            np.linalg.norm(
                np.array(p.getBasePositionAndOrientation(self.DRONE_IDS[i], physicsClientId=self.client)[0])
                - self.goal_pos
            ) <= self.GOAL_RADIUS
            for i in range(self.num_drones)
        ):
            done = True

        return {f"drone_{i}": done for i in range(self.num_drones)}

    def _computeTruncated(self):
        done = self.step_counter >= self.MAX_STEPS
        return {f"drone_{i}": done for i in range(self.num_drones)}

    def _loadDrones(self):
        for i in range(self.num_drones):
            pos = [np.cos(i), np.sin(i), 1]
            drone = p.createMultiBody(
                baseMass=1,
                basePosition=pos,
                physicsClientId=self.client
            )
            self.DRONE_IDS.append(drone)

    def _addObstacles(self):
        pass

    def _check_crashes(self):
        pass

    def close(self):
        p.disconnect(self.client)
