import pybullet as p
import pybullet_data
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import time
import math
import random


class TurtleBotEnv(gym.Env):

    metadata = {"render_modes": ["human", "none"], "render_fps": 240}

    def __init__(self,
                 render_mode="none",
                 lidar_num_rays=36,
                 lidar_max_dist=3.0,
                 arena_size=4.0,
                 cell_size=0.2,
                 dynamic_obstacles=True):

        super().__init__()

        # Rendering
        self.render_mode = render_mode
        if render_mode == "human":
            self.cid = p.connect(p.GUI)
        else:
            self.cid = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)

        # Store params
        self.lidar_num_rays = lidar_num_rays
        self.lidar_max_dist = lidar_max_dist
        self.arena_size = arena_size
        self.cell_size = cell_size
        self.dynamic_obstacles = dynamic_obstacles

        # ---- Action space ----
        # linear velocity, angular velocity
        self.action_space = spaces.Box(
            low=np.array([-0.3, -1.5], dtype=np.float32),
            high=np.array([0.3, 1.5], dtype=np.float32),
            shape=(2,),
            dtype=np.float32
        )

        # ---- Observation space ----
        # LiDAR + robot pose
        high = np.array(
            [lidar_max_dist] * lidar_num_rays +
            [arena_size, arena_size, np.pi],
            dtype=np.float32
        )
        low = np.array(
            [0.0] * lidar_num_rays +
            [-arena_size, -arena_size, -np.pi],
            dtype=np.float32
        )
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Episode variables
        self.max_steps = 2000
        self.step_count = 0
        self.robot = None
        self.visited = None
        self.obstacles = []

    # ---------------------------------------------------------
    # Differential-drive kinematics
    # ---------------------------------------------------------
    def diff_drive(self, linear, angular,
                   wheel_radius=0.03, wheel_distance=0.30):
        v_l = (linear - (angular * wheel_distance / 2)) / wheel_radius
        v_r = (linear + (angular * wheel_distance / 2)) / wheel_radius
        return v_l, v_r

    # ---------------------------------------------------------
    # Obstacles
    # ---------------------------------------------------------
    def _create_obstacles(self):
        self.obstacles = []
        for i in range(10):
            x = random.uniform(-1.5, 1.5)
            y = random.uniform(-1.5, 1.5)
            r = random.uniform(0.07, 0.25)

            visual = p.createVisualShape(
                p.GEOM_CYLINDER, radius=r, length=0.15, rgbaColor=[1, 0, 0, 1]
            )
            collision = p.createCollisionShape(
                p.GEOM_CYLINDER, radius=r, height=0.15
            )
            body = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=collision,
                baseVisualShapeIndex=visual,
                basePosition=[x, y, 0.075]
            )
            self.obstacles.append((body, x, y, r))

    def _update_dynamic_obstacles(self, t):
        if not self.dynamic_obstacles:
            return
        for i, (body, x0, y0, r) in enumerate(self.obstacles):
            radius = 0.5 + 0.2 * i
            speed = 0.3 + 0.1 * i
            x = x0 + radius * math.cos(t * speed)
            y = y0 + radius * math.sin(t * speed)
            p.resetBasePositionAndOrientation(body, [x, y, 0.075], [0,0,0,1])

    # ---------------------------------------------------------
    # LiDAR
    # ---------------------------------------------------------
    def _get_lidar(self):
        pos, orn = p.getBasePositionAndOrientation(self.robot)
        yaw = p.getEulerFromQuaternion(orn)[2]

        ray_from = []
        ray_to = []

        for i in range(self.lidar_num_rays):
            angle = yaw + (2 * math.pi * i / self.lidar_num_rays)
            start = [pos[0], pos[1], 0.15]
            end = [
                pos[0] + self.lidar_max_dist * math.cos(angle),
                pos[1] + self.lidar_max_dist * math.sin(angle),
                0.15
            ]
            ray_from.append(start)
            ray_to.append(end)

        hits = p.rayTestBatch(ray_from, ray_to)

        dists = []
        for h in hits:
            if h[0] == -1:
                d = self.lidar_max_dist
            else:
                d = h[2] * self.lidar_max_dist
            dists.append(d)

        return np.array(dists, dtype=np.float32)

    # ---------------------------------------------------------
    # Reset
    # ---------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        p.loadURDF("plane.urdf")

        # Load robot
        self.robot = p.loadURDF("turtlebot.urdf", [0, 0, 0.05])

        # Obstacles
        self._create_obstacles()

        # Grid tracking
        grid_n = int((self.arena_size * 2) / self.cell_size)
        self.visited = np.zeros((grid_n, grid_n), dtype=np.int8)

        self.step_count = 0

        obs = self._get_obs()
        info = {}

        return obs, info

    # ---------------------------------------------------------
    # Observation
    # ---------------------------------------------------------
    def _get_obs(self):
        lidar = self._get_lidar()

        pos, orn = p.getBasePositionAndOrientation(self.robot)
        yaw = p.getEulerFromQuaternion(orn)[2]

        return np.concatenate([lidar, [pos[0], pos[1], yaw]]).astype(np.float32)

    # ---------------------------------------------------------
    # Reward
    # ---------------------------------------------------------
    def _compute_reward(self):
        pos, _ = p.getBasePositionAndOrientation(self.robot)
        x, y = pos[0], pos[1]

        gx = int((x + self.arena_size) / self.cell_size)
        gy = int((y + self.arena_size) / self.cell_size)

        reward = 0

        # Visit new grid cell
        if 0 <= gx < self.visited.shape[0] and 0 <= gy < self.visited.shape[1]:
            if self.visited[gx, gy] == 0:
                reward += 1.0
                self.visited[gx, gy] = 1

        # Collision penalty
        if len(p.getContactPoints(self.robot)) > 0:
            reward -= 5.0

        return reward

    # ---------------------------------------------------------
    # Termination
    # ---------------------------------------------------------
    def _terminated(self):
        if self.step_count >= self.max_steps:
            return True
        if len(p.getContactPoints(self.robot)) > 10:
            return True
        return False

    # ---------------------------------------------------------
    # Step
    # ---------------------------------------------------------
    def step(self, action):
        self.step_count += 1

        linear, angular = action
        v_l, v_r = self.diff_drive(linear, angular)

        p.setJointMotorControl2(self.robot, 0, p.VELOCITY_CONTROL, targetVelocity=v_l)
        p.setJointMotorControl2(self.robot, 1, p.VELOCITY_CONTROL, targetVelocity=v_r)

        # simulate physics
        for _ in range(4):
            self._update_dynamic_obstacles(self.step_count * 0.01)
            p.stepSimulation()
            if self.render_mode == "human":
                time.sleep(1/240)

        obs = self._get_obs()
        reward = self._compute_reward()
        terminated = self._terminated()
        truncated = False
        info = {}

        return obs, reward, terminated, truncated, info

    # ---------------------------------------------------------
    # Render
    # ---------------------------------------------------------
    def render(self):
        # GUI handled automatically by PyBullet
        return

    def close(self):
        p.disconnect()
