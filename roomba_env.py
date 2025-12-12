import gymnasium as gym
from gymnasium import spaces
from build_world import build_empty_room, build_obstacle_room
import numpy as np
import pybullet as p
import pybullet_data
import time
import math

GLOBAL_SCALING = 1
WHEEL_RADIUS = 0.0352 * GLOBAL_SCALING
WHEEL_BASE   = 0.23   * GLOBAL_SCALING

NUM_LASER_RAYS = 24
MAX_LASER_RANGE = 3.0

class RoombaEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, world_type="empty", gui=True):
        super().__init__()

        self.gui = gui
        self.world_type = world_type
        self.physics_client = None
        self.robot = None
        self.left_wheel = None
        self.right_wheel = None

        self.sim_time = 0.0

        self.arena_half_size = 2 
        self.grid_size = 0.5  # meters per cell
        self.grid_width = int((self.arena_half_size*2) / self.grid_size)
        self.grid_height = int((self.arena_half_size*2) / self.grid_size)
        self.visited_grid = np.zeros((self.grid_width, self.grid_height), dtype=np.bool_)

        self.max_coverage = 0.0

        self.reward_new_area = 10
        self.reward_collision = -10
        self.reward_timestep = -0.01

        self.step_count = 0
        self.collision_count = 0
        self.max_steps = 5000
        self.max_collisions = 500

        self.dt = 1 / 240  

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32
        )


        obs_high = np.array(
            [2.0, 2.0, np.pi, 1.5, 1.5] + [MAX_LASER_RANGE]*NUM_LASER_RAYS + [1.0]*(self.grid_width*self.grid_height),
            dtype=np.float32
        )
        obs_low  = np.array(
            [-2.0, -2.0, -np.pi, .15, -1.5] + [0.0]*NUM_LASER_RAYS + [0.0]*(self.grid_width*self.grid_height),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)

        self.reset()


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.step_count = 0
        self.collision_count = 0
        self.sim_time = 0.0

        if self.physics_client is None:
            self.physics_client = p.connect(p.GUI if self.gui else p.DIRECT)
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            p.setGravity(0, 0, -9.8)
        else:
   
            p.resetSimulation()
            p.setGravity(0, 0, -9.8)

        self.visited_grid.fill(False)

        
        if self.world_type == "empty":
            build_empty_room()
        elif self.world_type == "obstacle":
            build_obstacle_room()
        elif self.world_type == "carousel":
            self.build_carousel_room()
        else:
            raise ValueError(f"Unknown world type {self.world_type}")


        self.robot = p.loadURDF("turtlebot.urdf", [0, 0, 0.05], globalScaling=0.75)
        self._get_wheel_joints()

        p.resetDebugVisualizerCamera(
            cameraDistance=5, cameraYaw=0, cameraPitch=-89, 
            cameraTargetPosition=[0, 0, 0]
        )   

        self.draw_grid()

        return self._get_obs(), {}


    def _cmd_vel_to_wheel_vel(self, v, omega):
        # Diff drive equations
        v_left  = (v - (omega * WHEEL_BASE / 2)) / WHEEL_RADIUS
        v_right = (v + (omega * WHEEL_BASE / 2)) / WHEEL_RADIUS
        return v_left, v_right


    def step(self, action):
        self.sim_time += self.dt
        self.step_count += 1

        done = False
        
        v     = np.interp(action[0], [-1, 1], [-1.5, 1.5])
        omega = np.interp(action[1], [-1, 1], [-1.5, 1.5])

        # Convert to wheel speeds
        v_left, v_right = self._cmd_vel_to_wheel_vel(v, omega)


        p.setJointMotorControl2(self.robot, self.left_wheel,
                                p.VELOCITY_CONTROL, targetVelocity=v_left, force=2)
        p.setJointMotorControl2(self.robot, self.right_wheel,
                                p.VELOCITY_CONTROL, targetVelocity=v_right, force=2)
        
        if self.world_type == "carousel" and hasattr(self, "carousel_ids"):
            self.step_carousel()

        p.stepSimulation()

        '''if self.gui:
            time.sleep(self.dt)  # Training vs Rendering'''

        obs = self._get_obs()

        r_explore = 0.0
        
        i, j = self._pos_to_grid(obs[:2])
        new_cell = False

        if not self.visited_grid[i, j]:
            r_explore += self.reward_new_area
            self.visited_grid[i, j] = True
            new_cell = True
        else:
            r_explore += -1


    
        lidar = obs[5: 5 + NUM_LASER_RAYS]  # LIDAR readings

        r_collision = 0.0
        collision = False

        min_front = np.min(lidar)


        if min_front < 0.30:
            r_collision += -5.0 * (0.30 - min_front) / .30

        if min_front < 0.15:
            collision = True
            self.collision_count += 1
            r_collision += self.reward_collision


        if self.collision_count >= self.max_collisions:
            r_collision += -100

        reward = r_explore + r_collision + self.reward_timestep

        if self.step_count >= self.max_steps or self.collision_count >= self.max_collisions:
            done = True
            print(f"NUM collisions = {self.collision_count}")


        info = {
            "wheel_left": v_left,
            "wheel_right": v_right,
            "new_cell": new_cell,
            "collision": collision,
            "coverage": np.sum(self.visited_grid) / (self.grid_width * self.grid_height)
        }

        #self.draw_visited_cells()

        return obs, reward, done, False, info

    def _get_obs(self, visualize_lidar=False):
        pos, orn = p.getBasePositionAndOrientation(self.robot)
        yaw = p.getEulerFromQuaternion(orn)[2]

        lin_vel, ang_vel = p.getBaseVelocity(self.robot)
        v = np.linalg.norm(lin_vel[:2])
        omega = ang_vel[2]

        lidar = []
        if not hasattr(self, "lidar_lines"):
            self.lidar_lines = [None]*NUM_LASER_RAYS

        for i in range(NUM_LASER_RAYS):
            ray_angle = yaw + i * 2*np.pi / NUM_LASER_RAYS
            from_pos = pos + np.array([0,0,0.05])
            to_pos = from_pos + MAX_LASER_RANGE * np.array([np.cos(ray_angle), np.sin(ray_angle), 0])
            ray = p.rayTest(from_pos, to_pos)[0]
            hit_fraction = ray[2]
            distance = hit_fraction * MAX_LASER_RANGE
            distance = np.clip(distance, 0.0, MAX_LASER_RANGE)
            lidar.append(distance)

            if visualize_lidar and self.gui: # GUI true or false
                hit_pos = from_pos + hit_fraction * (to_pos - from_pos)
                
                
                if self.lidar_lines[i] is not None:
                    try:
                        p.removeUserDebugItem(self.lidar_lines[i])
                    except Exception:
                        pass  
                    self.lidar_lines[i] = None
                
                
                try:
                    self.lidar_lines[i] = p.addUserDebugLine(
                        from_pos, hit_pos, [0,0,1], lineWidth=2, lifeTime=self.dt*5
                    )
                except Exception:
                    self.lidar_lines[i] = None  

        visited_flat = self.visited_grid.astype(np.float32).flatten()

        return np.array([pos[0], pos[1], yaw, v, omega] + lidar + visited_flat.tolist(), dtype=np.float32)



    def _get_wheel_joints(self):
        joint_name_to_index = {}
        for i in range(p.getNumJoints(self.robot)):
            info = p.getJointInfo(self.robot, i)
            name = info[1].decode('utf-8')
            joint_name_to_index[name] = i

        self.left_wheel = joint_name_to_index['wheel_left_joint']
        self.right_wheel = joint_name_to_index['wheel_right_joint']


    def build_carousel_room(self):
        build_empty_room()
        cyl_radius = 0.2
        cyl_height = 0.4
        cyl_collision = p.createCollisionShape(p.GEOM_CYLINDER, radius=cyl_radius, height=cyl_height)
        cyl_visual = p.createVisualShape(p.GEOM_CYLINDER, radius=cyl_radius,
                                         length=cyl_height, rgbaColor=[1,0,0,1])

        num_cyls = 4
        r = 0.7
        self.carousel_ids = []

        for i in range(num_cyls):
            angle = i * 2*np.pi / num_cyls
            x = r * np.cos(angle)
            y = r * np.sin(angle)
            cid = p.createMultiBody(0, cyl_collision, cyl_visual,
                                    basePosition=[x, y, cyl_height/2])
            self.carousel_ids.append(cid)

    def step_carousel(self):
        r = 1.25
        speed = .3
        for i, cid in enumerate(self.carousel_ids):
            angle = i * 2*np.pi/len(self.carousel_ids) + speed*self.sim_time
            x = r * np.cos(angle)
            y = r * np.sin(angle)
            pos, orn = p.getBasePositionAndOrientation(cid)
            p.resetBasePositionAndOrientation(cid, [x, y, pos[2]], orn)


    def _pos_to_grid(self, pos):
        x = pos[0]
        y = pos[1]
        i = int((x + self.arena_half_size) / self.grid_size)
        j = int((y + self.arena_half_size) / self.grid_size)
        i = np.clip(i, 0, self.grid_width - 1)
        j = np.clip(j, 0, self.grid_height - 1)
        return i, j
    
    def _grid_to_pos(self, i, j):
        x = -self.arena_half_size + (i + 0.5) * self.grid_size
        y = -self.arena_half_size + (j + 0.5) * self.grid_size
        return x, y
    
    def draw_grid(self):
        cell = self.grid_size
        half = self.arena_half_size

        if hasattr(self, "grid_line_ids"):
            for line in self.grid_line_ids:
                p.removeUserDebugItem(line)

        self.grid_line_ids = []

        x_vals = np.arange(-half, half + cell, cell)
        for x in x_vals:
            line_id = p.addUserDebugLine(
                [x, -half, 0.01],
                [x,  half, 0.01],
                lineColorRGB=[0.6, 0.6, 0.6],
                lineWidth=1.0
            )
            self.grid_line_ids.append(line_id)

        y_vals = np.arange(-half, half + cell, cell)
        for y in y_vals:
            line_id = p.addUserDebugLine(
                [-half, y, 0.01],
                [ half, y, 0.01],
                lineColorRGB=[0.6, 0.6, 0.6],
                lineWidth=1.0
            )

    def draw_visited_cells(self):
        cell = self.grid_size
        half = self.arena_half_size
        z = 0.015

        if hasattr(self, "cell_mark_ids"):
            for cid in self.cell_mark_ids:
                p.removeUserDebugItem(cid)
        self.cell_mark_ids = []

        for i in range(self.grid_width):
            for j in range(self.grid_height):
                if self.visited_grid[i, j]:
                    x, y = self._grid_to_pos(i, j)
                    cid = p.addUserDebugLine(
                        [x - cell/2, y - cell/2, z],
                        [x + cell/2, y + cell/2, z],
                        lineColorRGB=[0, 1, 0],
                        lineWidth=5
                    )
                    self.cell_mark_ids.append(cid)

    def render(self, mode="human"):
        pass

    def close(self):
        if self.physics_client is not None:
            p.disconnect(self.physics_client)
            self.physics_client = None

if __name__ == "__main__":
    import time
    import numpy as np


    env = RoombaEnv(world_type="obstacle", gui=True)


    obs, _ = env.reset()
    print("Initial observation:", obs)

    collision_count = 0

    linear_speed = 2   
    angular_speed = .2 

    for step in range(5000):
        action = np.array([linear_speed, angular_speed], dtype=np.float32)
        obs, reward, done, truncated, info = env.step(action)

        min_distance = min(obs[5: 5 + NUM_LASER_RAYS]) 
        if min_distance < .15:
            collision_count += 1

    print(f"Total collisions: {collision_count}")
    env.close()
