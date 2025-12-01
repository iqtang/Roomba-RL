import pybullet as p
import pybullet_data
import time
import math

# Connect to PyBullet
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)

# Load plane and robot
plane = p.loadURDF("plane.urdf")

wall_height = 0.5
wall_thickness = 0.05
arena_size = 2  # half-length of arena

# Collision and visual shapes for walls
wall_collision_x = p.createCollisionShape(p.GEOM_BOX, halfExtents=[wall_thickness, arena_size, wall_height])
wall_collision_y = p.createCollisionShape(p.GEOM_BOX, halfExtents=[arena_size, wall_thickness, wall_height])
wall_visual_x = p.createVisualShape(p.GEOM_BOX, halfExtents=[wall_thickness, arena_size, wall_height], rgbaColor=[1,0,0,1])
wall_visual_y = p.createVisualShape(p.GEOM_BOX, halfExtents=[arena_size, wall_thickness, wall_height], rgbaColor=[1,0,0,1])

# Wall positions: [(x, y, collisionShape, visualShape)]
walls = [
    (arena_size + wall_thickness, 0, wall_collision_x, wall_visual_x),   # +x
    (-arena_size - wall_thickness, 0, wall_collision_x, wall_visual_x),  # -x
    (0, arena_size + wall_thickness, wall_collision_y, wall_visual_y),   # +y
    (0, -arena_size - wall_thickness, wall_collision_y, wall_visual_y)   # -y
]

# Create walls in a loop
for pos in walls:
    p.createMultiBody(baseMass=0, baseCollisionShapeIndex=pos[2], baseVisualShapeIndex=pos[3],
                      basePosition=[pos[0], pos[1], wall_height])

    
robot = p.loadURDF("turtlebot.urdf", [0,0,0.05], globalScaling = .75)
  
# Find joint indices by name
joint_name_to_index = {}
for i in range(p.getNumJoints(robot)):
    info = p.getJointInfo(robot, i)
    joint_name = info[1].decode('utf-8')
    joint_name_to_index[joint_name] = i

left_wheel = joint_name_to_index['wheel_left_joint']
right_wheel = joint_name_to_index['wheel_right_joint']

p.resetDebugVisualizerCamera(
    cameraDistance=5,          # distance above the robot
    cameraYaw=0,               # rotation around Z axis
    cameraPitch=-89,           # straight down
    cameraTargetPosition=[0, 0, 0]  # focus on robot
)

time.sleep(2)
# Move forward for 5 seconds

for _ in range(3*240):
    p.setJointMotorControl2(robot, left_wheel, p.VELOCITY_CONTROL, targetVelocity=50)
    p.setJointMotorControl2(robot, right_wheel, p.VELOCITY_CONTROL, targetVelocity=-50)
    p.stepSimulation()
    time.sleep(1/240)

for _ in range(5*240): 
    p.setJointMotorControl2(robot, left_wheel, p.VELOCITY_CONTROL, targetVelocity=50)
    p.setJointMotorControl2(robot, right_wheel, p.VELOCITY_CONTROL, targetVelocity=50)
    p.stepSimulation()
    
    time.sleep(1/240)

# Turn in place for 3 seconds
for _ in range(3*240):
    p.setJointMotorControl2(robot, left_wheel, p.VELOCITY_CONTROL, targetVelocity=50)
    p.setJointMotorControl2(robot, right_wheel, p.VELOCITY_CONTROL, targetVelocity=-50)
    p.stepSimulation()
    time.sleep(1/240)

p.disconnect()
