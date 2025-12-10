import pybullet as p
import pybullet_data
import time
import math

def build_empty_room():
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)

    plane = p.loadURDF("plane.urdf")

    wall_height = 0.5
    wall_thickness = 0.05
    arena_size = 2  # half-length of arena

    # Collision and visual shapes for walls
    wall_collision_x = p.createCollisionShape(p.GEOM_BOX, halfExtents=[wall_thickness, arena_size, wall_height])
    wall_collision_y = p.createCollisionShape(p.GEOM_BOX, halfExtents=[arena_size, wall_thickness, wall_height])
    wall_visual_x = p.createVisualShape(p.GEOM_BOX, halfExtents=[wall_thickness, arena_size, wall_height], rgbaColor=[.6,.3,0,1])
    wall_visual_y = p.createVisualShape(p.GEOM_BOX, halfExtents=[arena_size, wall_thickness, wall_height], rgbaColor=[.6,.3,0,1])

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


    p.resetDebugVisualizerCamera(
        cameraDistance=5,      
        cameraYaw=0,               
        cameraPitch=-89,         
        cameraTargetPosition=[0, 0, 0] 
    )


def build_obstacle_room():
    build_empty_room()

    box_size = [0.2, 0.2, 0.2] 
    box_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=box_size)
    box_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=box_size, rgbaColor=[0.6, 0.3, 0, 1]) 

    # Center for each quadrant
    d = 0.7  # distance from origin
    obstacle_positions = [
        [ d,  d, box_size[2]],
        [-d,  d, box_size[2]], 
        [-d, -d, box_size[2]], 
        [ d, -d, box_size[2]]  
    ]

    for pos in obstacle_positions:
        p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=box_collision,
            baseVisualShapeIndex=box_visual,
            basePosition=pos
        )