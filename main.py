import sys
import numpy as np
import pybullet as p
import pybullet_data as pd
from utils import get_collision_fn_PR2, load_env, execute_trajectory, draw_sphere_marker, draw_line
from pybullet_tools.utils import connect, disconnect, get_joint_positions, wait_if_gui, set_joint_positions, joint_from_name, get_link_pose, link_from_name
from pybullet_tools.pr2_utils import PR2_GROUPS
import path_data
import time
from KF import kalman_filter
from PF import particle_filter
from models import sensor_model, motion_model

def main(screenshot=False):
    # initialize PyBullet
    connect(use_gui=True)
    # load robot and obstacle resources
    p.setAdditionalSearchPath(pd.getDataPath())
    planeId = p.loadURDF('plane.urdf', useFixedBase=True)
    p.changeDynamics(planeId, -1, lateralFriction=0.99)
    # set camera
    p.resetDebugVisualizerCamera(cameraDistance=13.5,
                                    cameraYaw=0,
                                    cameraPitch=-89,
                                    cameraTargetPosition=[15, 5, 0])
    robots, obstacles = load_env('rob.json')
    
    # define active DoFs
    base_joints = [joint_from_name(robots['pr2'], name) for name in PR2_GROUPS['base']]
    collision_fn = get_collision_fn_PR2(robots['pr2'], base_joints, list(obstacles.values()))

    path = path_data.get_path()
    path_data.draw_path()
    
    control_input = np.diff(path, axis=0)   # u
    u = control_input[0]   # u0
    x = tuple(get_joint_positions(robots['pr2'], base_joints))   # x0
    
    x_upd, P_upd = kalman_filter()
    X_upd, W_upd = particle_filter()
    
    for pth in path:
        current_state = tuple(get_joint_positions(robots['pr2'], base_joints))   # x
        sensor_model(current_state)
        set_joint_positions(robots['pr2'], base_joints, pth)
        
    # Execute planned path
    # execute_trajectory(robots['pr2'], base_joints, path, sleep=0.2)
    # Keep graphics window opened
    wait_if_gui()
    disconnect()

if __name__ == '__main__':
    main()