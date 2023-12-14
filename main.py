import sys
import numpy as np
import pybullet as p
import pybullet_data as pd
from utils import get_collision_fn_PR2, load_env, execute_trajectory, draw_sphere_marker, draw_line
from pybullet_tools.utils import connect, disconnect, get_joint_positions, wait_if_gui, set_joint_positions, joint_from_name, get_link_pose, link_from_name
from pybullet_tools.pr2_utils import PR2_GROUPS
from path_data import get_motion, get_path, draw_path
import time
from KF import KalmanFilter
from PF import ParticleFilter
from models import sensor_model, motion_model

param = {
    'dt': 0.1,
    'A': np.eye(3),
    'B': np.eye(3) * 0.1,
    'C': np.eye(3),
    'Q': np.diag([0.001, 0.001, 0.001]) * 0.000000000000000000001,
    'R': np.diag([0.001, 0.001, 0.001]) * 0.0000000000000000000000001,
    'Sample_time': 100,
    'Sample_cov': np.diag([0.1, 0.1, 0.1])  # covariance of sampling
}

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

    path = get_path()
    draw_path()
    
    dt = 0.1
    KF = KalmanFilter(param)
    Q = KF.Q   # motion noise
    R = KF.R   # sensor noise
    
    motion_input, theta = get_motion()
    
    # # Kalman Filter    
    for i, motion in enumerate(motion_input):
        mu = tuple(get_joint_positions(robots['pr2'], base_joints))
        u = np.array([motion[0]*np.cos(theta[i]), motion[0]*np.sin(theta[i]), motion[-1]], dtype=float)
        z = sensor_model(mu, R)
        mu_new = KF.KalmanFilter(mu, z, u)
        draw_sphere_marker((mu_new[0], mu_new[1], 0.1), 0.1, (1, 1, 0, 1))
        # x_new = motion_model(mu_new, u, dt, Q)
        # draw_sphere_marker((x_new[0], x_new[1], 0.1), 0.1, (1, 1, 0, 1))
        
        # hardcode
        # x_new = np.random.multivariate_normal(mu_new, Q, 1)[0]
        # Execute planned path
        set_joint_positions(robots['pr2'], base_joints, mu_new)
        
    # execute_trajectory(robots['pr2'], base_joints, path, sleep=0.2)
    # Keep graphics window opened
    wait_if_gui()
    disconnect()

if __name__ == '__main__':
    main()