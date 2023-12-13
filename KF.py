import pybullet as p
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from utils import get_collision_fn_PR2, load_env, execute_trajectory, draw_sphere_marker
from pybullet_tools.utils import connect, disconnect, get_joint_positions, wait_if_gui, set_joint_positions, joint_from_name, get_link_pose, link_from_name
from pybullet_tools.pr2_utils import PR2_GROUPS

# Define the noise covariances
Q = np.diag([0.01, 0.01, 0.01]) # process noise covariance
R = np.diag([0.1, 0.1, 0.1]) # measurement noise covariance

# Define the initial state and control
x0 = np.array([0, 0, 0]) # initial state
u = np.array([0.5, 0.1]) # constant control
dt = 0.1 # time step

# Define the motion model
def motion_model(x, u, dt):
    # x: state vector [x, y, theta]
    # u: control vector [v, omega]
    # dt: time step
    # returns: next state vector
    x_next = x + np.array([u[0] * np.cos(x[2]) * dt, u[0] * np.sin(x[2]) * dt, u[1] * dt])
    return x_next

# Define the sensor model
def sensor_model(x, sigma):
    # x: state vector [x, y, theta]
    # sigma: standard deviation of the sensor noise
    # returns: noisy measurement of the state vector
    z = x + np.random.normal(0, sigma, 3)
    return z


def kalman_filter(x0, P0, u, z, dt):
    # x0: initial state estimate
    # P0: initial state covariance
    # u: control vector
    # z: measurement vector
    # dt: time step
    # returns: updated state estimate and covariance
    
    # Predict step
    x_pred = motion_model(x0, u, dt)
    F = np.array([[1, 0, -u[0] * np.sin(x0[2]) * dt],
                  [0, 1, u[0] * np.cos(x0[2]) * dt],
                  [0, 0, 1]]) # state transition matrix
    P_pred = F @ P0 @ F.T + Q # predicted state covariance
    
    # Update step
    y = z - x_pred # innovation
    H = np.eye(3) # measurement matrix
    S = H @ P_pred @ H.T + R # innovation covariance
    K = P_pred @ H.T @ np.linalg.inv(S) # Kalman gain
    x_upd = x_pred + K @ y # updated state estimate
    P_upd = (np.eye(3) - K @ H) @ P_pred # updated state covariance
    
    return x_upd, P_upd