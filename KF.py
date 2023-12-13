import pybullet as p
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from models import motion_model, sensor_model

# Define the noise covariances
Q = np.diag([0.01, 0.01, 0.01]) # process noise covariance
R = np.diag([0.1, 0.1, 0.1]) # measurement noise covariance

# Define the initial state and control
x0 = np.array([0, 0, 0]) # initial state
u = np.array([0.5, 0.1]) # constant control
dt = 0.1 # time step

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