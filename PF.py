import pybullet as p
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from utils import get_collision_fn_PR2, load_env, execute_trajectory, draw_sphere_marker
from pybullet_tools.utils import connect, disconnect, get_joint_positions, wait_if_gui, set_joint_positions, joint_from_name, get_link_pose, link_from_name
from pybullet_tools.pr2_utils import PR2_GROUPS


# numParticles = 100

# class PF():
#     def __init__(self, initial_pose, numParticles):
#         pass
#     def action_model(self, odom_pose):
#         pass

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


def particle_filter(X0, W0, u, z, dt, Np):
    # X0: initial particle set (Np x 3 matrix)
    # W0: initial weight vector (Np x 1 vector)
    # u: control vector
    # z: measurement vector
    # dt: time step
    # Np: number of particles
    # returns: updated particle set and weight vector
    
    # Predict step
    X_pred = np.zeros((Np, 3))
    for i in range(Np):
        x = X0[i]
        x_next = motion_model(x, u, dt) + np.random.multivariate_normal([0, 0, 0], Q)
        X_pred[i] = x_next
    
    # Update step
    W_upd = np.zeros(Np)
    for i in range(Np):
        x = X_pred[i]
        p_z = stats.multivariate_normal.pdf(z, x, R)
        W_upd[i] = W0[i] * p_z
    W_upd = W_upd / np.sum(W_upd) # normalize the weights
    
    # Resample step
    X_upd = np.zeros((Np, 3))
    indices = np.random.choice(Np, Np, p=W_upd) # resample with replacement
    for i in range(Np):
        X_upd[i] = X_pred[indices[i]]
    W_upd = np.ones(Np) / Np # reset the weights
    
    return X_upd, W_upd