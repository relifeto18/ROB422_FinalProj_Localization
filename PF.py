import pybullet as p
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from models import motion_model, sensor_model

def ParticleFilter(X0, W0, u, z, dt, Np):
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