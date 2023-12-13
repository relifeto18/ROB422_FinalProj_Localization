import pybullet as p
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from models import motion_model, sensor_model


# numParticles = 100
# dt = 0.1

# class particle():
#     def __init__(self):
#         self.x = 0
#         self.y = 0
#         self.theta = 0
#         self.weight = 0

# class PF():
#     def __init__(self, initial_pose, numParticles):
#         self.numParticles = numParticles
#         self.particles = []

#         for i in range(self.numParticles):
#             p = particle()
#             p.x = initial_pose[0]
#             p.y = initial_pose[1]
#             p.theta = initial_pose[2]
#             p.weight = 1.0 / self.numParticles
#             self.particles.append(p)

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