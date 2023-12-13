import numpy as np
from noise import gaussian_noise

# Define the motion model
def motion_model(x, u, dt, noise):
    # x: state vector [x, y, theta]
    # u: control vector [v, omega]
    # dt: time step
    # returns: next state vector
    x_next = np.zeros_like(x)
    x_next[0] += x[0] + dt * u[0] * np.cos(x[2])
    x_next[1] += x[1] + dt * u[0] * np.sin(x[2])
    x_next[2] += x[2] + dt * u[1]
    x_next = gaussian_noise(x_next, noise)
    return x_next

# Define the sensor model
def sensor_model(z, noise):
    # x: state vector [x, y, theta]
    # sigma: standard deviation of the sensor noise
    # returns: noisy measurement of the state vector
    z = gaussian_noise(z, noise)
    return z
