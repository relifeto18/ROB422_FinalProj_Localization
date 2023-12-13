import numpy as np

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
