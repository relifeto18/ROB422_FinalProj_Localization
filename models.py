import numpy as np

# Define the motion model
def motion_model(x, u, dt, noise):
    # x: state vector [x, y, theta]
    # u: control vector [vx, vy, w]
    # dt: time step
    # returns: next state vector
    
    x_next = np.zeros_like(x)
    x_next[0] += x[0] + dt * u[0]
    x_next[1] += x[1] + dt * u[1]
    x_next[2] += x[2] + dt * u[2]
    motion_noise = np.random.multivariate_normal([0, 0, 0], noise)
    x_next += motion_noise
    
    return x_next

# Define the sensor model
def sensor_model(z, noise):
    # x: state vector [x, y, theta]
    # sigma: standard deviation of the sensor noise
    # returns: noisy measurement of the state vector
    
    # sensor_noise = np.random.multivariate_normal([2.0, 5.5, -1.57], noise)
    sensor_noise = np.random.multivariate_normal([0, 0, 0], noise)
    z += sensor_noise
    
    return z
