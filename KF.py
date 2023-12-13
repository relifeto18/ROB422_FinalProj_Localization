import numpy as np

## HW5 code
def KalmanFilter(mu, Sigma, z, u, A, B, C, Q, R):

    #prediction step    
    mu_bar = A @ mu + B @ u
    Sigma_bar = A @ Sigma @ A.T + R

    #correction step
    K = Sigma_bar @ C.T @ np.linalg.inv(C @ Sigma_bar @ C.T + Q)
    mu_new = mu_bar + K @ (z - C @ mu_bar)
    Sigma_new = (np.eye(2) - K @ C) @ Sigma_bar

    # mu_new = mu; Sigma_new = Sigma #comment this out to use your code
    ###YOUR CODE HERE###
    return mu_new, Sigma_new