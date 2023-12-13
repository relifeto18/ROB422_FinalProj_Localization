import numpy as np

class KalmanFilter:
    def __init__(self, param: list):
        self.A = param['A']
        self.B = param['B']
        self.C = param['C']
        self.Q = param['Q']   # motion noise
        self.R = param['R']   # sensor noise
        self.Sigma = np.diag(np.zeros(self.A.shape[0]))
        
    ## HW5 code
    def KalmanFilter(self, mu: list, z: list, u: list):

        #prediction step    
        mu_bar = self.A @ mu + self.B @ u
        Sigma_bar = self.A @ self.Sigma @ self.A.T + self.R

        #correction step
        K = Sigma_bar @ self.C.T @ np.linalg.inv(self.C @ Sigma_bar @ self.C.T + self.Q)
        mu_new = mu_bar + K @ (z - self.C @ mu_bar)
        Sigma_new = (np.eye(3) - K @ self.C) @ Sigma_bar

        # mu_new = mu; Sigma_new = Sigma #comment this out to use your code
        ###YOUR CODE HERE###
        return mu_new, Sigma_new