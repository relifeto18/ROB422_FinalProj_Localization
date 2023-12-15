import numpy as np

class KalmanFilter:
    def __init__(self, param: list):
        self.A = param['A']
        self.B = param['B']
        self.C = param['C']
        self.Q = param['Q']   # sensor noise
        self.R = param['R']   # motion noise
        self.Sigma = np.eye(3)
        
    def KalmanFilter(self, mu: list, z: list, u: list):

        #prediction step    
        mu_bar = self.A @ mu + self.B @ u
        Sigma_bar = self.A @ self.Sigma @ self.A.T + self.R

        #correction step
        K = Sigma_bar @ self.C.T @ np.linalg.inv(self.C @ Sigma_bar @ self.C.T + self.Q)
        mu_new = mu_bar + K @ (z - self.C @ mu_bar)
        Sigma_new = (np.eye(Sigma_bar.shape[0]) - K @ self.C) @ Sigma_bar

        self.Sigma = Sigma_new   # update sigma
        
        return mu_new, Sigma_new