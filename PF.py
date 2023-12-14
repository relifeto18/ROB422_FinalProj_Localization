import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import random
import copy
from utils import draw_sphere_marker
from models import motion_model

class particle:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.theta = 0
        self.weight = 0

class ParticleFilter:
    def __init__(self, param: list):
        self.sample_times = param["Sample_time"]
        self.particles = []
        self.sample_cov = param["Sample_cov"]
        self.sensor_cov = param["R"]
        self.Q = param["Q"]
        self.dt = param["dt"]
        self.A = param['A']
        self.B = param['B']
        self.X = np.random.multivariate_normal([0.0, 5.5, -1.57], self.sample_cov, self.sample_times)
        self.W = np.ones(self.sample_times) / self.sample_times

    def ParticleFilter(self, u: list, z: list, draw=False):
        Np = self.sample_times
        # Predict step
        X_pred = np.zeros((Np, 3))
        for i in range(Np):
            x = self.X[i]
            x_next_noise = np.random.multivariate_normal([0, 0, 0], self.Q)
            x_next = self.A @ x + self.B @ u + x_next_noise
            if draw:
                draw_sphere_marker((x_next[0], x_next[1], 0.05), 0.05, (1, 1, 0, 1))
            X_pred[i] = x_next
        # Update step
        W_pred = np.zeros(Np)
        for i in range(Np):
            x = X_pred[i]
            p_z = stats.multivariate_normal.pdf(z, x, self.sensor_cov)
            W_pred[i] = self.W[i] * p_z
        W_pred = W_pred / np.sum(W_pred) # normalize the weights
        
        self.X = X_pred
        self.W = W_pred
        # # # Resample step
        # X_upd = np.zeros((Np, 3))
        # indices = np.random.choice(Np, Np, p=W_upd) # resample with replacement
        # for i in range(Np):
        #     X_upd[i] = X_pred[indices[i]]
        # W_upd = np.ones(Np) / Np # reset the weights
        # self.X = X_upd
        # self.W = W_upd
        # return np.mean(X_upd, axis=0)

        # sort
        indices = list(range(len(self.W)))
        combined = list(zip(self.W, indices))
        sorted_combined = sorted(combined, key=lambda x: x[0])
        sorted_value, sorted_indices = zip(*sorted_combined)
        self.W = list(sorted_value)[::-1]
        sorted_indices = list(sorted_indices)[::-1]
        self.X = self.X[sorted_indices]
    
        # low var resample
        X_upd = np.zeros((Np, 3))
        W_upd = np.zeros(Np)
        weight_sum = 0
        r = random.uniform(0, 1.0 / Np)     
        c = self.W[0]
        i = 0
        for m in range(Np):
            U = r + m * (1 / Np)
            while(U > c and i < Np - 1):
                i += 1
                c += self.W[i]
            X_upd[m] = self.X[i].copy()
            W_upd[m] = self.W[i].copy()
            weight_sum += self.W[i]
        self.W = W_upd / weight_sum # reset the weights
        self.X = X_upd
        # self.W = W_upd

        x_mean = 0
        y_mean = 0
        cos_theta_mean = 0
        sin_theta_mean = 0

        sum_weight_10 = 0
        for i in range(Np//10):
            sum_weight_10 += self.W[i]

        # calculate pose
        for i in range(Np//10):
            x_mean += self.W[i] * self.X[i][0]/sum_weight_10
            y_mean += self.W[i] * self.X[i][1]/sum_weight_10
            cos_theta_mean += self.W[i] * np.cos(self.X[i][2])/sum_weight_10
            sin_theta_mean += self.W[i] * np.sin(self.X[i][2])/sum_weight_10

        return [x_mean, y_mean, np.arctan2(sin_theta_mean, cos_theta_mean)]
        # return np.mean(self.X, axis=0)

