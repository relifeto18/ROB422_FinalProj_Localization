import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import random

class ParticleFilter:
    def __init__(self, param: list):
        self.sample_times = param["Sample_time"]
        self.particles = []
        self.sample_cov = param["Sample_cov"]
        self.sensor_cov = param["Q"]  # sensor noise
        self.R = param["R"]   # motion noise
        self.dt = param["dt"]
        self.A = param['A']
        self.B = param['B']
        self.X = np.random.multivariate_normal([0.0, 5.5, -1.57], self.sample_cov, self.sample_times)
        self.W = np.ones(self.sample_times) / self.sample_times
        
    