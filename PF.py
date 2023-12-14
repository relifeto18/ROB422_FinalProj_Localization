import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from models import motion_model, sensor_model
import random
import copy

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
        self.R = param["R"]
        self.Q = param["Q"]
        self.dt = param["dt"]
        for _ in range(self.sample_times):
            p = particle()
            p.x, p.y, p.theta = np.random.multivariate_normal((0, 0, 0), self.sample_cov, 1)[0]
            p.weight = 1.0 / self.sample_times

    def PF(self, u: list, z: list):
        S = []
        for m in range(self.sample_times):
            next_particle = particle()
            next_sample = motion_model(self.particles[m], u[m], self.dt, self.R)
            next_weight = sensor_model(z[m], self.Q)

            next_particle.x = next_sample[0]
            next_particle.y = next_sample[1]
            next_particle.theta = next_sample[2]
            next_particle.weight = next_weight
            S.append(next_particle)
        
        ## resample
        weight_sum = 0
        new_particles = copy.deepcopy(self.particles)
        r = random.uniform(0, 1.0 / self.sample_times)     
        c = self.particles[0].weight
        i = 0
        for m in range(self.sample_times):
            U = r + m * (1 / self.sample_times)

            while(U > c and i < self.sample_times - 1):
                i += 1
                c += self.particles[i].weight
            new_particles[m] = copy.deepcopy(self.particles[i])
            weight_sum += self.particles[i].weight

        self.particles = new_particles

        for p in self.particles:
            p.weight /= weight_sum
