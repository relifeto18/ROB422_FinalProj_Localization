import numpy as np
import scipy.stats as stats
from utils import draw_sphere_marker

np.set_printoptions(precision=10)

class ParticleFilter:
    def __init__(self, param: list):
        self.sample_times = param["Sample_time"]
        self.init_cov = param["Sample_cov"]
        self.sensor_cov = param["Q"]  # sensor noise
        self.R = param["R"]   # motion noise
        self.A = param['A']
        self.B = param['B']
        self.init_state = [0.0, 5.5, -1.57]
        # self.init_state = [2.0, 5.5, -1.57]
        self.particles = np.random.multivariate_normal(self.init_state, self.init_cov, self.sample_times)
        self.W = np.ones(self.sample_times) / self.sample_times
        self.count = 0

    def PF_update(self, u, z, draw=False):
        self.count += 1

        # predict
        for i in range(self.sample_times):
            self.particles[i] = self.A @ self.particles[i] + self.B @ u
            self.particles[i] += np.random.multivariate_normal([0, 0, 0], self.R)
        
        # # Euclidean
        # weights_sum = 0
        # for i in range(self.sample_times):
        #     self.W[i] = 1 / np.sqrt((self.particles[i][0] - z[0])**2 + (self.particles[i][1] - z[1])**2)
        #     weights_sum += self.W[i]
        # self.W /= weights_sum   # normalize
        
        # update
        weights_sum = 0
        for i in range(self.sample_times):
            p_z = stats.multivariate_normal.pdf(z, self.particles[i], self.sensor_cov)
            self.W[i] *= p_z
            weights_sum += self.W[i]
            
        self.W /= weights_sum   # normalize
                
        # low_variance_resample
        N = self.sample_times
        X_resampled = np.zeros_like(self.particles)
        cum_sum = np.cumsum(self.W)
        step = 1.0 / N
        position = np.random.uniform(0, step)
        i, count = 0, 0
        for m in range(N):
            while position > cum_sum[i]:
                i += 1
            X_resampled[m] = self.particles[i]
            position += step
            
        self.particles = X_resampled
        self.W = np.ones(N) / N
        
        # draw sampling particals
        if (self.count - 1) % 80 == 0 and draw:
            for particle in self.particles:
                draw_sphere_marker((particle[0], particle[1], 0.1), 0.05, (0, 1, 0, 1))
        
        
        # estimate
        return np.average(self.particles, weights=self.W, axis=0)