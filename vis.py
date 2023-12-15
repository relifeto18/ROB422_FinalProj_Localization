import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
   
class Plot_KF:
    def __init__(self, N, kf_error, sensor_err, sensor, estimation, ground):
        self.N = list(range(1, N+1))
        self.kf_err = kf_error
        self.sensor_err = sensor_err
        self.sensor = sensor
        self.est = estimation        
        self.gt = ground 

    def plot_error(self):
        """Plots the first set of data."""
        plt.figure()
        plt.plot(self.N, self.kf_err, label='Kalman filter error', color='blue')
        plt.plot(self.N, self.sensor_err, label='sensor error', color='orange')
        plt.title('Data Error')
        plt.xlabel('Iteration Number')
        plt.ylabel('Error')
        plt.legend()

    def plot_map(self):
        """Plots the second set of data."""
        sensor_x = self.sensor[:, 0]
        sensor_y = self.sensor[:, 1]
        estimation_x = self.est[:, 0]
        estimation_y = self.est[:, 1]
        ground_x = self.gt[:, 0]
        ground_y = self.gt[:, 1]
        
        plt.figure()
        plt.plot(ground_x, ground_y, label='ground truth', linewidth=5, color='limegreen')
        plt.plot(sensor_x, sensor_y, 'o', label='sensor measurement', markersize=5, color='blue')
        plt.plot(estimation_x, estimation_y, 'o', label='estimation', markersize=5, color='red')
        plt.title('Map')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()

    def show_plots(self):
        """Displays both plots."""
        self.plot_error()
        self.plot_map()
        plt.show()
        
class Plot_PF:
    def __init__(self, N, pf_error, sensor_err, sensor, estimation, ground):
        self.N = list(range(1, N+1))
        self.pf_err = pf_error
        self.sensor_err = sensor_err
        self.sensor = sensor
        self.est = estimation        
        self.gt = ground 

    def plot_error(self):
        """Plots the first set of data."""
        plt.figure()
        plt.plot(self.N, self.pf_err, label='Partical filter error', color='blue')
        plt.plot(self.N, self.sensor_err, label='sensor error', color='orange')
        plt.title('Data Error')
        plt.xlabel('Iteration Number')
        plt.ylabel('Error')
        plt.legend()

    def plot_map(self):
        """Plots the second set of data."""
        sensor_x = self.sensor[:, 0]
        sensor_y = self.sensor[:, 1]
        estimation_x = self.est[:, 0]
        estimation_y = self.est[:, 1]
        ground_x = self.gt[:, 0]
        ground_y = self.gt[:, 1]
        
        plt.figure()
        plt.plot(ground_x, ground_y, label='ground truth', linewidth=5, color='limegreen')
        plt.plot(sensor_x, sensor_y, 'o', label='sensor measurement', markersize=5, color='blue')
        plt.plot(estimation_x, estimation_y, 'o', label='estimation', markersize=5, color='red')
        plt.title('Map')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()

    def show_plots(self):
        """Displays both plots."""
        self.plot_error()
        self.plot_map()
        plt.show()
    
def plot_cov(mean, cov, plot_axes):
    
    lambda_, v = np.linalg.eig(cov)
    lambda_ = np.sqrt(lambda_)

    ell = Ellipse(xy=mean,
              width=lambda_[0]*2, height=lambda_[1]*2,
              angle=np.rad2deg(np.arccos(v[0, 0])))
    #ell.set_facecolor('none')
    ell.set_facecolor((1.0, 1.0, 1.0, 0))
    plot_axes.add_artist(ell)
    plt.scatter(mean[0,0],mean[1,0],c='r',s=5)