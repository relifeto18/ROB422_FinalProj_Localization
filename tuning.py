#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import pickle
from KF import KalmanFilter
from PF import ParticleFilter
from models import sensor_model, motion_model
from path_data import get_motion, get_path, draw_path

if __name__ == '__main__':
    
    #initialize plotting        
    plt.ion()
    
    #load in the data
    PIK = "kfdata.dat"
    with open(PIK, "rb") as f:
        noisy_measurement,actions,ground_truth_states,N = pickle.load(f)
        
    motion_input, theta = get_motion()

    #your model parameters are imported here
    A = np.eye(3)
    B = np.eye(3) * 0.1
    C = np.eye(3)

    #we are assuming both the motion and sensor noise is 0 mean
    motion_errors = np.zeros((3,N))
    sensor_errors = np.zeros((3,N))
    for i in range(1,N):
        x_t = np.matrix(ground_truth_states[:,i]).transpose()
        x_tminus1 = np.matrix(ground_truth_states[:,i-1]).transpose()
        u_t = np.matrix(actions[:,i]).transpose()
        z_t = np.matrix(noisy_measurement[:,i]).transpose()
        ###YOUR CODE HERE###
        #use the above variables, as well as A,B, and C (loaded above)...
        #to compute the motion and sensor error
        motion_errors[:,i] = [0,0] #change this
        sensor_errors[:,i] = [0,0] #change this
        
        predicted_x_t = A * x_tminus1 + B * u_t
        motion_errors[:,i] = np.array(x_t - predicted_x_t).flatten()
        
        predicted_z_t = C * x_t
        sensor_errors[:,i] = np.array(z_t - predicted_z_t).flatten()
        ###YOUR CODE HERE###
    
    motion_cov=np.cov(motion_errors)
    sensor_cov=np.cov(sensor_errors)
     
    print("Motion Covariance:")
    print(motion_cov)
    print("Measurement Covariance:")
    print(sensor_cov)
    