import time
import numpy as np
import pybullet as p
import pybullet_data as pd
from utils import get_collision_fn_PR2, load_env, execute_trajectory, draw_sphere_marker
from pybullet_tools.utils import connect, disconnect, get_joint_positions, wait_if_gui, set_joint_positions, joint_from_name
from pybullet_tools.pr2_utils import PR2_GROUPS
from path_data import get_motion, get_path, draw_path
from vis import Plot_KF, Plot_PF, plot_compare
from KF import KalmanFilter
from PF import ParticleFilter
from models import sensor_model
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

np.set_printoptions(precision=20)

param = {
    'dt': 0.1,
    'A': np.eye(3),
    'B': np.eye(3) * 0.1,
    'C': np.eye(3),
    'Q': np.diag([0.05, 0.05, 0.05]),    # sensor noise
    'R': np.diag([0.05, 0.05, 0.05]),    # motion noise
    'Sample_time': 100,
    'Sample_cov': np.diag([0.2, 0.2, 0.2])   # covariance of initial sampling
}

def main(filter="KF"):
    # initialize parameters    
    draw_KF = False
    draw_PF = False
    draw_sample = False
    plot_cov = False
    
    KF_error = []
    KF_sensor_error = []
    KF_sensor_data = []
    KF_estimation = []
    PF_error = []
    PF_sensor_error = []
    PF_sensor_data = []
    PF_estimation = []
    particles = []
        
    KF = KalmanFilter(param)
    PF = ParticleFilter(param)
    Q = param["Q"]   # sensor noise
    motion_input, theta = get_motion()
    
    print("\nPlease close all the plots and press enter to continue.")
    print("Loading Kalman Filter ...\n")
    time.sleep(2)
    

    ################ Kalman Filter ################
    if filter == "KF":
        # initialize PyBullet
        connect(use_gui=True)
        # load robot and obstacle resources
        p.setAdditionalSearchPath(pd.getDataPath())
        planeId = p.loadURDF('plane.urdf', useFixedBase=True)
        p.changeDynamics(planeId, -1, lateralFriction=0.99)
        # set camera
        p.resetDebugVisualizerCamera(cameraDistance=13.5,
                                        cameraYaw=0,
                                        cameraPitch=-89,
                                        cameraTargetPosition=[15, 5, 0])
        robots, obstacles = load_env('rob.json')
        
        # define active DoFs
        base_joints = [joint_from_name(robots['pr2'], name) for name in PR2_GROUPS['base']]
        
        path = get_path()
        draw_path()
        ground = path[:, :2].copy()
    
        print("\nKalman Filter running: expected 4mins to complete drawing.")
        print("The sensor measurements: blue dots.\nThe filter estimations: yellow dots.\nThe ground truth: red line.")
        
        # initialize covariance plot
        if plot_cov:
            plt.ion()
            plot_axes = plt.subplot(111, aspect='equal')  
            plot_axes.set_xlim([-2.5, 10])
            plot_axes.set_ylim([-2.5, 10])
        
        start_time = time.time()
        
        for i, motion in enumerate(motion_input):
            motion = motion_input[i]
            mu = tuple(get_joint_positions(robots['pr2'], base_joints))   # current state
            u = np.array([float(motion[0]*np.cos(theta[i])), float(motion[0]*np.sin(theta[i])), motion[-1]])   # control input
            z = sensor_model(path[i+1].copy(), Q)   # noisy measurement
            mu_new, Sigma_new = KF.KalmanFilter(mu, z, u)   # Kalman Filter
            
            KF_error.append(path[i+1] - mu_new)
            KF_sensor_error.append(path[i+1] - z)
            KF_sensor_data.append(z)
            KF_estimation.append(mu_new)
            
            # plot cov
            if plot_cov:
                if i%8==0:
                    x = np.array(mu).reshape(-1,1)
                    lambda_, v = np.linalg.eig(Sigma_new)
                    lambda_ = np.sqrt(lambda_)
                    angle = np.rad2deg(np.arctan2(v[1, 0], v[0, 0]))  # Correct angle calculation
                    ell = Ellipse(xy=mu_new,
                                width=lambda_[0] * 2, height=lambda_[1] * 2,
                                angle=angle)
                    ell.set_facecolor((1.0, 1.0, 1.0, 0))
                    ell.set_edgecolor('k')
                    plot_axes.add_artist(ell)
                    plt.scatter(x[0,0],x[1,0],c='r',s=5)
                # adaptive screen
                plot_axes.set_xlim([min(plot_axes.get_xlim()[0], mu_new[0]-5), max(plot_axes.get_xlim()[1], mu_new[0]+5)])
                plot_axes.set_ylim([min(plot_axes.get_ylim()[0], mu_new[1]-5), max(plot_axes.get_ylim()[1], mu_new[1]+5)])
                plt.draw()
                plt.pause(0.0001)
            
            # draw sensor and estimation data
            if draw_KF == True:
                draw_sphere_marker((z[0], z[1], 0.15), 0.08, (0, 0, 1, 1))   # sensor measurement in blue
                draw_sphere_marker((mu_new[0], mu_new[1], 0.1), 0.08, (1, 1, 0, 1))   # estimation in yellow
            
            # set the robot to estimated state
            set_joint_positions(robots['pr2'], base_joints, mu_new)

        print("Kalman Filter run time: ", time.time() - start_time)
        
        # close the plot
        if plot_cov:
            plt.ioff()
            plt.show()

        print("Plot Kalman Filter ...\n")
        # visualize
        kf_err = np.linalg.norm(np.vstack(KF_error.copy()), axis=1)
        kf_sensor_err = np.linalg.norm(np.vstack(KF_sensor_error.copy()), axis=1)
        kf_sensor_data = np.vstack(KF_sensor_data)[:, :2]
        kf_estimation = np.vstack(KF_estimation)[:, :2]
        plot_kf = Plot_KF(len(kf_err), kf_err, kf_sensor_err, kf_sensor_data, kf_estimation, ground)
        plot_kf.show_plots()
        
        # print("KF error mean: ", np.mean(kf_err))
        # print("Sensor error mean: ", np.mean(kf_sensor_err))

        wait_if_gui()
        disconnect()
    
    filter = "PF"
    print("\nLoading Particle Filter ...\n")
    time.sleep(2)
    
    ################ Particle Filter ################
    if filter == "PF":
        # initialize PyBullet
        connect(use_gui=True)
        # load robot and obstacle resources
        p.setAdditionalSearchPath(pd.getDataPath())
        planeId = p.loadURDF('plane.urdf', useFixedBase=True)
        p.changeDynamics(planeId, -1, lateralFriction=0.99)
        # set camera
        p.resetDebugVisualizerCamera(cameraDistance=13.5,
                                        cameraYaw=0,
                                        cameraPitch=-89,
                                        cameraTargetPosition=[15, 5, 0])
        robots, obstacles = load_env('rob.json')
        
        # define active DoFs
        base_joints = [joint_from_name(robots['pr2'], name) for name in PR2_GROUPS['base']]
        
        path = get_path()
        draw_path()
        ground = path[:, :2].copy()
        
        print("\nParticle Filter running: expected 4-5mins to complete drawing.")
        print("The sampling particles: green dots.\nThe filter estimations: blue dots.\nThe ground truth: red line.")
        
        start_time = time.time()
        
        for i, motion in enumerate(motion_input):
            u = np.array([float(motion[0]*np.cos(theta[i])), float(motion[0]*np.sin(theta[i])), motion[-1]])   # control input
            z = sensor_model(path[i+1].copy(), Q)   # noisy measurement
            pose_estimated, particles = PF.PF_update(u, z, draw=draw_sample)   # Partical Filter
            PF_error.append(path[i+1] - pose_estimated)
            PF_sensor_error.append(path[i+1] - z)
            PF_sensor_data.append(z)
            PF_estimation.append(pose_estimated)
                    
            # draw sampling and estimation data
            if draw_PF:
                draw_sphere_marker((pose_estimated[0], pose_estimated[1], 0.1), 0.08, (0, 0, 1, 1))

            # set the robot to estimated state
            set_joint_positions(robots['pr2'], base_joints, pose_estimated)
        
        print("Particle Filter run time: ", time.time() - start_time) 
        
        print("Plot Particle Filter ...\n")
        # visualize
        pf_err = np.linalg.norm(np.vstack(PF_error.copy()), axis=1)
        pf_sensor_err = np.linalg.norm(np.vstack(PF_sensor_error.copy()), axis=1)
        pf_sensor_data = np.vstack(PF_sensor_data)[:, :2]
        pf_estimation = np.vstack(PF_estimation)[:, :2]
        pf_particles = np.vstack(particles)[:, :2]
        plot_pf = Plot_PF(len(pf_err), pf_err, pf_sensor_err, pf_sensor_data, pf_estimation, pf_particles, ground)
        plot_pf.show_plots()
        
        # print("PF error mean: ", np.mean(pf_err))
        # print("Sensor error mean: ", np.mean(pf_sensor_err))

        wait_if_gui()
        disconnect()
    
    print("\nPlot KF vs PF ...")
    # plot KF vs PF data
    kf_err = np.linalg.norm(np.vstack(KF_error.copy()), axis=1)
    kf_estimation = np.vstack(KF_estimation)[:, :2]
    pf_err = np.linalg.norm(np.vstack(PF_error.copy()), axis=1)
    pf_estimation = np.vstack(PF_estimation)[:, :2]
    N = len(kf_err)
    num = list(range(1, N+1))
    plot_compare(num, kf_err, pf_err, kf_estimation, pf_estimation, ground)
    
    print("\n================================================")
    print("KF Performance ...")
    # print("KF collision times: ", KF_collision)
    print("KF error mean: ", np.mean(kf_err))
    print("KF Sensor error mean: ", np.mean(kf_sensor_err))
    
    print("\n================================================")
    print("PF Performance ...")
    # print("PF collision times: ", PF_collision)
    print("PF error mean: ", np.mean(pf_err))
    print("PF Sensor error mean: ", np.mean(pf_sensor_err))
    
    time.sleep(2)
    print("\nBye!")

if __name__ == '__main__':
    main()