import time
import numpy as np
import pybullet as p
import pybullet_data as pd
from utils import get_collision_fn_PR2, load_env, execute_trajectory, draw_sphere_marker, draw_line
from pybullet_tools.utils import connect, disconnect, get_joint_positions, wait_if_gui, set_joint_positions, joint_from_name, get_link_pose, link_from_name
from pybullet_tools.pr2_utils import PR2_GROUPS
from path_data import get_motion, get_path, draw_path
from vis import Plot_KF, plot_cov
from KF import KalmanFilter
from PF import ParticleFilter
from models import sensor_model, motion_model
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


param = {
    'dt': 0.1,
    'A': np.eye(3),
    'B': np.eye(3) * 0.1,
    'C': np.eye(3),
    'Q': np.diag([0.5, 0.5, 0.5]),    # sensor noise
    'R': np.diag([0.01, 0.01, 0.01]),    # motion noise
    'Sample_time': 300,
    'Sample_cov': np.diag([0.1, 0.1, 0.1])   # covariance of sampling
}

def main(screenshot=False):
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
    collision_fn = get_collision_fn_PR2(robots['pr2'], base_joints, list(obstacles.values()))
    
    path = get_path()
    draw_path()
    
    KF_error = []
    sensor_error = []
    sensor_data = []
    estimation = []
    draw_KF = False
    plot_cov = False
    dt = 0.1
    KF = KalmanFilter(param)
    Q = KF.Q   # motion noise
    R = KF.R   # sensor noise
    motion_input, theta = get_motion()
    PF = ParticleFilter(param)

    if plot_cov:
        plt.ion()
        plot_axes = plt.subplot(111, aspect='equal')  
        plot_axes.set_xlim([-2.5, 10])  # Adjust these values as needed
        plot_axes.set_ylim([-2.5, 10])

    ################ Kalman Filter ################
    start_time = time.time()
    for i, motion in enumerate(motion_input):
        motion = motion_input[i]
        mu = tuple(get_joint_positions(robots['pr2'], base_joints))

        u = np.array([motion[0]*np.cos(theta[i]), motion[0]*np.sin(theta[i]), motion[-1]], dtype=float)
        z = sensor_model(path[i+1].copy(), Q)
        mu_new, Sigma_new = KF.KalmanFilter(mu, z, u) 
        KF_error.append(path[i+1] - mu_new)
        sensor_error.append(path[i+1] - z)
        sensor_data.append(z)
        estimation.append(mu_new)
        
        # plot cov
        if plot_cov:
            if i%5==0:
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
            plot_axes.set_xlim([min(plot_axes.get_xlim()[0], mu_new[0]-5), max(plot_axes.get_xlim()[1], mu_new[0]+5)])
            plot_axes.set_ylim([min(plot_axes.get_ylim()[0], mu_new[1]-5), max(plot_axes.get_ylim()[1], mu_new[1]+5)])
            plt.draw()
            plt.pause(0.0001)
            # plot_cov(np.array(mu).reshape(-1,1),Sigma_new, plot_axes)
        
        if draw_KF == True:
            draw_sphere_marker((z[0], z[1], 0.15), 0.08, (0, 0, 1, 1))
            draw_sphere_marker((mu_new[0], mu_new[1], 0.1), 0.08, (1, 1, 0, 1))
            
        set_joint_positions(robots['pr2'], base_joints, mu_new)

    print("Kalman Filter run time: ", time.time() - start_time)
    
    if plot_cov:
        plt.ioff()
        plt.show()
    
    # visualize
    kf_err = np.linalg.norm(np.vstack(KF_error.copy()), axis=1)
    sensor_err = np.linalg.norm(np.vstack(sensor_error.copy()), axis=1)
    sensor_data = np.vstack(sensor_data)[:, :2]
    estimation = np.vstack(estimation)[:, :2]
    ground = path[:, :2].copy()
    plot_kf = Plot_KF(len(KF_error), kf_err, sensor_err, sensor_data, estimation, ground)
    plot_kf.show_plots()
    
    
    ################ Particle Filter ################

    for i, motion in enumerate(motion_input):
        u = np.array([float(motion[0]*np.cos(theta[i])), float(motion[0]*np.sin(theta[i])), motion[-1]])
        if i+1 == len(path):
            wait_if_gui()
            print("PF finished")
            break
        z = sensor_model(path[i+1], R)
        pose_estimated = PF.ParticleFilter(u, z, False)

        draw_sphere_marker((pose_estimated[0], pose_estimated[1], 0.1), 0.1, (0, 0, 1, 1))
        set_joint_positions(robots['pr2'], base_joints, pose_estimated)
 

    # Test Path
    # for pa in path:
    #     draw_sphere_marker((pa[0], pa[1], 0.1), 0.1, (1, 1, 0, 1))
    #     set_joint_positions(robots['pr2'], base_joints, pa)

    # Keep graphics window opened
    wait_if_gui()
    disconnect()

if __name__ == '__main__':
    main()