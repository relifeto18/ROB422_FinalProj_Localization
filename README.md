# ROB 422: Introduction to Algorithmic Robotics Fall 2023 Final Project - Localization

This project implemented a Kalman filter and a particle filter to estimate the robot’s position as it executes the path.

1. run ./install.sh (Make the script executable by running chmod +x install.sh in the terminal.)
2. python3 demo.py
3. You should see PR2 localizing in GUI and two plots of each filter
4. Demo outputs are shown below

## Result

### Kalman Filter

Below are the output results for the Kalman Filter. 

In the video, the red line is the ground truth, the yellow dots are estimations from the Kalman Filter and the blue dots are noisy sensor measurements.

In the picture, the green line is the ground truth, the red dots are estimations from the Kalman Filter and the blue dots are noisy sensor measurements.

![](https://github.com/relifeto18/ROB422_FinalProj_Localization/blob/master/figs/KF/Kalman%20Filter.gif)
![](https://github.com/relifeto18/ROB422_FinalProj_Localization/blob/master/figs/KF/Kalman%20Filter%20Map.png)

### Particle Filter

Below are the output results for the Particle Filter. 

In the video, the red line is the ground truth, the yellow dots are estimations from the Particle Filter and the blue dots are noisy sensor measurements.

In the picture, the green line is the ground truth, the red dots are estimations from the Particle Filter and the blue dots are noisy sensor measurements.

![](https://github.com/relifeto18/ROB422_FinalProj_Localization/blob/master/figs/PF/Particle%20Filter%20Map.png)

### Kalman Filter vs Particle Filter

Below shows the Kalman Filter and Particle Filter under the same scenario. The green line is the ground truth, the blue dots are estimations from the Particle Filter and the red dots are estimations from the Particle Filter.

![](https://github.com/relifeto18/ROB422_FinalProj_Localization/blob/master/figs/KF%20vs%20PF/KF_vs_PF_est.png)
