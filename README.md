# ROB 422: Introduction to Algorithmic Robotics Fall 2023 Final Project - Localization

This project implemented a Kalman filter and a particle filter to estimate the robot’s position as it executes the path.

1. run ./install.sh (Make the script executable by running chmod +x install.sh in the terminal.)
2. python3 demo.py

## Result

### Kalman Filter

Below is the output result for the Kalman Filter. The green line is the ground truth, the red dots are estimations from the Kalman Filter and the blue dots are noisy sensor measurements.

![](https://github.com/relifeto18/ROB422_FinalProj_Localization/blob/master/figs/Kalman%20Filter.gif)
![](https://github.com/relifeto18/ROB422_FinalProj_Localization/blob/master/figs/Kalman%20Filter%20Map.png)
