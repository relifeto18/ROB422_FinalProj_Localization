import sys
import numpy as np
from utils import draw_line

np.set_printoptions(3, suppress=True)
np.set_printoptions(threshold=sys.maxsize)

section1_start = [0.0, 5.5, -1.57]
section1_end = [0.0, -0.5, -1.57]
section2_start = [0.0, -0.5, 0]
section2_end = [10.0, -0.5, 0]
section3_start = [10.0, -0.5, 1.57]
section3_end = [10.0, 2.5, 1.57]
section4_start = [10.0, 2.5, 2.676]
section4_end = [7.0, 4.0, 2.676]
section5_start = [7.0, 4.0, 0.46]
section5_end = [10.0, 5.5, 0.46]
# section4_start = [10.0, 2.5, 2.36]
# section4_end = [7.0, 4.0, 2.36]
# section5_start = [7.0, 4.0, 0.79]
# section5_end = [10.0, 5.5, 0.79]
section6_start = [10.0, 5.5, 1.57]
section6_end = [10.0, 12.0, 1.57]
section7_start = [10.0, 12.0, 0]
section7_end = [21.0, 12.0, 0]
section8_start = [21.0, 12.0, -1.57]
section8_end = [21.0, 8.0, -1.57]
section9_start = [21.0, 8.0, -1.95]
section9_end = [19.0, 3.0, -1.95]
# section9_start = [21.0, 8.0, -2.36]
# section9_end = [19.0, 3.0, -2.36]
section10_start = [19.0, 3.0, -1.57]
section10_end = [19.0, -0.5, -1.57]
section11_start = [19.0, -0.5, 0]
section11_end = [31.0, -0.5, 0]
section12_start = [31.0, -0.5, 1.57]
section12_end = [31.0, 5.5, 1.57]

path = np.array([section1_start, section1_end, section2_start, section2_end, section3_start, section3_end, section4_start,  \
                section4_end, section5_start, section5_end, section6_start, section6_end, section7_start, section7_end, \
                section8_start, section8_end, section9_start, section9_end, section10_start, section10_end, section11_start, \
                section11_end, section12_start, section12_end])


def save_path():
    Traj = []
    Traj.append(path[0].tolist())
    
    for i in range(path.shape[0]-1):
        if (path[i][0] == path[i+1][0]) and (path[i][1] == path[i+1][1]):
            Traj.append(path[i+1].tolist())
        elif (path[i][0] == path[i+1][0]) and (path[i][1] != path[i+1][1]):
            num = int((path[i+1][1] - path[i][1]) / 0.1)
            for i in range(abs(num)):
                if num > 0:
                    new_traj = [round(a + b, 2) for a, b in zip(Traj[-1], [0, 0.1, 0])]
                else:
                    new_traj = [round(a + b, 2) for a, b in zip(Traj[-1], [0, -0.1, 0])]
                Traj.append(new_traj)
        elif (path[i][0] != path[i+1][0]) and (path[i][1] == path[i+1][1]):
            num = int((path[i+1][0] - path[i][0]) / 0.1)
            for i in range(abs(num)):
                if num > 0:
                    new_traj = [round(a + b, 2) for a, b in zip(Traj[-1], [0.1, 0, 0])]
                else:
                    new_traj = [round(a + b, 2) for a, b in zip(Traj[-1], [-0.1, 0, 0])]
                Traj.append(new_traj)
        else:
            x_num = int((path[i+1][0] - path[i][0]) / 0.1)
            y_num = int((path[i+1][1] - path[i][1]) / 0.1)
            
            if x_num < y_num:
                mul = abs(y_num / x_num)
                for i in range(abs(x_num)):
                    if (x_num > 0 and y_num > 0):
                        new_traj = [round(a + b, 2) for a, b in zip(Traj[-1], [0.1, mul*0.1, 0])]
                    elif (x_num > 0 and y_num < 0):
                        new_traj = [round(a + b, 2) for a, b in zip(Traj[-1], [0.1, -mul*0.1, 0])]
                    elif (x_num < 0 and y_num < 0):
                        new_traj = [round(a + b, 2) for a, b in zip(Traj[-1], [-0.1, -mul*0.1, 0])]
                    else:
                        new_traj = [round(a + b, 2) for a, b in zip(Traj[-1], [-0.1, mul*0.1, 0])]
                    Traj.append(new_traj)
            else:
                mul = abs(x_num / y_num)
                for i in range(abs(y_num)):
                    if (x_num > 0 and y_num > 0):
                        new_traj = [round(a + b, 2) for a, b in zip(Traj[-1], [mul*0.1, 0.1, 0])]
                    elif (x_num > 0 and y_num < 0):
                        new_traj = [round(a + b, 2) for a, b in zip(Traj[-1], [mul*0.1, -0.1, 0])]
                    elif (x_num < 0 and y_num < 0):
                        new_traj = [round(a + b, 2) for a, b in zip(Traj[-1], [-mul*0.1, -0.1, 0])]
                    else:
                        new_traj = [round(a + b, 2) for a, b in zip(Traj[-1], [-mul*0.1, 0.1, 0])]
                    Traj.append(new_traj)
                
    with open("Traj.txt", "w") as f:
        for t in Traj:
            f.write(f"{t}\n")
    
def draw_path():
    points = path[::2]
    points = np.vstack((points, path[-1]))
    points[:, -1] = 0.1
    
    for i in range(points.shape[0]-1):        
        line_start = (points[i])
        line_end = (points[i+1])
        line_width = 20
        line_color = (1, 0, 0)
        draw_line(line_start, line_end, line_width, line_color)
        
def get_path():
    data = []

    # Open the file and read line by line
    with open('Traj.txt', 'r') as file:
        for line in file:
            # Convert the line to a NumPy array
            array = np.fromstring(line.strip('[]\n'), sep=',')
            data.append(array)

    # Convert the list of arrays into a 2D array
    robot_path = np.vstack(data)
    
    return robot_path

def save_motion():
    motion = []

    for i in range(path.shape[0] - 1):
        diff = path[i+1] - path[i]
        
        # move y
        if diff[-1] == 0 and diff[0] == 0:     
            num = int(diff[1] / 0.1)  
            new_motion = np.full((abs(num), 2), [1, diff[-1]])
            motion.append(new_motion)
            # if num > 0:
            #     new_motion = np.full((abs(num), 2), [0.1, path[i][-1]])
            #     motion.append(new_motion)
            # if num < 0:
            #     new_motion = np.full((abs(num), 2), [-0.1, path[i][-1]])
            #     motion.append(new_motion)
            
        # move x
        elif diff[-1] == 0 and diff[1] == 0:   
            num = int(diff[0] / 0.1)  
            new_motion = np.full((abs(num), 2), [1, diff[-1]])
            motion.append(new_motion) 
        #     num = int(diff[0] / 0.1)  
        #     if num > 0:
        #         new_motion = np.full((abs(num), 2), [0.1, path[i][-1]])
        #         motion.append(new_motion)
        #     if num < 0:
        #         new_motion = np.full((abs(num), 2), [-0.1, path[i][-1]])
        #         motion.append(new_motion)
        
        # move diag
        elif diff[-1] == 0 and diff[0] != 0 and diff[1] != 0:    
            dis = np.sqrt(diff[0]**2 + diff[1]**2)
            num = int(dis / 0.1)  
            new_motion = np.full((abs(num), 2), [1, diff[-1]])
            motion.append(new_motion)
            # if num > 0:
            #     new_motion = np.full((abs(num), 2), [0.1, path[i][-1]])
            #     motion.append(new_motion)
            # if num < 0:
            #     new_motion = np.full((abs(num), 2), [-0.1, path[i][-1]])
            #     motion.append(new_motion)
            
        # rotate
        elif diff[-1] != 0 and diff[0] == 0 and diff[1] == 0:   
            num = 10
            new_motion = np.full((num, 2), [0, diff[-1]])
            motion.append(new_motion)
        
    
    motion_input = np.vstack(motion)
    
    with open("Motion.txt", "w") as f:
        for t in motion_input:
            f.write(f"{t}\n")
    
def get_motion():
    data = []

    # Open the file and read line by line
    with open('Motion.txt', 'r') as file:
        for line in file:
            # Convert the line to a NumPy array
            array = np.fromstring(line.strip('[]\n'), sep=' ')
            data.append(array)

    # Convert the list of arrays into a 2D array
    robot_motion = np.vstack(data)
    
    return robot_motion
    
# save_motion()