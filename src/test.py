import cv2 as cv
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from capture.stereo_camera import StereoCamera
from process.pose_detector import MediaPipeDetector
from process.stereo_normal_point_solver import StereoNormalPointSolver


CAM_RES_WIDTH = 368
CAM_RES_HEIGHT = 368

CAM_DIS = 1.53125

# create two named windows
cv.namedWindow("cam1", cv.WINDOW_NORMAL)
cv.namedWindow("cam2", cv.WINDOW_NORMAL)

# move them side by side
cv.moveWindow("cam1", 100, 100)
cv.moveWindow("cam2", 800, 100)  

# instantiate cameras
stereo_cam = StereoCamera((CAM_RES_WIDTH, CAM_RES_HEIGHT))

pose_solver = MediaPipeDetector()
depth_solver = StereoNormalPointSolver(CAM_DIS, (CAM_RES_WIDTH, CAM_RES_HEIGHT), 29)


# Visualization
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# Initial limits of coordinate space
ax.set_xlim([-8, 8])
ax.set_ylim([0, 16])
ax.set_zlim([-8, 8])

# Initial scatter plot
xs = np.zeros(21)
ys = np.zeros(21)
zs = np.zeros(21)
sc = ax.scatter(xs, ys, zs, c='r', s=40)

plt.ion()  # interactive mode ON
plt.show()


while True:
    captured, frame1, frame2 = stereo_cam.get_frame()

    if not captured:
        print("Capture Fail for current loop")
        continue

    rgb_frame1 = cv.cvtColor(frame1, cv.COLOR_BGR2RGB)
    rgb_frame2 = cv.cvtColor(frame2, cv.COLOR_BGR2RGB)
    
    text = "Not Found"

    found_left, found_right, coords_left, coords_right = pose_solver.get_poses(rgb_frame1, rgb_frame2)

    for coord in coords_left:
        cv.circle(frame1, coord, 10, (255, 0, 255), cv.FILLED)
    
    for coord in coords_right:
        cv.circle(frame2, coord, 10, (255, 0, 255), cv.FILLED)

    if found_left and found_right:
        coords3d = np.zeros((21, 3))
        if len(coords_left) == len(coords_right):
            for i in range(len(coords_left)):
                coords3d[i] = depth_solver.get_coordinate(coords_left[i], coords_right[i])
                distance = depth_solver.get_distance_from_cam(coords3d[i])
            text = "Found In Both"
            
            xs = coords3d[:, 0]
            ys = coords3d[:, 2]
            zs = coords3d[:, 1]
            sc._offsets3d = (xs, ys, zs)
            plt.draw()
            plt.pause(0.01)  # 10ms pause to allow the GUI to update

        else:
            print("ERROR, number of landmarks not same")
    elif found_left:
        text = "Found In Left Only"
    elif found_right:
        text = "Found In Right Only"

    cv.putText(frame1, text, (184, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

    cv.imshow('cam1', frame1)
    cv.imshow('cam2', frame2)
    if cv.waitKey(20) & 0xFF==ord('d'):
        break

