from calibrate_chessboard import calibrate_chessboard
from coefficients import save_coefficients
import cv2
import numpy as np
import os.path
from os import listdir

# Parameters
dir_path = r"C:\Users\dacun\PycharmProjects\CRAFT-pytorch\robot\calibration\imgs"
square_size = 1.83
width = 7
height = 11

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(8,6,0)
objp = np.zeros((height * width, 3), np.float32)
objp[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)

objp = objp * square_size

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

images = listdir(dir_path)
# Iterate through all images

for fname in images:
    print(os.path.join(dir_path, fname))
    img = cv2.imread(os.path.join(dir_path, fname), cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (width, height), None)

    # If found, add object points, image points (after refining them)
    if ret:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

img = cv2.imread(os.path.join(dir_path, '1.jpeg'), cv2.IMREAD_COLOR)
graysth = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Calibrate camera
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, graysth.shape[::-1], None, None)

# Save coefficients into a file
save_coefficients(mtx, dist, "calibration_chessboard.yml")
