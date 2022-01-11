import numpy as np
from calibration.coefficients import load_coefficients
import cv2
from scipy.spatial import distance as dist
import imutils


def unistort(fr):
    mtx, dist = load_coefficients('./calibration/calibration_chessboard.yml')
    dst = cv2.undistort(fr, mtx, dist, None, None)

    return dst


def midpoint(ptA, ptB):
    return (ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5


# Load an color image in grayscale, threshold
image = cv2.imread('cualquier.png')
h, w, _ = image.shape
print(h)
print(w)
# image = unistort(img)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (3, 3), 0)
canny = cv2.Canny(blurred, 50, 255, 1)

# Find contours
cnts = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]

big_contour = []
area_thresh = 0
for c in cnts:
    area = cv2.contourArea(c)
    if area > area_thresh:
        area_thresh = area
        big_contour = c

# get rotated rectangle from contour
rot_rect = cv2.minAreaRect(big_contour)
box = cv2.boxPoints(rot_rect)
box = np.int0(box)

cX = np.average(box[:, 0])
cY = np.average(box[:, 1])

(tl, tr, br, bl) = box
(tlblX, tlblY) = midpoint(tl, bl)
(trbrX, trbrY) = midpoint(tr, br)

D = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
# refObj = (box, (cX, cY), D / )

print(D)
print(box)
# draw rotated rectangle on copy of img
rot_bbox = image.copy()
cv2.drawContours(rot_bbox, [box], 0, (0, 0, 255), 2)

cv2.imshow('asd', image)
cv2.imshow('canny', canny)
cv2.imshow('image', rot_bbox)
cv2.waitKey(0)
