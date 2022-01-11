from calibrate_chessboard import calibrate_chessboard
from coefficients import save_coefficients
import cv2
import numpy as np
import os.path
from os import listdir

# Parameters
dir_path = r"C:\Users\dacun\OneDrive\Escritorio\Apps\imgs"
images = listdir(dir_path)
# Iterate through all images
waitTime = 33
for fname in images:
    print(os.path.join(dir_path, fname))
    img = cv2.imread(os.path.join(dir_path, fname), cv2.IMREAD_COLOR)
    # cv2.imshow('',img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('', gray)

    # Wait longer to prevent freeze for videos.
    while True:
        if cv2.waitKey(waitTime) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()


