import numpy as np
import cv2 as cv
import time

cap = cv.VideoCapture(0)
counter = 0
if not cap.isOpened():
    print("Cannot open camera")
    exit()

print('camera opened')
time.sleep(5)
print('entering loop')
while True:
    print('in')
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Display the resulting frame
    cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'):
        cv.imwrite(str(counter) + '.png', frame)
        counter = counter + 1


# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
