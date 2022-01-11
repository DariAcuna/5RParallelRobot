from Object import Object
from calibration.coefficients import load_coefficients
from helper import helper
import cv2
import imutils
import serial
import numpy as np
import random
import time


def getFrame():
    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    # Capture frame
    ret, fr = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
    cap.release()

    return fr


def findRatio():

    rectH = 16.5  # cm
    rectW = 27  # cm

    pixH = 668 - 335
    pixW = 931 - 415

    rtX = rectW / pixW
    rtY = rectH / pixH

    return rtX, rtY


def undistort(fr):
    mtx, dist = load_coefficients('./calibration/calibration_chessboard.yml')
    dst = cv2.undistort(fr, mtx, dist, None, None)

    return dst


def setLocations(num):

    # xo = 960    # 0
    # yo = 480    # 480p (rty) 25.12
    # R = 320     # 320p (rty) 16.74
    #
    X = []
    Y = []
    #
    # for i in range(num):
    #     X.append(xo + R * np.cos((i+1) * np.pi / (2 * (num + 1))))
    #     Y.append(yo + R * np.sin((i+1) * np.pi / (2 * (num + 1))))

    X = [621, 1378, 1383, 625]
    Y = [423, 416, 600, 600]

    return X, Y


def sendQueu(X, Y, x0, y0, objs):

    s = serialStuff()

    rx, ry = findRatio()
    r = 27 / 757.5323425438679

    xOrg = x0 * r - 960 * r
    yOrg = y0 * r

    for i, _ in enumerate(objs):

        for j in range(2):

            # 0: released; 1:holding
            grab = j

            # shift to deposit area with object
            if grab:
                xDes = -1 * (X[objs[i].label] * r - 960 * r)
                yDes = Y[objs[i].label] * r
            # shift to object of interest
            else:
                (xDes, yDes) = objs[i].coord
                xDes = -1 * (xDes * r - 960 * r)
                yDes = yDes * r

            # [xDes, yDes, xOrg, yOrg, bool, any]
            send = str([xDes, yDes, xOrg, yOrg, grab, 0])
            print(send)
            sendStuff(send, s)

            xOrg = xDes
            yOrg = yDes

    # head to initial position
    send = str([0, 540 * r, xOrg, yOrg, 0, 0])
    print(send)
    sendStuff(send, s)


def serialStuff():

    ser = serial.Serial()
    ser.port = 'COM4'  # set port
    ser.baudrate = 57600  # to confirm
    ser.bytesize = serial.EIGHTBITS  # set bytesize to eight bits
    ser.timeout = 0
    ser.open()  # open serial port

    return ser


def sendStuff(stuff, arduino):

    time.sleep(1)
    arduino.write(stuff.encode())

    # time.sleep(1)
    # while arduino.in_waiting:
    #
    #     if arduino.in_waiting > 0:
    #         receivedData = arduino.read()
    #         someChar = receivedData.decode()
    #
    #         if someChar == "F":
    #             print('Ready for next movement')
    #             break


with open("model/classes.names", "r", encoding="utf-8") as f:
    labels = f.read().strip().split("\n")

yolo_config_path = "model/yolov4-tiny_test.cfg"
yolo_weights_path = "model/yolov4-tiny_train_best.weights"

useCuda = True

net = cv2.dnn.readNetFromDarknet(yolo_config_path, yolo_weights_path)

if useCuda:
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

frame_width = 1920


if __name__ == '__main__':

    #frame = undistort(getFrame())
    frame = undistort(cv2.imread('reference4.png'))

    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    yolo_width_height = (416, 416)

    frame_resize_width = 1920   # cv2.width?
    confidence_threshold = 0.9
    overlapping_threshold = 0.1

    if frame_resize_width:
        frame = imutils.resize(frame, width=frame_resize_width)
    (H, W) = frame.shape[:2]

    # Construct blob of frames by standardization, resizing, and swapping Red and Blue channels (RBG to RGB)
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, yolo_width_height, swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)

    boxes = []
    confidences = []
    classIDs = []
    p = []

    for output in layerOutputs:

        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if confidence > confidence_threshold:
                # Scale the bboxes back to the original image size
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX)
                y = int(centerY)
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # Remove overlapping bounding boxes
    bboxes = cv2.dnn.NMSBoxes(
        boxes, confidences, confidence_threshold, overlapping_threshold)

    idx = 0
    uniqueIDs = []
    # Get objects' properties
    if len(bboxes) > 0:
        for i in bboxes.flatten():

            (x, y) = (boxes[i][0], boxes[i][1])
            # (w, h) = (boxes[i][2], boxes[i][3])

            # Assign properties and asign an ID
            uniqueIDs.append(classIDs[i])
            p.append(Object(idx, classIDs[i], (x, y)))
            idx = idx + 1

    labelsNum = len(np.unique(uniqueIDs))

    (xOrig, yOrig) = helper(frame)

    (xlist, ylist) = setLocations(labelsNum)
    sendQueu(xlist, ylist, xOrig, yOrig, p)
