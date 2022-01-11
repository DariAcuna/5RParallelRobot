import numpy as np
import serial
import cv2
import time

ser = serial.Serial()
ser.port = 'COM4'                  # set port
ser.baudrate = 9600                # to confirm
ser.bytesize = serial.EIGHTBITS     # set bytesize to eight bits
ser.timeout = 0
ser.open()                          # open serial port

testString = str([1, 2, 3, 4, 5, 6])

while True:

    time.sleep(3)
    ser.write(testString.encode())

    time.sleep(3)
    while ser.in_waiting:

        if ser.in_waiting > 0:
            receivedData = ser.read()
            someChar = receivedData.decode()

            if someChar == "F":
                print('Ready for next movement')
                new = ser.read()
                print(int.from_bytes(new, byteorder='big'))
                break
