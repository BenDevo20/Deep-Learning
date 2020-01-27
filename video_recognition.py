"""
program detects faces in videos
"""

import numpy as np
import cv2 as cv2
import matplotlib.pyplot as plt
import argparse
import imutils
from imutils.video import VideoStream
import time

# txt files grabbed from github - fulfills program model specifications
ap = argparse.ArgumentParser()
ap.add_argument('-p', '--prototxt', required=True, help='path to Caffe "Deploy" prototxt file')
ap.add_argument('-m', '--model', required=True, help='path to Caffe pre-trained model')
ap.add_argument('-c', '--confidence', type=float, default=0.5, help='min probability to filter weak detection')
args = vars(ap.parse_args())

# load serialized model from disk
print('[INFO] loading model....')
net = cv2.dnn.readNetFromCaffe(args['prototxt'], args['model'])
# initialize video stream and allow camera sensor to start
print('[INFO] starting the video stream...')
# specify camera index 0 -- first camera detected - can replace with raspberry pi function
vs = VideoStream(src=0).start()
time.sleep(2.0)

# compute face detections with openCV
# loop over frames from video stream
# read a frame -> create blob -> pass through dnn -> obtain face detection
while True:
    # resize frame to have max width 400 pix
    frame = vs.read()
    frame = imutils.resize(frame, width=400)
    # convert frame dimension to blob - wrapper over the data being processed - passed along by caffe
    # blob - n-dimensional array stored in a C-contiguous fashion
    (h,w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame,(300,300)),1.0, (300,300),(104.0, 177.0, 123.0))
    # pass blob through network and obtain detections and predcitions
    net.setInput(blob)
    detections = net.forward()
    # loop over the detections and compare to the confidence threshold and display confidence on screen
    for i in range(0, detections.shape[2]):
        # extract probability within the prediction
        confidence = detections[0,0,i,2]
        # filter out weak detections - ensure the confidence is greater than the min confidence
        if confidence < args['confidence']:
            continue
        # (x,y)-coordinates of the bounding box on object
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype('int')
        # drawing the bounding box with probability of correct detection
        text = '{:.2f}%'.format(confidence*100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
        cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
    # displaying the output frame
    cv2.imshow('Frame', frame)
    # breaking the loop if user presses q
    if KeyboardInterrupt == ord('q'):
        break
# stopping program fully
cv2.destroyAllWindows()
vs.stop()