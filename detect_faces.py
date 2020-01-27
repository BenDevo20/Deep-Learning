import numpy as np
import argparse
import cv2

"""
 construct the argument parse and parse arguments 
 image - path to input image
 prototxt - the path to the Caffe prototxt file
 model - the path to the pretrained caffe model 
 confidence interval of 95% - optional implication  
"""
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='path to input image')
ap.add_argument('-p', '--prototxt', required=True, help="path to Caffe 'deploy' prototxt file")
ap.add_argument('-m', '--model', required=True, help='path to Caffee pre-trained model')
ap.add_argument('-c', '--confidence', type=float, default=0.5, help='minimum probability to filter weak detections')
args = vars(ap.parse_args())

#dnn.blobFromImage - pre-processing includes setting the blob dimensions and normalization
print('[INFO] loading model....')
net = cv2.dnn.readNetFromCaffe(args['prototxt'], args['model'])
image = cv2.imread(args['image'])
(h,w) = image.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(image,(300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

# pass the blob through the network and obtain the detections and predictions
print('[INFO] computing object detections....')
net.setInput(blob)
detections = net.forward()

# loop over the detections and draw boxes around the detected faces
for i in range(0, detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    # filter out weak detection- ensuring the confidence is greater than the minimum confidence
    if confidence > args['confidence']:
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype('int')
        # draw bounding box of the face along with the associated probability
        text = '{:.2f}%'.format(confidence*100)
        y = startY-10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(image,(startX,startY), (endX, endY), (0, 0, 255), 2)
        cv2.putText(image, text, (startX,y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
# show the output image
cv2.imshow('Output', image)
cv2.waitKey(0)
