import cv2
import time, os
import numpy as np

# Method in order to get network using models
def getNetwork():
    protoFile = "Pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt"
    weightsFile = "Pose/mpi/pose_iter_160000.caffemodel"
    
    net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
    return net

# Method in order to get network output using models
def getNetworkOutput(net, frame):
    inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (368, 368), (0, 0, 0), swapRB=False, crop=False)
    net.setInput(inpBlob)

    return net.forward()

threshold = 0.1
numberOfPoints = 15 
PAIRS = [[0, 1], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7], [1, 14], [14, 8], [8, 9], [9, 10], [14, 11], [11, 12], [12, 13]]

input_source = "Inputs/sample_video.mp4"
cap = cv2.VideoCapture(input_source)
hasFrame, frame = cap.read()

frameWidth = frame.shape[1] 
frameHeight = frame.shape[0]

outputVideo = cv2.VideoWriter('Outputs/output.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frameWidth, frameHeight))

net = getNetwork()

while cv2.waitKey(1) < 0:
    hasFrame, frame = cap.read()

    if hasFrame == False:
        break

    output = getNetworkOutput(net, frame)
    outputHeight = output.shape[2]
    outputWidth = output.shape[3]

    # List in order to keep detected keypoint's location
    points = []

    for i in range(numberOfPoints):
        # Confidence map for body part
        probMap = output[0, i, :, :]

        # Finding global maxima of the probability map
        minValue, probability, minLocation, location = cv2.minMaxLoc(probMap)
    
        # Draw keypoints
        if threshold < probability: 
            # Scale the point to fit on the original image
            x = int((frameWidth * location[0]) / outputWidth)
            y = int((frameHeight * location[1]) / outputHeight)

            #cv2.circle(frame, (x, y), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            #cv2.putText(frame, "{}".format(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, lineType=cv2.LINE_AA)

            # Add the point to the list if the probability is greater than the threshold
            points.append((x, y))
        else:
            points.append(None)

    # Draw Skeleton
    for pair in PAIRS:
        firstNode = pair[0]
        secondNode = pair[1]

        if (points[firstNode] is None) or (points[secondNode] is None):
            continue

        else:
            cv2.line(frame, points[firstNode], points[secondNode], (0, 255, 255), 2)
            cv2.circle(frame, points[firstNode], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.circle(frame, points[secondNode], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)

    cv2.imshow('Skeleton', frame)
    outputVideo.write(frame)

outputVideo.release()
