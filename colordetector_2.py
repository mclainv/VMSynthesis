from centroidtracker import CentroidTracker
from imutils.video import FileVideoStream
import numpy as np
import time
import cv2
import argparse
import imutils
from collections import deque


#Declare arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--file", type=str,
                help="Path to video file")
args = vars(ap.parse_args())

#simple blob detector from cv2 class
detector = cv2.SimpleBlobDetector()

#declare green, and points to be tracked
greenLower = (29, 86, 6)
greenUpper = (64, 255, 255)
pts = deque(maxlen=64)

vs = cv2.VideoCapture(args["file"])

#centroid tracking setup
ct = CentroidTracker()
(H, W) = (None, None)

#allow time for video to load
time.sleep(2.0)

#loop over the frames from the video stream
while True:
        #read the frame
        frame = vs.read()
        frame = frame[1]
        if frame is None:
            break
        #resize
        frame = imutils.resize(frame, width=600)
        #blur and convert image for analysis
        blurred = cv2.GaussianBlur(frame, (11,11), 0) #what does this line do?
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        #green mask!
        mask = cv2.inRange(hsv, greenLower, greenUpper)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        # find contours in the mask and initialize the current
        # (x, y) center of the ball

        #cnts = contours
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        center = None
        rects = []

        #initialize 
        # only proceed if at least one contour was found
        if len(cnts) > 0:
            #go through all contours in the mask
            for c in cnts:
                ((x, y), radius) = cv2.minEnclosingCircle(c)
                M = cv2.moments(c)
                rects.append(cv2.boundingRect(c[0]))
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                # only proceed if the radius meets a minimum size
                if radius > 10:
                    # draw the circle and centroid on the frame,
                    # then update the list of tracked points
                    cv2.circle(frame, (int(x), int(y)), int(radius),
                        (0, 255, 255), 2)
                    cv2.circle(frame, center, 5, (0, 0, 255), -1)

                    # update the points queue
                    pts.appendleft(center)
            objects = ct.update(rects)

            for (objectID, centroid) in objects.items():
                #draw both the ID of the object on the output frame
                text = "ID {}".format(objectID)
                cv2.putText(frame, text, (centroid[0] - 10, centroid[1] -
                                10),
                                cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,255,0), 2)
                print(objectID, " x: ", x, " y: ", y)

        # show the frame to our screen
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        # if the 'q' key is pressed, stop the loop
        if key == ord("q"):
            break
# close all windows
cv2.destroyAllWindows()
