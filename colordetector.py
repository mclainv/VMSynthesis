from centroidtracker import CentroidTracker
from imutils.video import FileVideoStream, VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2

#setup cli arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--file", type=str,
               help="Path to video file, else read from webcam")
args = vars(ap.parse_args())

#initialize centroid tracker and video frame dimensions
ct = CentroidTracker()
(H, W) = (None, None)

#initialize the detector
detector = cv2.SimpleBlobDetector()

#initialize webcam

# if a video path was not supplied, grab the reference
# to the webcam
if not args.get("file", False):
#	vs = VideoStream(src=0).start()
    cap = cv2.VideoCapture(0)
# otherwise, grab a reference to the video file
else:
	vs = cv2.VideoCapture(args["file"])

time.sleep(2.0)
#main loop
while True:
    #read frame
    ret, frame = cap.read()
    # frame = imutils.resize(frame, width=600)
    # (H, W) = (600, 400)

    #greyscale the image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #blobs!
    keypoints = detector.detect(gray)

    #draw blobs as red circles
    frame_with_keypoints = cv2.drawKeypoints(frame, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    #show keypoints on output frame
    cv2.imshow("Keypoints", gray)
    if cv2.waitKey(1) & 0xFF == ord('1'):
        break

cv2.destroyAllWindows()
vs.stop()
