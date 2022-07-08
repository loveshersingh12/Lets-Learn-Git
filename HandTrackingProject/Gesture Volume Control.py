import cv2
import time
import numpy as np
import handtrackingmodule as htm


#######################
wcam, hCam = 640, 418
#######################
cap = cv2.VideoCapture(0)
cap.set(3, wcam)
cap.set(4, hCam)
cTime = 0
pTime = 0
while True:
    ret, frame = cap.read()
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    cv2.putText(frame, f'FPS: {int(fps)}', (40,70), cv2.FONT_HERSHEY_COMPLEX,
                1, (255,0,0), 3)
    cv2.imshow("frame", frame)
    cv2.waitKey(1)