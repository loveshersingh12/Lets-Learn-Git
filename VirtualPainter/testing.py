import cv2
import numpy as np
import time
import os
import handtrackingmodule as htm

##########################
brushThickness = 15
eraserThickness = 100
##########################
folderPath = "Header"
myList = os.listdir(folderPath)
print(myList)
overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)
print(len(overlayList))

header = overlayList[0]
drawColor = (0, 0, 255)
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = htm.handDetector(detectionCon=0.85)
xp, yp = 0, 0
frameCanvas = np.zeros((720, 1280, 3), np.uint8)


while True:

    # 1. import image
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    # 2. Find Landmarks
    frame = detector.findHands(frame)
    lmList = detector.findPosition(frame, draw=False)

    if len(lmList) != 0:
        # print(lmList)

        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]


    # 3. Check which fingers are up

        fingers = detector.fingersUp()
        #print(fingers)

        # 4. If selection mode - Two fingers are up
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            cv2.rectangle(frame, (x1, y1 - 25), (x2, y2 + 25),
                          drawColor, cv2.FILLED)
            print("Selection Mode")
            if y1<136:
                if 250 < x1 < 450:
                    header = overlayList[0]
                    drawColor = (0, 0, 255)
                elif 550 < x1 < 750:
                    header = overlayList[1]
                    drawColor = (0, 255, 0)
                elif 800 < x1 < 950:
                    header = overlayList[2]
                    drawColor = (255, 0, 0)
                if 1050 < x1 < 1280:
                    header = overlayList[3]
                    drawColor = (0, 0, 0)

        # 5. If Drawing Mode - Index finger is up
        if fingers[1] and fingers[2] == False:
            cv2.circle(frame, (x1,y1), 15, drawColor, cv2.FILLED)
            print("Drawing Mode")
            if xp == 0 and yp == 0:
                xp, yp = x1, y1
            if drawColor == (0, 0, 0):
                cv2.line(frame, (xp, yp), (x1, y1), drawColor, eraserThickness)
                cv2.line(frameCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
            else:
                cv2.line(frame, (xp, yp), (x1, y1), drawColor, brushThickness)
                cv2.line(frameCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)

            xp, yp = x1, y1

    frameGray = cv2.cvtColor(frameCanvas, cv2.COLOR_BGR2GRAY)
    _, frameInv = cv2.threshold(frameGray, 25, 255, cv2.THRESH_BINARY_INV)
    frameInv = cv2.cvtColor(frameInv, cv2.COLOR_GRAY2BGR)
    frame = cv2.bitwise_and(frame, frameInv)
    frame = cv2.bitwise_or(frame, frameCanvas)

    # Setting the header image
    frame[0:136, 0:1280] = header
    frame = cv2.addWeighted(frame, 0.5, frameCanvas, 0.5, 0)
    cv2.imshow("Image", frame)
    cv2.imshow("Canvas", frameCanvas)
    cv2.imshow("inverse", frameInv)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
