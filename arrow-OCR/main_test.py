import numpy as np
import cv2
import math
import time

hsv_min = np.array((0, 100, 100), np.uint8)
hsv_max = np.array((180, 255, 255), np.uint8)

BLUE = (255, 0, 0)
YELLOW = (0, 255, 255)
RED = (0, 0, 255)
capture = cv2.VideoCapture(0)
t = time.time()

def on_change(value):
    print(value)

while True:
    ret, img = capture.read()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    thresh = cv2.inRange(hsv, hsv_min, hsv_max)
    contours0, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image=img, contours=contours0, contourIdx=-1, color=BLUE, thickness=2, lineType=cv2.LINE_AA)
    for cnt in contours0:
        if len(cnt) > 30:
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            centre = (int(rect[0][0]), int(rect[0][1]))
            area = int(rect[1][0] * rect[1][1])

            edge1 = np.int0((box[1][0] - box[0][0], box[1][1] - box[0][1]))
            edge2 = np.int0((box[2][0] - box[0][0], box[2][1] - box[0][1]))

            usedEdge = edge1
            if cv2.norm(edge2) > cv2.norm(edge1):
                usedEdge = edge2
            reference = (1, 0)
            
            angle = 180.0 / math.pi * math.acos((reference[0] * usedEdge[0] + reference[1] * usedEdge[1]) / (cv2.norm(reference, cv2.NORM_L2) * cv2.norm(usedEdge, cv2.NORM_L2)))
            if 750 < area < 5000:
                cv2.drawContours(img, [box], 0, RED, 2)
                cv2.circle(img, centre, 5, YELLOW, 2)
                cv2.putText(img, '%d' % int(angle), (centre[0] + 20, centre[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, YELLOW, 2)
                if time.time() - t >= 0.5:
                    t = time.time()
                    print(angle)

    cv2.imshow('d', thresh)
    cv2.imshow('f', img)
    k = cv2.waitKey(30) & 0xFF
    if k == 27:
        break

capture.release()
cv2.destroyAllWindows()