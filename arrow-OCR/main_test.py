import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
import time
import scipy.signal as signal

hsv_min = np.array((0, 100, 100), np.uint8)
hsv_max = np.array((180, 255, 255), np.uint8)
record = 0
record_c = False
recordList = np.array([], dtype=np.float64)
timeList = np.array([], dtype=np.float64)
writeTime = np.array([], dtype=np.float64)
writeValue = np.array([], dtype=np.float64)
writeValue2 = np.array([], dtype=np.float64)
area_min = 0
area_max = 25000
unCounter = 0
shift = 0
plt.style.use('Solarize_Light2')

BLUE = (255, 0, 0)
YELLOW = (0, 255, 255)
RED = (0, 0, 255)
capture = cv2.VideoCapture(0)
counter = True
t = time.time()
origin = time.time()

def on_change(value):
    pass



while True:
    ret, img = capture.read()
    #img = cv2.imread('speed2.jpg')
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

            edge1 = (np.int0((box[1][0] - box[0][0], box[1][1] - box[0][1])))
            edge2 = (np.int0((box[2][0] - box[0][0], box[2][1] - box[0][1])))

            usedEdge, unUsedEdge = edge1, edge2
            if cv2.norm(edge2) > cv2.norm(edge1):
                usedEdge, unUsedEdge = edge2, edge1
            un = cv2.norm(usedEdge) / cv2.norm(unUsedEdge)
            reference = (1, 0)
            angle = 180.0 / math.pi * math.acos(
                (reference[0] * usedEdge[0] + reference[1] * usedEdge[1]) / (cv2.norm(reference) * cv2.norm(usedEdge))
            ) - shift
            if area_min < area < area_max and un > unCounter:
                cv2.drawContours(img, [box], 0, RED, 2)
                cv2.circle(img, centre, 5, YELLOW, 2)
                cv2.putText(img, '%d' % int(angle), (centre[0] + 20, centre[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, YELLOW, 2)
                if record == 1:
                    if time.time() - t >= 0.01:
                        t = time.time()
                        #print(angle)
                        recordList = np.append(recordList, angle)
                        timeList = np.append(timeList, t - origin)
                    record_c = True
    if record_c == True and record == 0:
        #print(recordList)
        break

    cv2.imshow('d', thresh)
    cv2.imshow('f', img)
    if counter:
        cv2.createTrackbar('shift', 'd', 0, 90, on_change)
        cv2.createTrackbar('area_min', 'd', 0, 25000, on_change)
        cv2.createTrackbar('area_max', 'd', 25000, 25000, on_change)
        cv2.createTrackbar('color', 'd', 0, 255, on_change)
        cv2.createTrackbar('color_max', 'd', 255, 255, on_change)
        cv2.createTrackbar('saturation', 'd', 0, 255, on_change)
        cv2.createTrackbar('saturation_max', 'd', 255, 255, on_change)
        cv2.createTrackbar('light', 'd', 0, 255, on_change)
        cv2.createTrackbar('light_max', 'd', 255, 255, on_change)
        #cv2.createTrackbar('ratio', 'd', 1, 30, on_change)
        cv2.createTrackbar('record', 'd', 0, 1, on_change)
        counter = False
    shift = cv2.getTrackbarPos('shift', 'd')
    area_min = cv2.getTrackbarPos('area_min', 'd')
    area_max = cv2.getTrackbarPos('area_max', 'd')
    hsv_min[0] = cv2.getTrackbarPos('color', 'd')
    hsv_min[1] = cv2.getTrackbarPos('saturation', 'd')
    hsv_min[2] = cv2.getTrackbarPos('light', 'd')
    hsv_max[0] = cv2.getTrackbarPos('color_max', 'd')
    hsv_max[1] = cv2.getTrackbarPos('saturation_max', 'd')
    hsv_max[2] = cv2.getTrackbarPos('light_max', 'd')
    record = cv2.getTrackbarPos('record', 'd')
    #unCounter = cv2.getTrackbarPos('ratio', 'd')

    k = cv2.waitKey(30) & 0xFF
    if k == 27:
        break

figure = plt.figure()
ax = figure.add_subplot()
ax.plot(timeList, recordList, linewidth=1, alpha=1)
sos = signal.butter(50, 35, 'lp', fs=len(recordList), output='sos')
trueRecord = np.copy(recordList)
print(len(recordList))
for i in range(len(recordList)):
    if i <= 5 or i > len(recordList) - 6:
        trueRecord[i] = recordList[i]
    else:
        count = 0
        for j in range(i-3, i+3):
            count += recordList[j]
        trueRecord[i] = count/7
filtered = signal.sosfiltfilt(sos, recordList)
for i in range(0, len(timeList), 5):
    writeTime = np.append(writeTime, timeList[i])
    writeValue = np.append(writeValue, filtered[i])
    writeValue2 = np.append(writeValue2, trueRecord[i])
#ax.plot(timeList, trueRecord, color='red', linewidth=1, alpha=1)
ax.plot(timeList, filtered, color='green', linewidth=1, alpha=1)
#ax.plot(writeTime, writeValue2, color='black', linewidth=1, alpha=1)
ax.plot(writeTime, writeValue, color='purple', linewidth=1, alpha=1)
capture.release()
#cv2.waitKey()
cv2.destroyAllWindows()
plt.show()
№print(writeTime)
