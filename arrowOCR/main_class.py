import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
import time
import scipy.signal as signal
from numba import prange, jit


plt.style.use('Solarize_Light2')

time_n = time.time()

class arrow_detection():
    def __init__(self, dst_folder_name=str, hsvminH=int, hsvminS=int, hsvminV=int,
                 hsvmaxH=int, hsvmaxS=int, hsvmaxV=int, area_min=int, area_max=int, shift=int, factor=int, record=int):

        self.dst_folder_name = dst_folder_name
        self.hsvminH = hsvminH
        self.hsvminS = hsvminS
        self.hsvminV = hsvminV
        self.hsvmaxH = hsvmaxH
        self.hsvmaxS = hsvmaxS
        self.hsvmaxV = hsvmaxV
        self.hsv_min = np.array((hsvminH, hsvminS, hsvminV), np.uint8)
        self.hsv_max = np.array((hsvmaxH, hsvmaxS, hsvmaxV), np.uint8)
        self.record = record
        self.record_c = False
        self.recordList = np.array([], dtype=np.float64)
        self.timeList = np.array([], dtype=np.float64)
        self.writeTime = np.array([], dtype=np.float64)
        self.writeValue = np.array([], dtype=np.int64)
        self.writeValue2 = np.array([], dtype=np.float64)
        self.area_min = area_min
        self.area_max = area_max
        self.unCounter = 0
        self.shift = shift
        self.factor = factor
        self.referenceNew = [1, 0]
        self.BLUE = (255, 0, 0)
        self.YELLOW = (0, 255, 255)
        self.RED = (0, 0, 255)
        self.capture = cv2.VideoCapture(0)
        self.counter = True
        self.t = time.time()
        self.origin = time.time()



    def videoRedaction(self, img):
        self.hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        self.thresh = cv2.inRange(self.hsv, self.hsv_min, self.hsv_max)
        contours0, hierarchy = cv2.findContours(self.thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image=img, contours=contours0, contourIdx=-1, color=self.BLUE, thickness=2, lineType=cv2.LINE_AA)
        reference = [1, 0]
        self.shift = math.pi * self.shift / 180 if self.factor == 1 else - math.pi * self.shift / 180
        self.referenceNew[0] = reference[0] * math.cos(self.shift) - reference[1] * math.sin(self.shift)
        self.referenceNew[1] = reference[1] * math.cos(self.shift) + reference[0] * math.sin(self.shift)
        referenceNewTouple = (self.referenceNew[0], self.referenceNew[1])
        cv2.line(img, (100, 100), (int(self.referenceNew[0] * 50) + 100, int(self.referenceNew[1] * 50) + 100), (0, 255, 0), 3)
        cv2.circle(img, (100, 100), 50, self.RED, 2)

        for cnt in contours0:
            if len(cnt) > 30:
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                centre = (int(rect[0][0]), int(rect[0][1]))
                area = int(rect[1][0] * rect[1][1])

                edge1 = (np.int0((box[1][0] - box[0][0], box[1][1] - box[0][1])))
                edge2 = (np.int0((box[2][0] - box[1][0], box[2][1] - box[1][1])))

                usedEdge, unUsedEdge = edge1, edge2
                if cv2.norm(edge2) > cv2.norm(edge1):
                    usedEdge, unUsedEdge = edge2, edge1
                un = cv2.norm(usedEdge) / cv2.norm(unUsedEdge)
                self.angle = 180.0 / math.pi * math.acos(
                    (self.referenceNew[0] * usedEdge[0] + self.referenceNew[1] * usedEdge[1]) / (
                                cv2.norm(referenceNewTouple) * cv2.norm(usedEdge))
                )
                if self.area_min < area < self.area_max and un > self.unCounter:
                    cv2.drawContours(img, [box], 0, self.RED, 2)
                    cv2.circle(img, centre, 5, self.YELLOW, 2)
                    cv2.putText(img, '%d' % int(self.angle), (centre[0] + 20, centre[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                self.YELLOW, 2)
        self.res_image = img

    def recordData(self):
        if self.record == 1:
            if time.time() - self.t >= 0.01:
                self.t = time.time()
                self.recordList = np.append(self.recordList, self.angle)
                self.timeList = np.append(self.timeList, self.t - self.origin)

    def stopRecordData(self):
        self.record = 0

    def chart(self):
        figure = plt.figure()
        ax = figure.add_subplot()
        ax.plot(self.timeList, self.recordList, linewidth=1, alpha=1)
        sos = signal.butter(50, 35, 'lp', fs=len(self.recordList), output='sos')
        trueRecord = np.copy(self.recordList)
        print(len(self.recordList))
        for i in prange(len(self.recordList)):
            if i <= 5 or i > len(self.recordList) - 6:
                trueRecord[i] = self.recordList[i]
            else:
                count = 0
                for j in prange(i - 3, i + 3):
                    count += self.recordList[j]
                trueRecord[i] = count / 7
        filtered = signal.sosfiltfilt(sos, self.recordList)
        for i in prange(0, len(self.timeList), 5):
            self.writeTime = np.append(self.writeTime, self.timeList[i])
            self.writeValue = np.append(self.writeValue, int(filtered[i]))
            self.writeValue2 = np.append(self.writeValue2, trueRecord[i])
        ax.plot(self.timeList, filtered, color='green', linewidth=1, alpha=1)
        ax.plot(self.writeTime, self.writeValue, color='purple', linewidth=1, alpha=1)
        with open(self.dst_file_name_value, 'w') as f:
            f.write(str(self.writeValue))
        with open(self.dst_file_name_time, 'w') as f:
            f.write(str(self.writeTime))

image_processor = arrow_detection( 
            dst_folder_name="F:\PyhonScripts\Individual_project\kort",
            hsvminH=0,
            hsvminS=73,
            hsvminV=190,
            hsvmaxH=255,
            hsvmaxS=255,
            hsvmaxV=255,
            area_min=0,
            area_max=50000,
            shift=0,
            factor=1
        )

while True:
    print(time.time()-time_n)
    time_n = time.time()
    cv_vid = cv2.VideoCapture(0)
    ret, cv_img = cv_vid.read()
    print("Before " + str(time.time()-time_n))
    print("After " + str(time.time()-time_n))
    image_processor.videoRedaction(img=cv_img)
    print("After1 " + str(time.time()-time_n))
    processed_image = image_processor.res_image
    cv2.imshow('d', processed_image)
    k = cv2.waitKey(30) & 0xFF
    if k == 27:
        break