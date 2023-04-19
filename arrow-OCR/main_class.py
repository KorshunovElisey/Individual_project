import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
import time
import scipy.signal as signal


plt.style.use('Solarize_Light2')

class arrow_detection():
    def __init__(self, dst_file_name_value=str, dst_file_name_time=str, hsvminH=int, hsvminS=int, hsvminV=int,
                 hsvmaxH=int, hsvmaxS=int, hsvmaxV=int, area_min=int, area_max=int, shift=int, factor=int, record=int):
        self.dst_file_name_value = dst_file_name_value
        self.dst_file_name_time = dst_file_name_time
        self.hsvminH = hsvminH
        self.hsvminH = hsvminS
        self.hsvminH = hsvminV
        self.hsvmaxH = hsvmaxH
        self.hsvmaxH = hsvmaxS
        self.hsvmaxH = hsvmaxV
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

    def getVideo(self):
        ret, self.img = self.capture.read()
        return self.img

    def threshvideo(self):
        self.hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        self.thresh = cv2.inRange(self.hsv, self.hsv_min, self.hsv_max)
        return self.thresh

    def videoRedaction(self):
        contours0, hierarchy = cv2.findContours(self.thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image=self.img, contours=contours0, contourIdx=-1, color=self.BLUE, thickness=2, lineType=cv2.LINE_AA)
        reference = [1, 0]
        self.shift = math.pi * self.shift / 180 if self.factor == 1 else - math.pi * self.shift / 180
        self.referenceNew[0] = reference[0] * math.cos(self.shift) - reference[1] * math.sin(self.shift)
        self.referenceNew[1] = reference[1] * math.cos(self.shift) + reference[0] * math.sin(self.shift)
        referenceNewTouple = (self.referenceNew[0], self.referenceNew[1])
        cv2.line(self.img, (100, 100), (int(self.referenceNew[0] * 50) + 100, int(self.referenceNew[1] * 50) + 100), (0, 255, 0), 3)
        cv2.circle(self.img, (100, 100), 50, self.RED, 2)

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
                angle = 180.0 / math.pi * math.acos(
                    (self.referenceNew[0] * usedEdge[0] + self.referenceNew[1] * usedEdge[1]) / (
                                cv2.norm(referenceNewTouple) * cv2.norm(usedEdge))
                )
                if self.area_min < area < self.area_max and un > self.unCounter:
                    cv2.drawContours(self.img, [box], 0, self.RED, 2)
                    cv2.circle(self.img, centre, 5, self.YELLOW, 2)
                    cv2.putText(self.img, '%d' % int(angle), (centre[0] + 20, centre[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                self.YELLOW, 2)
        cv2.imshow('d', self.thresh)
        cv2.imshow('f', self.img)

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
        for i in range(len(self.recordList)):
            if i <= 5 or i > len(self.recordList) - 6:
                trueRecord[i] = self.recordList[i]
            else:
                count = 0
                for j in range(i - 3, i + 3):
                    count += self.recordList[j]
                trueRecord[i] = count / 7
        filtered = signal.sosfiltfilt(sos, self.recordList)
        for i in range(0, len(self.timeList), 5):
            self.writeTime = np.append(self.writeTime, self.timeList[i])
            self.writeValue = np.append(self.writeValue, int(filtered[i]))
            self.writeValue2 = np.append(self.writeValue2, trueRecord[i])
        ax.plot(self.timeList, filtered, color='green', linewidth=1, alpha=1)
        ax.plot(self.writeTime, self.writeValue, color='purple', linewidth=1, alpha=1)
        with open(self.dst_file_name_value, 'w') as f:
            f.write(str(self.writeValue))
        with open(self.dst_file_name_time, 'w') as f:
            f.write(str(self.writeTime))
