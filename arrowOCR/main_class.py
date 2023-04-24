import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
import time
import scipy.signal as signal



plt.style.use('Solarize_Light2')

time_n = time.time()

class arrow_detection():
    def __init__(self, dst_folder_name):

        self.dst_folder_name = dst_folder_name
        self.recordList = np.array([], dtype=np.float64)
        self.timeList = np.array([], dtype=np.float64)
        self.writeTime = np.array([], dtype=np.float64)
        self.writeValue = np.array([], dtype=np.int64)
        self.writeValue2 = np.array([], dtype=np.float64)
        self.BLUE = (255, 0, 0)
        self.YELLOW = (0, 255, 255)
        self.RED = (0, 0, 255)
        self.t = time.time()
        self.origin = time.time()



    def videoRedaction(self, img, hsvminH, hsvminS, hsvminV, hsvmaxH, hsvmaxS, hsvmaxV, area_min, area_max, shift, factor):
        hsv_min = np.array((hsvminH, hsvminS, hsvminV), np.uint8)
        hsv_max = np.array((hsvmaxH, hsvmaxS, hsvmaxV), np.uint8)
        referenceNew = [1, 0]
        unCounter = 0
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        thresh = cv2.inRange(hsv, hsv_min, hsv_max)
        contours0, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image=img, contours=contours0, contourIdx=-1, color=self.BLUE, thickness=2, lineType=cv2.LINE_AA)
        reference = [1, 0]
        shift = math.pi * shift / 180 if factor == 1 else - math.pi * shift / 180
        referenceNew[0] = reference[0] * math.cos(shift) - reference[1] * math.sin(shift)
        referenceNew[1] = reference[1] * math.cos(shift) + reference[0] * math.sin(shift)
        referenceNewTouple = (referenceNew[0], referenceNew[1])
        cv2.line(img, (100, 100), (int(referenceNew[0] * 50) + 100, int(referenceNew[1] * 50) + 100), (0, 255, 0), 3)
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
                    (referenceNew[0] * usedEdge[0] + referenceNew[1] * usedEdge[1]) / (
                                cv2.norm(referenceNewTouple) * cv2.norm(usedEdge))
                )
                if area_min < area < area_max and un > unCounter:
                    cv2.drawContours(img, [box], 0, self.RED, 2)
                    cv2.circle(img, centre, 5, self.YELLOW, 2)
                    cv2.putText(img, '%d' % int(self.angle), (centre[0] + 20, centre[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                self.YELLOW, 2)
        res_image = img
        res_image = cv2.resize(res_image, [640, 480])
        return res_image

    def recordData(self, record):
        if record == 1:
            if time.time() - self.t >= 0.01:
                self.t = time.time()
                self.recordList = np.append(self.recordList, self.angle)
                self.timeList = np.append(self.timeList, self.t - self.origin)

    def stopRecordData(record):
        record = 0
        return record

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
        with open(self.dst_file_name_value, 'w') as f:
            f.write(str(self.writeValue))
        with open(self.dst_file_name_time, 'w') as f:
            f.write(str(self.writeTime))

# image_processor = arrow_detection(
#             dst_folder_name="\kort",)
# cv_vid = cv2.VideoCapture(0)

# while True:
#     print(time.time()-time_n)
#     time_n = time.time()
#     ret, cv_img = cv_vid.read()
#     cv2.imshow('dtest', cv_img)

#     print("Before " + str(time.time()-time_n))
#     #image_processor.videoRedaction(img=cv_img, hsvminH=0, hsvminS=73, hsvminV=190, hsvmaxH=255, hsvmaxS=255,
#                                    #hsvmaxV=255, area_min=0, area_max=50000, shift=0, factor=1)

#     processed_image = image_processor.videoRedaction(img=cv_img, hsvminH=0, hsvminS=73, hsvminV=190, hsvmaxH=255, hsvmaxS=255,
#                                    hsvmaxV=255, area_min=0, area_max=50000, shift=0, factor=1)
#     print("After " + str(time.time() - time_n))
#     cv2.imshow('d', processed_image)
#     print("After1 " + str(time.time() - time_n))
#     k = cv2.waitKey(30) & 0xFF
#     if k == 27:
#         break

# cv2.destroyAllWindows()