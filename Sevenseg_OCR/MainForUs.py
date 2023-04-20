# from frame_extractor import frameExtractor
# import imutils
# import glob
# import os
# import shutil
import cv2
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pytesseract as pts
from transformation import frameExtractor


# file = 'img/test2.jpg'


# p = Path('Seven-Segment-OCR-master/Datasets/HQ_digital/')
# for i in p.glob('*.jpg'):
#     f = frameExtractor(image=None,
#                         src_file_name=str(i),
#                         dst_file_name=f'imgCr/{i.stem}.jpg' ,
#                         return_image=False,
#                         output_shape=(400, 100))
#     f.extractAndSaveFrame()

capture = cv2.VideoCapture(0)
pts.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
config = rf"--psm 6 --oem 0 -c tessedit_char_whitelist=.0123456789"
counter = True
hsv_min = np.array((0, 100, 100), np.uint8)
hsv_max = np.array((180, 255, 255), np.uint8)

coordinates = (100,100)
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
color = (255,0,255)
thickness = 2


def on_change(value):
    pass
 
while True:
    # ret, img = capture.read()
    # cv2.imwrite('img/imgCroppedOUT/img.jpg', img)
    img = cv2.imread(r'Sevenseg_OCR\img\test.jpg')
    file = r'Sevenseg_OCR\img\test.jpg'
    img = cv2.resize(img, (500,500))
    cv2.imshow('cum', img)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    thresh = cv2.inRange(hsv, hsv_min, hsv_max)
    if counter:
        cv2.createTrackbar('color', 'cum', 0, 255, on_change)
        cv2.createTrackbar('color_max', 'cum', 0, 255, on_change)
        cv2.createTrackbar('saturation', 'cum', 0, 255, on_change)
        cv2.createTrackbar('saturation_max', 'cum', 255, 255, on_change)
        cv2.createTrackbar('light', 'cum', 0, 255, on_change)
        cv2.createTrackbar('light_max', 'cum', 0, 255, on_change)
        cv2.createTrackbar('resize_factor', 'cum', 1, 10, on_change)
        cv2.createTrackbar('margin_x', 'cum', 0, 255, on_change)
        cv2.createTrackbar('margin_y', 'cum', 0, 255, on_change)
        cv2.createTrackbar('digits', 'cum', 0, 255, on_change)
        counter = False
    gamma = cv2.getTrackbarPos('gamma', 'cum')
    hsv_min[0] = cv2.getTrackbarPos('color', 'cum')
    hsv_min[1] = cv2.getTrackbarPos('saturation', 'cum')
    hsv_min[2] = cv2.getTrackbarPos('light', 'cum')
    hsv_max[0] = cv2.getTrackbarPos('color_max', 'cum')
    hsv_max[1] = cv2.getTrackbarPos('saturation_max', 'cum')
    hsv_max[2] = cv2.getTrackbarPos('light_max', 'cum')
    resize_factor = cv2.getTrackbarPos('resize_factor', 'cum')
    margin_x = cv2.getTrackbarPos('margin_x', 'cum')
    margin_y = cv2.getTrackbarPos('margin_y', 'cum')
    digits = cv2.getTrackbarPos('digits', 'cum')
    f = frameExtractor(1/resize_factor, 
                        margin_x, 
                        margin_y, 
                        img=None, 
                        src_file_name=file, 
                        dst_folder_name='test/',
                        digits_for_res=digits)
    res = f.final_prediction()
    image = cv2.putText(img, res, coordinates, font, fontScale, color, thickness, cv2.LINE_AA)
    cv2.imshow('cam', image)   
    k = cv2.waitKey(30) & 0xFF
    if k == 27:
        break



# print(f[0][0])
# width, height = f[2][0] - f[0][0], f[2][1] - f[0][1] 
# x, y = f[0][0], f[0][1]
# crim = img[y:height+5000, x:width+5000]
# cv2.imshow('img', crim)
# # cv2.imshow('img', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
