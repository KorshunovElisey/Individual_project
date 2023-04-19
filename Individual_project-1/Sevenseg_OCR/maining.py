import cv2
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pytesseract as pts
from transformation import frameExtractor

pts.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class Sevenseg_ocr():
    def __init__(self, screen_name, src_file_name, dst_folder_name):
        self.dst_folder_name = dst_folder_name
        self.src_file_name = src_file_name
        self.screen_name = screen_name
        self.capture = cv2.VideoCapture(0)
        self.config = rf"--psm 6 --oem 0 -c tessedit_char_whitelist=.0123456789"
        self.counter = True
        self.hsv_min = np.array((0, 100, 100), np.uint8)
        self.hsv_max = np.array((180, 255, 255), np.uint8)

        self.coordinates = (100,100)
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.fontScale = 1
        self.color = (255,0,255)
        self.thickness = 2

        #for pytesseract, now the values are set to default
        self.resize_factor = 1
        self.margin_x = 20
        self.margin_y = 20
        self.digits = 2
    

    def on_change(value):
        pass

    def image_preporation(self, resizex, resizey, src_file_name=str):
        # ret, self.img = self.capture.read()
        self.img = cv2.imread(r'Sevenseg_OCR\img\test.jpg')
        # cv2.imwrite(dst_file_name, self.img)
        self.img = cv2.resize(self.img, (resizex, resizey))
        return self.img

    def trackbar(self, name, screen=str, min_value=int, max_value=int):
        cv2.createTrackbar(str(name), screen, min_value, max_value)

    def trackbar_pos(self, name, screen=str):
        name_value = cv2.getTrackbarPos(str(name), screen)
        return name_value
    
    def number(self, resize_factor, margin_x, margin_y, src_file_name, dst_folder_name,digits_for_res):
        img_h = None
        f = frameExtractor(1/resize_factor, 
                        margin_x, 
                        margin_y, 
                        img_h, 
                        src_file_name, 
                        dst_folder_name,
                        digits_for_res)
        res = f.final_prediction()
        image = cv2.putText(self.img, res, self.coordinates, self.font, self.fontScale, self.color, self.thickness, cv2.LINE_AA)
        return image

        
dst_folder_name = 'test/'
file_path = r'Sevenseg_OCR\img\test.jpg'
s = Sevenseg_ocr('cum', file_path, dst_folder_name)


while True:
    img = s.image_preporation(500, 500, r'Sevenseg_OCR\img\test.jpg')
    cv2.imshow(s.screen_name, img)
    '''
    create trackbars for s.resize_factor, s.margin_x, s.margin_y, s.digits
    '''
    # image = s.number(1/resize_factor, 
    #                     margin_x,   
    #                     margin_y, 
    #                     img = None, 
    #                     src_file_name=s.src_file_name, 
    #                     dst_folder_name=s.dst_folder_name, 
    #                     digits=digits)
    f = frameExtractor(1/s.resize_factor, 
                        s.margin_x, 
                        s.margin_y, 
                        img=None, 
                        src_file_name=s.src_file_name, 
                        dst_folder_name='test/',
                        digits_for_res=s.digits)
    res = f.final_prediction()
    image = cv2.putText(img, res, s.coordinates, s.font, s.fontScale, s.color, s.thickness, cv2.LINE_AA)

    cv2.imshow(s.screen_name, image) 
    k = cv2.waitKey(30) & 0xFF
    if k == 27:
        break

