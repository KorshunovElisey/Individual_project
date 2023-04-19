import  sys 
import os
import tkinter
import tkinter.messagebox
import customtkinter
import cv2 

sys.path.insert(1, os.path.join(sys.path[0], '../Sevenseg_OCR'))
from maining import Sevenseg_ocr
from transformation import frameExtractor



s = Sevenseg_ocr()
s.pridurki


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


