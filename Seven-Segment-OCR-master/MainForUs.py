from frame_extractor import frameExtractor
import imutils
import glob
import os
import shutil
import cv2
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


file = 'img/test2.jpg'


p = Path('Seven-Segment-OCR-master/Datasets/HQ_digital/')
for i in p.glob('*.jpg'):
    f = frameExtractor(image=None,
                        src_file_name=str(i),
                        dst_file_name=f'imgCr/{i.stem}.jpg' ,
                        return_image=False,
                        output_shape=(400, 100))
    f.extractAndSaveFrame()

 

# print(f[0][0])
# width, height = f[2][0] - f[0][0], f[2][1] - f[0][1] 
# x, y = f[0][0], f[0][1]
# crim = img[y:height+5000, x:width+5000]
# cv2.imshow('img', crim)
# # cv2.imshow('img', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
