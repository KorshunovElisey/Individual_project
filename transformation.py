import cv2 as cv
import matplotlib.pyplot as plt

img = plt.imread("img/test_CROPPED.jpg")
ret,thresh1 = cv.threshold(img,127,255,cv.THRESH_BINARY)
plt.imshow(thresh1,'gray',vmin=0,vmax=255)
plt.show()