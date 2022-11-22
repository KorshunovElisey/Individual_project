import cv2
 

def on_change(value):
    return value 
img = cv2.imread('img/stef.png')
 
windowName = 'image'
 
cv2.imshow(windowName, img)
cv2.createTrackbar('slider', windowName, 0, 255, on_change)
cv2.waitKey(0)
cv2.destroyAllWindows()