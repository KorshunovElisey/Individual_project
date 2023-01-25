import cv2
import operator
import numpy as np


def pre_process_image(img, dilate=False):
    proc = cv2.GaussianBlur(img.copy(), (9, 9), 0)
    proc = cv2.adaptiveThreshold(proc, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    proc = cv2.bitwise_not(proc, proc)
    if dilate:
        kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)
        proc = cv2.dilate(proc, kernel)
    return proc


def find_corners_of_largest_polygon(img):
    contours, h = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    polygon = contours[0]  # Largest image
    bottom_right, _ = max(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
    top_left, _ = min(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
    bottom_left, _ = min(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
    top_right, _ = max(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
    return [polygon[top_left][0], polygon[top_right][0], polygon[bottom_right][0], polygon[bottom_left][0]]


def distance(p1, p2):
    a = p2[0] - p1[0]
    b = p2[1] - p1[1]
    return np.sqrt((a**2) + (b**2))


def crop(img, crop_rect):
    top_left, top_right, bottom_right, bottom_left = crop_rect[0], crop_rect[1], crop_rect[2], crop_rect[3]
    # crop the image
    left = min(top_left[0], bottom_left[0])
    right = max(top_right[0], bottom_right[0])
    top = min(top_left[1], top_right[1])
    bottom = max(bottom_left[1], bottom_right[1])
    return img[top:bottom, left:right]


import sys
import pytesseract as pts

img = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
proc = pre_process_image(img, dilate=True)
corners = find_corners_of_largest_polygon(proc)
cropped = crop(img, corners)
cv2.imwrite("1cropped.png", cropped)
cropped = cropped[20 : cropped.shape[0] - 20, 20 : cropped.shape[1] - 20]
cv2.imwrite("2cropped.png", cropped)
alpha = 3  # Contrast control (1.0-3.0)
beta = 0  # Brightness control (0-100)
adjusted = cv2.convertScaleAbs(cropped, alpha=alpha, beta=beta)
cv2.imwrite("3adjusted.png", adjusted)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 5))
opening = cv2.morphologyEx(adjusted, cv2.MORPH_OPEN, kernel)
cv2.imwrite("4opening.png", opening)
# simple thresholding
# thresh = cv2.threshold(opening, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
# thresh = cv2.adaptiveThreshold(opening, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 51, 5)
# cv2.imwrite("5threshold.png", thresh)
res = pts.image_to_string(opening, lang="letsgodigital", config="--psm 6 --oem 0 -c tessedit_char_whitelist=123456789")
print(res)