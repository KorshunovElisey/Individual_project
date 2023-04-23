#!/usr/bin/env python3
import sys
import cv2
import pytesseract as pts
import time

pts.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
resize_factor = 0.15
margin_x = 30
margin_y = 70
alpha = 3  # Contrast control (1.0-3.0)
beta = 0  # Brightness control (0-100)
blur_kernel_size = 5
thresh_block_size = 51
thresh_offset = 10
pts_model = "ssd"  # "ssd", "letsgodigital" or "7seg"
psm = 6
oem = 1
erosion_kernel_size = 3
erosion_iterations = 1
denoise_h = 10
denoise_template_window_size = 7
denoise_search_window_size = 21
time_n = time.time()

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

while True:
    # file = r'Seven-Segment-OCR-master\Datasets\HQ_digital\1d0cef2855b9886e7c075b09eb4d19cacb98f225.jpg'
    # img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    print(time.time()-time_n)
    cap = cv2.VideoCapture(0)
    ret, img = cap.read()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    proc = pre_process_image(img, dilate=True)
    corners = find_corners_of_largest_polygon(proc)
    img = crop(img, corners)
    cv2.imwrite("test/1cropped.png", img)
    img = img[margin_x : img.shape[0] - margin_x, margin_y : img.shape[1] - margin_y]
    cv2.imwrite("test/2cropped.png", img)
    ## brightness and contrast
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    cv2.imwrite("test/3contrast.png", img)
    # smoothing
    img = cv2.GaussianBlur(img, (blur_kernel_size, blur_kernel_size), 0)
    cv2.imwrite("test/4blur.png", img)
    # denoise
    img = cv2.fastNlMeansDenoising(
        img,
        None,
        h=denoise_h,
        templateWindowSize=denoise_template_window_size,
        searchWindowSize=denoise_search_window_size,
    )
    cv2.imwrite("test/5denoise.png", img)
    # thresholding
    img = cv2.adaptiveThreshold(
        img,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        thresh_block_size,
        thresh_offset,
    )
    cv2.imwrite("test/6threshold.png", img)
    # erode
    kernel = np.ones((erosion_kernel_size, erosion_kernel_size), np.uint8)
    img = cv2.erode(img, kernel, iterations=erosion_iterations)
    cv2.imwrite("test/7erode.png", img)
    # dilate
    # kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)
    # img = cv2.dilate(img, kernel)
    # cv2.imwrite("test/8dilate.png", img)
    img = cv2.resize(img, (0, 0), fx=resize_factor, fy=resize_factor)
    cv2.imwrite("test/9downscale.png", img) 
    res = pts.image_to_string(
        img,
        lang=pts_model,
        config=f"--psm {psm} --oem {oem} -c tessedit_char_whitelist=.0123456789",
    )

    cv2.imshow('f', img)
    k = cv2.waitKey(30) & 0xFF
    if k == 27:
        break
