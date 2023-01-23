import sys
import numpy as np
import cv2 as cv

# параметры цветового фильтра
hsv_min = np.array((0,0,0), np.uint8)
hsv_max = np.array((0,0,0), np.uint8)
green = (0,255,0)
counter = True
CounterForHierarhy = True

def on_change(value):
    pass


while True:
    fn = 'img/test_CROPPED.jpg' # путь к файлу с картинкой
    img = cv.imread(fn)
    # img = cv.resize(img, (500, 500))
    backInGrey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, backInBlack = cv.threshold(backInGrey, 177, 255, 0)
    hsv = cv.cvtColor( img, cv.COLOR_BGR2HSV ) # меняем цветовую модель с BGR на HSV 
    thresh = cv.inRange( hsv, hsv_min, hsv_max ) # применяем цветовой фильтр
    # ищем контуры и складируем их в переменную contours
    contours, hierarchy = cv.findContours( thresh.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours( img, contours, -1, (255,0,0), 3, cv.LINE_AA, hierarchy, cv.LINE_AA)
    cv.imshow('contours', img) # выводим итоговое изображение в окно
    if counter:
        cv.createTrackbar('color', 'contours', 0, 255, on_change)
        cv.createTrackbar('color_max', 'contours', 255, 255, on_change)
        cv.createTrackbar('saturation', 'contours', 0, 255, on_change)
        cv.createTrackbar('saturation_max', 'contours', 255, 255, on_change)
        cv.createTrackbar('light', 'contours', 0, 255, on_change)
        cv.createTrackbar('light_max', 'contours', 255, 255, on_change)
        cv.createTrackbar('hierarhy', 'contours', 0, 1, on_change)
        counter = False

    hsv_min[1] = cv.getTrackbarPos('saturation', 'contours')
    hsv_min[2] = cv.getTrackbarPos('light', 'contours')
    hsv_max[0] = cv.getTrackbarPos('color_max', 'contours')
    hsv_max[1] = cv.getTrackbarPos('saturation_max', 'contours')
    hsv_max[2] = cv.getTrackbarPos('light_max', 'contours')
    if cv.getTrackbarPos('hierarhy', 'contours') == 1:
        if CounterForHierarhy:
            print(hierarchy[0][1])
            print(contours)
            current_hierarhy = hierarchy
            CounterForHierarhy = False
    if cv.getTrackbarPos('hierarhy', 'contours') == 0:
        CounterForHierarhy = True

    # # отображаем контуры поверх изображения
    # cv.drawContours( img, contours, -1, (255,0,0), 3, cv.LINE_AA, hierarchy, 1 )
    # cv.imshow('contours', img) # выводим итоговое изображение в окно

    k = cv.waitKey(30) & 0xFF
    if k == 27:
        break
