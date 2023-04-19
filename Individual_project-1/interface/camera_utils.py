import cv2
import numpy as np
from PyQt5 import QtGui
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap


def return_camera_indexes():
    # checks the first 10 indexes.
    index = 0
    arr = []
    i = 10
    while i > 0:
        cap = cv2.VideoCapture(index)
        if cap.read()[0]:
            arr.append(index)
            cap.release()
        index += 1
        i -= 1
    return arr


def convert_cv_to_qt_pixmap(
    cv_img: np.ndarray, output_width: int, output_height: int
) -> QPixmap:
    """Convert from an opencv image to QPixmap"""
    rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb_image.shape
    bytes_per_line = ch * w
    convert_to_Qt_format = QtGui.QImage(
        rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888
    )
    p = convert_to_Qt_format.scaled(output_width, output_height, Qt.KeepAspectRatio)
    return QPixmap.fromImage(p)
