import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
import time


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self, camera_index: int):
        super().__init__()
        self.camera_index = camera_index

    def run(self):
        # capture from webcam
        cap = cv2.VideoCapture(self.camera_index)
        while True:
            ret, cv_img = cap.read()
            if ret:
                self.change_pixmap_signal.emit(cv_img)
                time.sleep(6)
