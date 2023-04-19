from pathlib import Path

import cv2
import numpy as np
from PyQt5 import uic
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QMainWindow

from Sevenseg_OCR.transformation import frameExtractor
from interface.camera_utils import convert_cv_to_qt_pixmap
from interface.video_thread import VideoThread

CUR_DIR = Path(__file__).resolve().parent


class CameraWindowSevenSeg(QMainWindow):
    camera_display_width = 640
    camera_display_height = 480
    default_resize_factor = 0.15
    default_margin_x = 1
    default_margin_y = 1

    camera_index: int
    device_type: str
    folder_path: str
    camera_video_thread: VideoThread
    # For sliders:
    resize_factor: int
    margin_x: int
    margin_y: int

    def __init__(
        self,
        camera_index: int,
        device_type: str,
        folder_path: str,
        video_thread: VideoThread,
    ):
        super().__init__()

        uic.loadUi(CUR_DIR / "camera_window_7seg.ui", self)
        self.title = f"Camera #{camera_index}"
        self.setWindowTitle(self.title)

        self.camera_index = camera_index
        self.device_type = device_type
        self.folder_path = folder_path
        self.camera_video_thread = video_thread

        self.init_window()

    def init_window(self):
        self.pushButton_stop.clicked.connect(self.close)
        self.label_deviceType.setText(self.device_type)

        self.horizontalSlider_resizeFactor.valueChanged.connect(
            self._handle_slider_resizeFactor
        )
        self.horizontalSlider_marginX.valueChanged.connect(self._handle_slider_marginX)
        self.horizontalSlider_marginY.valueChanged.connect(self._handle_slider_marginY)

        self.show()
        self._init_video_capture()

    def _init_video_capture(self):
        # connect its signal to the update_image slot
        self.camera_video_thread.change_pixmap_signal.connect(self.update_image)


    def _handle_slider_resizeFactor(self, value: int):
        self.default_resize_factor = value/20

    def _handle_slider_marginX(self, value: int):
        self.default_margin_x = value

    def _handle_slider_marginY(self, value: int):
        self.default_margin_y = value

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        """Updates the image_label with a new opencv image"""
        img_grey = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)
        image_processor = frameExtractor(
            img=img_grey,
            resize_factor=self.default_resize_factor,
            margin_x=self.default_margin_x,
            margin_y=self.default_margin_y,
            dst_folder_name=self.folder_path,
        )
        image_processor.final_prediction()
        processed_image = image_processor.res_image

        qt_img = convert_cv_to_qt_pixmap(
            processed_image, self.camera_display_width, self.camera_display_height
        )
        self.label_image.setPixmap(qt_img)

