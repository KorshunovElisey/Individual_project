from pathlib import Path

import cv2
import numpy as np
from PyQt5 import uic
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QMainWindow

from arrowOCR.main_class import arrow_detection
from interface.camera_utils import convert_cv_to_qt_pixmap
from interface.video_thread import VideoThread

CUR_DIR = Path(__file__).resolve().parent


class CameraWindowArrow(QMainWindow):
    camera_display_width = 640
    camera_display_height = 480

    HsvmaxH = 255
    HsvmaxS = 255
    HsvmaxV = 255

    HsvminH = 0
    HsvminS = 0
    HsvminV = 0

    areamax = 50000
    areamin = 0

    shift = 0
    factor = 0


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

        uic.loadUi(CUR_DIR / "camera_window_arrow.ui", self)
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

        self.pushButton_start_rec.clicked.connect(self._handle_button_start_rec)
        self.pushButton_stop_rec.clicked.connect(self._handle_button_stop_rec)

        self.horizontalSlider_HsvmaxH.valueChanged.connect(self._handle_slider_HsvmaxH)
        self.horizontalSlider_HsvmaxS.valueChanged.connect(self._handle_slider_HsvmaxS)
        self.horizontalSlider_HsvmaxV.valueChanged.connect(self._handle_slider_HsvmaxV)

        self.horizontalSlider_HsvminH.valueChanged.connect(self._handle_slider_HsvminH)
        self.horizontalSlider_HsvminS.valueChanged.connect(self._handle_slider_HsvminS)
        self.horizontalSlider_HsvminV.valueChanged.connect(self._handle_slider_HsvminV)

        self.horizontalSlider_areamin.valueChanged.connect(self._handle_slider_areamin)
        self.horizontalSlider_areamax.valueChanged.connect(self._handle_slider_areamax)


        self.horizontalSlider_shift.valueChanged.connect(self._handle_slider_shift)
        self.horizontalSlider_factor.valueChanged.connect(self._handle_slider_factor)

        self.show()
        self._init_video_capture()

    def _init_video_capture(self):
        # connect its signal to the update_image slot
        self.camera_video_thread.change_pixmap_signal.connect(self.update_image)

    # for sliders
    def _handle_slider_HsvmaxH(self, value: int):
        self.HsvmaxH = value

    def _handle_slider_HsvmaxS(self, value: int):
        self.HsvmaxH = value
    
    def _handle_slider_HsvmaxV(self, value: int):
        self.HsvmaxH = value


    
    def _handle_slider_HsvminH(self, value: int):
        self.HsvmaxH = value

    def _handle_slider_HsvminS(self, value: int):
        self.HsvmaxH = value
    
    def _handle_slider_HsvminV(self, value: int):
        self.HsvmaxH = value



    def _handle_slider_areamin(self, value: int):
        self.areamin = value
    
    def _handle_slider_areamax(self, value: int):
        self.areamax = value
    


    def _handle_slider_shift(self, value: int):
        self.shift = value
    
    def _handle_slider_factor(self, value: int):
        self.factor = value
    
    #for buttons
    def _handle_button_start_rec(self):
        self.image_processor.recordData()

    def _handle_button_stop_rec(self):
        self.image_processor.stopRecordData()
        self.image_processor.chart()


    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        """Updates the image_label with a new opencv image"""
        # hsv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
        self.image_processor = arrow_detection(
            img=cv_img, 
            dst_folder_name=self.folder_path,
            hsvminH=self.HsvminH,
            hsvminS=self.HsvminS,
            hsvminV=self.HsvminV,
            hsvmaxH=self.HsvmaxH,
            hsvmaxS=self.HsvmaxS,
            hsvmaxV=self.HsvmaxV,
            area_min=self.areamin,
            area_max=self.areamax,
            shift=self.shift,
            factor=self.factor
        )
        self.image_processor.videoRedaction()
        processed_image = self.image_processor.res_image

        qt_img = convert_cv_to_qt_pixmap(
            processed_image, self.camera_display_width, self.camera_display_height
        )
        self.label_image.setPixmap(qt_img)



