from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PyQt5 import QtWidgets, uic
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QMainWindow

from interface.camera_utils import convert_cv_to_qt_pixmap, return_camera_indexes
from interface.camera_window_7seg import CameraWindowSevenSeg
from interface.camera_window_arrow import CameraWindowArrow
from interface.video_thread import VideoThread

CUR_DIR = Path(__file__).resolve().parent


class MainWindow(QMainWindow):
    DEVICE_TYPES = ["Семисегментный", "Стрелочный"]
    camera_display_width = 640
    camera_display_height = 480

    selected_camera: int = 0
    selected_device_type: str = DEVICE_TYPES[0]
    video_thread_by_camera_number: Dict[int, Optional[VideoThread]] = defaultdict(lambda: None)
    folder_path: Optional[str] = None
    camera_windows: List[QMainWindow] = []

    def __init__(self):
        super().__init__()

        uic.loadUi(CUR_DIR / "main_window.ui", self)
        self.title = "Main window"
        self.setWindowTitle(self.title)
        self.init_window()

    def init_window(self):
        camera_indexes = return_camera_indexes()
        self.comboBox_deviceType.addItems(self.DEVICE_TYPES)
        self.comboBox_camera.addItems([str(index) for index in camera_indexes])
        self.comboBox_camera.currentTextChanged.connect(self._handle_change_camera)
        self.pushButton_selectFolder.clicked.connect(self._handle_select_folder_button)
        self.pushButton_start.setEnabled(False)
        self.pushButton_start.clicked.connect(self._handle_start_button)
        self.comboBox_deviceType.currentTextChanged.connect(self._handle_change_device_type)
        self.show()
        self._init_video_capture()

    def _handle_select_folder_button(self, event):
        self.folder_path = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Выберите папку"
        )
        self.label_selectedFolder.setText(str(self.folder_path))
        self._check_start_available()
        return True

    def _handle_start_button(self, value):
        if self.selected_device_type == "Семисегментный":
            camera_window = CameraWindowSevenSeg(
                camera_index=self.selected_camera,
                device_type=self.selected_device_type,
                folder_path=self.folder_path,
                video_thread=self.video_thread_by_camera_number[self.selected_camera],
            )
            self.camera_windows.append(camera_window)
            camera_window.show()
        if self.selected_device_type == "Стрелочный":  # "Стрелочный"
              #add CameraWindowArrow
            camera_window2 = CameraWindowArrow(
                camera_index=self.selected_camera,
                device_type=self.selected_device_type,
                folder_path=self.folder_path,
                video_thread=self.video_thread_by_camera_number[self.selected_camera],
            )
            self.camera_windows.append(camera_window2)
            camera_window2.show()
        return True

    def _handle_change_device_type(self, value: str):
        self.selected_device_type = value
        return True

    def _handle_change_camera(self, value: str):
        self.selected_camera = int(value)
        self._init_video_capture()
        return True

    def _check_start_available(self):
        validate_condition = (
            self.selected_camera is not None
            and self.selected_device_type is not None
            and self.folder_path is not None
        )
        self.pushButton_start.setEnabled(validate_condition)

    def _init_video_capture(self):
        # create the video capture thread
        if not self.video_thread_by_camera_number[self.selected_camera]:
            self.video_thread_by_camera_number[self.selected_camera] = VideoThread(camera_index=self.selected_camera)
            # connect its signal to the update_image slot
            self.video_thread_by_camera_number[self.selected_camera].change_pixmap_signal.connect(self.update_image)
            # start the thread
            self.video_thread_by_camera_number[self.selected_camera].start()
        for camera_thread in self.video_thread_by_camera_number.values():
            try:
                camera_thread.change_pixmap_signal.disconnect(self.update_image)
            except TypeError:
                pass
        # connect its signal to the update_image slot
        self.video_thread_by_camera_number[self.selected_camera].change_pixmap_signal.connect(self.update_image)

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img = convert_cv_to_qt_pixmap(
            cv_img, self.camera_display_width, self.camera_display_height
        )
        self.label_image.setPixmap(qt_img)
