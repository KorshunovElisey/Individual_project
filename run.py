import sys

from PyQt5.QtWidgets import QApplication

QT_PLUGIN_PATH = 'C:\Users\gabriel\anaconda3\Library\plugins'

from interface.main_window import MainWindow

if __name__ == "__main__":
    App = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(App.exec())
