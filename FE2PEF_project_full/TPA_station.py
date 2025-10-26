# TPA_station.py - Combined from user-provided snippets.
# NOTE: This file was assembled from two long pasted chunks by the user.
# It may require reformatting and dependency fixes (PyQt5, numpy, custom modules in Ressources_scripts).
import sys
import time
from time import sleep
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QListWidgetItem, QFileDialog, QMessageBox, QInputDialog, QLineEdit, QAbstractItemView
from PyQt5 import uic
from PyQt5.QtGui import QPixmap
import logging

logging.basicConfig(level=logging.INFO,
                    filename="TPA station.log",
                    filemode="w",
                    format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("")

# NOTE: The original code imports many project-specific modules (Ressources_scripts.Data_processing...).
# For this demo they are not available. This file is provided as reference and will require your environment.
class MyApp(QWidget):
    def __init__(self):
        super().__init__()
        try:
            uic.loadUi('TPA station.ui', self)
        except Exception as e:
            print('UI file missing or PyQt UI loading failed:', e)
        self.setWindowTitle('TPA station - demo')
        # ... (The full GUI code you pasted earlier goes here) ...
        print('TPA station demo module loaded.')
if __name__ == '__main__':
    app = QApplication(sys.argv)
    myApp = MyApp()
    myApp.show()
    try:
        sys.exit(app.exec())
    except SystemExit:
        print('Closing Windows')
