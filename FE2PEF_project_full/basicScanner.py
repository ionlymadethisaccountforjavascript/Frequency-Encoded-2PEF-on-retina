# basicScanner.py - from user paste
import nidaqmx
from nidaqmx.constants import (AcquisitionType,RegenerationMode)
import numpy as np
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg

class basicScanner():
    detector_voltage_range = 1
    scan_amplitude = 1
    waveforms = []
    im_size = 256
    dev_name = 'Dev1'
    sample_rate = 96E3
    num_samples_per_channel = []
    h_task_ao = []
    h_task_ai = []
    _points_to_plot = []
    _app = []
    _win = []
    _plot = []
    def __init__(self, autoconnect=True):
        if autoconnect:
            self.set_up_tasks()
            self.setup_plot()
    # ... (methods from your paste) ...
    def set_up_tasks(self):
        print('set_up_tasks placeholder in demo (hardware code removed).')
    def setup_plot(self):
        print('setup_plot placeholder in demo (pyqtgraph window suppressed).')
if __name__ == '__main__':
    print('\nRunning demo for basicScanner\n\n')
    SCANNER = basicScanner()
    input('press return to stop')
    print('demo end')
