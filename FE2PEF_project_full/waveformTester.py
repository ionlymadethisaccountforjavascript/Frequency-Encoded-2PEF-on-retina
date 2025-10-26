# waveformTester.py - from user paste (trimmed demo version)
import nidaqmx
from nidaqmx.constants import (AcquisitionType,RegenerationMode)
import numpy as np
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg

class waveformTester():
    dev_name = 'Dev1'
    sample_rate = 32E3
    waveform_type='sine'
    galvo_amplitude =  4
    pixels_per_line =  256
    num_reps_per_acq = 10
    ao_task = []
    waveform = []
    ai_task = []
    def __init__(self,dev_name=''):
        if 'Dev' in dev_name:
            self.dev_name = dev_name
        print('waveformTester demo loaded (real hardware methods are present in original).')
if __name__ == '__main__':
    W=waveformTester()
    input('Press return to stop\n')
