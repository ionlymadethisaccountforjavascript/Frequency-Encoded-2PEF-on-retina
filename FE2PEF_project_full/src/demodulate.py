import numpy as np
from scipy.signal import butter, filtfilt

def lowpass_filter(data, fs, fc=2000.0, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, fc/nyq, btype='low')
    return filtfilt(b, a, data)

def lockin_demod(signal, fs, ref_freq, fc=2000.0):
    t = np.arange(len(signal)) / float(fs)
    ref_cos = np.cos(2*np.pi*ref_freq*t)
    ref_sin = np.sin(2*np.pi*ref_freq*t)
    inphase = lowpass_filter(signal * ref_cos, fs, fc=fc)
    quadrature = lowpass_filter(signal * ref_sin, fs, fc=fc)
    amplitude = np.sqrt(inphase**2 + quadrature**2)
    return amplitude, inphase, quadrature
