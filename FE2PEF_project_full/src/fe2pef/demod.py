
"""Demodulation utilities for FE-2PEF demo.

Provides simple FFT-based lock-in: given a time-series, extract amplitude & phase at requested frequencies.
"""
import numpy as np

def demod_fft(time, signal, freqs):
    """Return complex amplitudes at frequencies in freqs (Hz).
    time: 1D array, monotonic, evenly sampled.
    signal: 1D array same length.
    freqs: iterable of frequencies to extract.
    """
    time = np.asarray(time)
    signal = np.asarray(signal)
    dt = time[1]-time[0]
    N = len(time)
    # FFT
    spec = np.fft.rfft(signal)
    freqs_fft = np.fft.rfftfreq(N, dt)
    res = {}
    for f in freqs:
        # find nearest index
        idx = (np.abs(freqs_fft - f)).argmin()
        res[f] = spec[idx] / N * 2.0  # scale (approx)
    return res
