import numpy as np

class LaserChannel:
    def __init__(self, wavelength_nm=800.0, mean_power=1.0, modulation_freq=10000.0, modulation_depth=0.8, phase=0.0):
        self.wavelength_nm = wavelength_nm
        self.mean_power = mean_power
        self.modulation_freq = modulation_freq
        self.modulation_depth = modulation_depth
        self.phase = phase

    def intensity_time_series(self, t):
        return self.mean_power * (1.0 + self.modulation_depth * np.sin(2*np.pi*self.modulation_freq*t + self.phase))
