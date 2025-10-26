
"\"\"AOTF / laser channel simulator.

Simulates N laser channels, each with:
- wavelength_nm
- max_power_mW
- modulation frequency (Hz)
- amplitude control 0..1

Provides a method to generate instantaneous power time traces and their contribution to TPEF via two-photon cross-section-like weighting.
"\"\"
import numpy as np

class AOTFChannel:
    def __init__(self, name, wavelength_nm, max_power_mW=100.0, mod_freq_hz=10000.0, duty=1.0):
        self.name = name
        self.wavelength_nm = wavelength_nm
        self.max_power_mW = max_power_mW
        self.mod_freq_hz = mod_freq_hz
        self.duty = duty
        self.amp = 1.0  # relative amplitude set by software (0..1)

    def instantaneous_power(self, t):
        \"\"\"Return instantaneous power in mW at time(s) t (numpy array or scalar).
        Use simple sinusoidal modulation for demo.
        \"\"\"
        base = self.max_power_mW * self.amp * self.duty
        return base * (0.5*(1 + np.sin(2*np.pi*self.mod_freq_hz*t)))

class AOTFSimulator:
    def __init__(self, channels=None):
        self.channels = channels if channels is not None else []

    def add_channel(self, chan):
        self.channels.append(chan)

    def powers_at(self, t):
        \"\"\"Return list of instantaneous powers for all channels at times t (array shape (n_channels, len(t))).\"\"\"
        t = np.asarray(t)
        p = np.vstack([c.instantaneous_power(t) for c in self.channels])
        return p

    def set_amplitude(self, name, amp):
        for c in self.channels:
            if c.name == name:
                c.amp = float(amp)
                return True
        return False
