
"""Simple optics model for FE-2PEF demo.

Contains:
- gaussian_psf: 2D Gaussian point-spread function approximating two-photon excitation spatial weighting.
- apply_scattering: simple exponential scattering kernel applied to emission photons (approx).
"""
import numpy as np
from scipy.signal import fftconvolve

def gaussian_psf(size=101, sigma_um=1.0, pixel_size_um=0.5, amplitude=1.0):
    """Return a normalized 2D Gaussian PSF array.
    size: odd integer pixels
    sigma_um: standard deviation in micrometers
    pixel_size_um: pixel size in micrometers
    """
    assert size % 2 == 1, "size should be odd"
    half = size//2
    xs = np.arange(-half, half+1) * pixel_size_um
    X, Y = np.meshgrid(xs, xs)
    r2 = X**2 + Y**2
    sigma_px = sigma_um
    psf = np.exp(-0.5 * r2 / (sigma_px**2))
    psf /= psf.sum()
    return psf

def apply_scattering(image, scattering_length_um=50.0, pixel_size_um=0.5):
    """Approximate scattering by convolving emission with an exponential kernel.
    scattering_length_um: longer => more blur
    """
    # kernel radius (in pixels)
    radius = int(max(3, scattering_length_um / pixel_size_um))
    xs = np.arange(-radius, radius+1) * pixel_size_um
    X, Y = np.meshgrid(xs, xs)
    R = np.sqrt(X**2 + Y**2)
    # exponential kernel
    kernel = np.exp(-R / scattering_length_um)
    kernel /= kernel.sum()
    img = fftconvolve(image, kernel, mode='same')
    return img
