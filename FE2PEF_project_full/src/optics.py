import numpy as np
from scipy import ndimage

def gaussian_psf(grid_x, grid_y, waist):
    """Return a normalized 2D Gaussian PSF centered on grid"""
    x0 = (grid_x.max() + grid_x.min())/2.0
    y0 = (grid_y.max() + grid_y.min())/2.0
    X, Y = np.meshgrid(grid_x, grid_y, indexing='xy')
    r2 = (X - x0)**2 + (Y - y0)**2
    psf = np.exp(-2*r2/waist**2)
    psf /= psf.sum()
    return psf

def apply_psf_image(image, waist_pixels):
    """Apply Gaussian PSF blur to 2D image using scipy.ndimage gaussian_filter"""
    if waist_pixels <= 0:
        return image
    sigma = waist_pixels/2.3548  # convert FWHM-ish to sigma approximation
    return ndimage.gaussian_filter(image, sigma=sigma)
