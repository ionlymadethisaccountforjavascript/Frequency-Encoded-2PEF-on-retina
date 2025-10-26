import numpy as np, os, json
from src.laser_model import LaserChannel
from src.optics import apply_psf_image
from src.utils import create_synthetic_fluorophore_map
from src.demodulate import lockin_demod
from src.nnmf import nnmf
import matplotlib.pyplot as plt

def simulate_pixel_trace(Cs, I1t, I2t, sigma1, sigma2, sigma12):
    total = np.zeros_like(I1t)
    for i,c in enumerate(Cs):
        total += c * (sigma1[i] * I1t**2 + sigma2[i] * I2t**2 + sigma12[i] * I1t * I2t)
    return total

def main():
    outdir = 'out'
    os.makedirs(outdir, exist_ok=True)
    nx, ny = 64, 64
    n_types = 2
    maps = create_synthetic_fluorophore_map(nx=nx, ny=ny, types=n_types, spots_per_type=4, seed=1)
    waist_pixels = 3.5
    maps_psf = maps.copy()
    for t in range(n_types):
        maps_psf[t] = apply_psf_image(maps[t], waist_pixels)
    fs = 50000.0
    duration = 0.3
    t = np.arange(0, duration, 1/fs)
    L1 = LaserChannel(wavelength_nm=470, mean_power=1.0, modulation_freq=12000.0, modulation_depth=0.8, phase=0.0)
    L2 = LaserChannel(wavelength_nm=640, mean_power=0.8, modulation_freq=15000.0, modulation_depth=0.7, phase=0.2)
    I1 = L1.intensity_time_series(t)
    I2 = L2.intensity_time_series(t)
    sigma1 = np.array([1.0, 0.2])
    sigma2 = np.array([0.1, 1.2])
    sigma12 = np.array([0.3, 0.3])
    trace = np.zeros((ny, nx, len(t)))
    for iy in range(ny):
        for ix in range(nx):
            Cs = maps_psf[:, iy, ix]
            trace[iy, ix, :] = simulate_pixel_trace(Cs, I1, I2, sigma1, sigma2, sigma12)
    detector_signal = trace.sum(axis=(0,1))
    rng = np.random.RandomState(2)
    detector_signal += rng.normal(scale=0.02 * detector_signal.max(), size=detector_signal.shape)
    amp_map1 = np.zeros((ny, nx))
    amp_map2 = np.zeros((ny, nx))
    amp_map_mix = np.zeros((ny, nx))
    for iy in range(ny):
        for ix in range(nx):
            s = trace[iy, ix, :] + rng.normal(scale=0.005 * trace.max(), size=len(t))
            a1,_,_ = lockin_demod(s, fs, L1.modulation_freq, fc=2000.0)
            a2,_,_ = lockin_demod(s, fs, L2.modulation_freq, fc=2000.0)
            am,_,_ = lockin_demod(s, fs, L1.modulation_freq+L2.modulation_freq, fc=2000.0)
            amp_map1[iy, ix] = a1.mean()
            amp_map2[iy, ix] = a2.mean()
            amp_map_mix[iy, ix] = am.mean()
    V = np.vstack([amp_map1.ravel(), amp_map2.ravel(), amp_map_mix.ravel()])
    W, H = nnmf(V, n_components=2, n_iter=200)
    k = H.shape[0]
    unmixed_maps = H.reshape(k, ny, nx)
    plt.imsave(os.path.join(outdir,'demod_f1.png'), amp_map1, cmap='inferno')
    plt.imsave(os.path.join(outdir,'demod_f2.png'), amp_map2, cmap='inferno')
    plt.imsave(os.path.join(outdir,'demod_fmix.png'), amp_map_mix, cmap='inferno')
    for i in range(unmixed_maps.shape[0]):
        plt.imsave(os.path.join(outdir,f'unmixed_{i}.png'), unmixed_maps[i], cmap='inferno')
    with open(os.path.join(outdir,'summary.json'),'w') as fh:
        json.dump({'nx':nx,'ny':ny}, fh)
    print('Done. Outputs saved to out/')

if __name__ == '__main__':
    main()
