import time
import numpy as np
from environment import RetinaEnvironment
from forward_model import ExcitationSource, FocalVolumeMixer, DetectorModel, simulate_pixel
from lockin import demodulate_pixel
from fast_pipeline import simulate_and_demodulate_grid

grid_shape = (80, 80)
env = RetinaEnvironment(grid_shape=grid_shape, seed=0, layout="separated_blobs",
                         separation_um=9.0, blob_sigma_um=6.0)
source = ExcitationSource(duration_s=2e-4, sample_rate_hz=2e6, f1_hz=1.1e5, f2_hz=1.3e5)
mixer = FocalVolumeMixer()
atten_map = env.attenuation("rpe")

# ---- OLD: per-pixel Python loop ----
rng_old = np.random.default_rng(1)
detector = DetectorModel(shot_noise_scale=0.02, gaussian_noise_std=0.01, rng=rng_old)
t0 = time.time()
ny, nx = grid_shape
conv_old = np.zeros(grid_shape)
I1_old = np.zeros(grid_shape)
for iy in range(ny):
    for ix in range(nx):
        raw = simulate_pixel(source, mixer, detector, env.density_A[iy, ix], env.density_B[iy, ix], atten_map[iy, ix])
        conv_old[iy, ix] = raw.mean()
        I1_old[iy, ix], _, _ = demodulate_pixel(raw, source.t, source.f1, source.f2, source.sample_rate_hz)
t_old = time.time() - t0

# ---- NEW: vectorized chunked pipeline ----
rng_new = np.random.default_rng(1)
t0 = time.time()
conv_flat, I1_flat, I2_flat, I3_flat = simulate_and_demodulate_grid(
    source, mixer, env.density_A.ravel(), env.density_B.ravel(), atten_map.ravel(),
    rng=rng_new, chunk_size=2000)
t_new = time.time() - t0

print(f"Grid: {grid_shape} = {ny*nx} pixels")
print(f"OLD per-pixel loop:      {t_old:.2f}s")
print(f"NEW vectorized/chunked:  {t_new:.2f}s")
print(f"Speedup: {t_old/t_new:.1f}x")
print()
print(f"conventional correlation old vs new: {np.corrcoef(conv_old.ravel(), conv_flat)[0,1]:.4f}")
print(f"I1 correlation old vs new:           {np.corrcoef(I1_old.ravel(), I1_flat)[0,1]:.4f}")
