"""
Vectorized, chunked forward-simulate + lock-in pipeline.
============================================================
Same physics as forward_model.py + lockin.py, but restructured to eliminate
two sources of pure waste in the original per-pixel Python loop:

1. Filter redesign: `scipy.signal.butter()` was being called fresh for every
   pixel x every target frequency x every quadrature channel (5400 calls for
   a 900-pixel scan), even though the filter only depends on (target_freq,
   sample_rate) -- 3 distinct filters total, ever. This alone was ~58% of
   total runtime (profiled).
2. Per-pixel carrier recomputation: m1, m2, and the three pathway carriers
   (m1^2, m1*m2, m2^2) don't depend on pixel data at all -- only on
   (f1, f2, t) -- yet were being rebuilt from scratch on every iteration.

Fix: compute everything that doesn't depend on pixel data exactly ONCE, then
batch the per-pixel-dependent math (concentration scaling, noise, filtfilt)
across many pixels at once as 2D array ops instead of a Python loop.

Memory is bounded by processing `chunk_size` pixels at a time rather than
materializing a (n_pixels, n_samples) array for the whole grid at once --
this keeps peak memory O(chunk_size * n_samples) instead of
O(n_pixels * n_samples), so it scales to large grids without blowing up RAM.
"""

import numpy as np
from scipy.signal import butter, filtfilt

from forward_model import ExcitationSource, FocalVolumeMixer


def _design_filters(f1_hz, f2_hz, sample_rate_hz, order=4):
    """Design the 3 lock-in low-pass filters ONCE. Returns a dict
    freq_name -> (b, a) filter coefficients, reused for every pixel."""
    targets = {"2f1": 2 * f1_hz, "f1+f2": f1_hz + f2_hz, "2f2": 2 * f2_hz}
    filters = {}
    nyq = 0.5 * sample_rate_hz
    for name, freq in targets.items():
        cutoff = freq * 0.05
        normal_cutoff = min(cutoff / nyq, 0.99)
        b, a = butter(order, normal_cutoff, btype="low", analog=False)
        filters[name] = (b, a, freq)
    return filters


def simulate_and_demodulate_grid(source: ExcitationSource, mixer: FocalVolumeMixer,
                                  conc_A_flat, conc_B_flat, atten_flat,
                                  shot_noise_scale=0.02, gaussian_noise_std=0.01,
                                  rng=None, chunk_size=2000, dtype=np.float32):
    """
    Vectorized replacement for the (simulate_pixel + demodulate_pixel) inner
    loop. Processes `chunk_size` pixels at a time as batched 2D arrays.

    Returns: conventional_flat, I1_flat, I2_flat, I3_flat -- each shape
    (n_pixels,), matching the flattened pixel order of the input arrays.

    Complexity: still fundamentally O(n_pixels * n_samples) flops (you can't
    avoid simulating every sample of every pixel), but replaces
    O(n_pixels * n_freq) individual Python-level filtfilt/butter calls with
    O(n_pixels / chunk_size * n_freq) batched calls -- for an 80x80 grid at
    chunk_size=2000, that's 3 chunks x 3 filters = 9 filtfilt calls total,
    versus 6400 x 3 x 2 = 38400 in the naive per-pixel version.
    """
    rng = rng or np.random.default_rng(0)
    n_pixels = conc_A_flat.shape[0]

    # ---- one-time setup: carriers, references, filters (pixel-independent) ---
    m1, m2 = source.modulated_carriers()
    p1_carrier = (m1 ** 2).astype(dtype)
    p2_carrier = (m1 * m2).astype(dtype)
    p3_carrier = (m2 ** 2).astype(dtype)

    t = source.t.astype(dtype)
    filters = _design_filters(source.f1, source.f2, source.sample_rate_hz)
    refs = {name: (np.cos(2 * np.pi * freq * t).astype(dtype),
                   np.sin(2 * np.pi * freq * t).astype(dtype))
            for name, (b, a, freq) in filters.items()}

    cA, cB = mixer.coeffs_A, mixer.coeffs_B
    n_samples = t.shape[0]
    transient = n_samples // 4

    conventional_out = np.empty(n_pixels, dtype=dtype)
    I_out = {name: np.empty(n_pixels, dtype=dtype) for name in filters}

    # ---- chunked batch processing -----------------------------------------
    for start in range(0, n_pixels, chunk_size):
        end = min(start + chunk_size, n_pixels)
        cA_chunk = conc_A_flat[start:end][:, None].astype(dtype)   # (chunk, 1)
        cB_chunk = conc_B_flat[start:end][:, None].astype(dtype)
        atten_chunk = atten_flat[start:end][:, None].astype(dtype)

        w1 = atten_chunk * (cA_chunk * cA[0] + cB_chunk * cB[0])   # (chunk, 1)
        w2 = atten_chunk * (cA_chunk * cA[1] + cB_chunk * cB[1])
        w3 = atten_chunk * (cA_chunk * cA[2] + cB_chunk * cB[2])

        # broadcast (chunk,1) x (n_samples,) -> (chunk, n_samples)
        clean = w1 * p1_carrier + w2 * p2_carrier + w3 * p3_carrier

        # vectorized noise across the whole chunk at once
        cmax = np.maximum(clean.max(axis=1, keepdims=True), 1e-12)
        safe = np.clip(clean / cmax, 0, None)
        shot = rng.poisson(safe * 1000).astype(dtype) / 1000.0 * shot_noise_scale * cmax
        shot -= shot.mean(axis=1, keepdims=True)
        gaussian = rng.normal(0, 1, size=clean.shape).astype(dtype) * (gaussian_noise_std * cmax)
        raw = np.clip(clean + shot + gaussian, 0, None)

        conventional_out[start:end] = raw.mean(axis=1)

        for name, (b, a, freq) in filters.items():
            ref_cos, ref_sin = refs[name]
            X = filtfilt(b, a, raw * ref_cos, axis=1)
            Y = filtfilt(b, a, raw * ref_sin, axis=1)
            X_avg = X[:, transient:].mean(axis=1)
            Y_avg = Y[:, transient:].mean(axis=1)
            I_out[name][start:end] = 2.0 * np.sqrt(X_avg ** 2 + Y_avg ** 2)

    return conventional_out, I_out["2f1"], I_out["f1+f2"], I_out["2f2"]
