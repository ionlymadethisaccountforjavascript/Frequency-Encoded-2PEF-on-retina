"""
Phase 1: Forward Simulation (Generating the Synthetic Signal)
================================================================
Step 1: Excitation Source and Encoding
Step 2: Multiplexed Excitation Pathways (Focal Volume)
Step 3: Single Detector Accumulation

Produces, per pixel, a time-domain "total raw signal" as if a single
filterless PMT/detector recorded the combined nonlinear emission while the
two lasers (modulated at f1, f2) were parked at that focal spot.
"""

import numpy as np


class ExcitationSource:
    """Step 1: two pulsed lasers, each intensity-modulated at its own
    (comparatively low, radio-frequency-scale) modulation frequency. The
    'pulsed' laser carrier itself (e.g. 80 MHz pulse train) is not simulated
    sample-by-sample -- what matters for lock-in demultiplexing is the slow
    intensity modulation envelope, so t/fs/sample_rate here describe that
    envelope timescale, not the femtosecond pulse train itself."""

    def __init__(self, duration_s=1e-3, sample_rate_hz=1e6, f1_hz=1.1e5, f2_hz=1.3e5):
        self.duration_s = duration_s
        self.sample_rate_hz = sample_rate_hz
        self.f1 = f1_hz
        self.f2 = f2_hz
        self.t = np.arange(0, duration_s, 1.0 / sample_rate_hz)

    def modulated_carriers(self):
        """Return the intensity-modulated envelopes for laser 1 and laser 2.
        Modulation depth is kept at 1.0 (full AM) with a small DC offset so
        both carriers stay non-negative, matching physical laser intensity."""
        m1 = 0.5 * (1 + np.sin(2 * np.pi * self.f1 * self.t))
        m2 = 0.5 * (1 + np.sin(2 * np.pi * self.f2 * self.t))
        return m1, m2


class FocalVolumeMixer:
    """
    Step 2: implements the three non-linear pathways at a single focal spot
    given local fluorophore concentrations (conc_A, conc_B).

      Path 1 (pure L1):     signal ~ conc * m1^2                 -> beats at 2*f1
      Path 2 (cross L1xL2): signal ~ conc * m1*m2                -> beats at f1+f2
      Path 3 (pure L2):     signal ~ conc * m2^2                 -> beats at 2*f2

    Each fluorophore has a distinct 3-vector of nonlinear susceptibility
    coefficients across (Path1, Path2, Path3) -- this is the "mixing matrix"
    NNMF later has to recover. A2E and lipofuscin are chosen to have
    overlapping-but-distinct coefficient vectors, reflecting that their
    excitation efficiencies for the sum-frequency (cross) pathway differ from
    their two pure-pathway efficiencies (a real consequence of their distinct,
    if broad, two-photon action cross-sections).
    """

    # (Path1, Path2, Path3) nonlinear response coefficients per fluorophore.
    # Illustrative values, but the cosine similarity between the two vectors
    # (~0.2 here) is the single biggest lever on unmixing quality -- if the
    # two vectors are too collinear, NNMF cannot separate the species no
    # matter how much averaging or noise reduction is applied downstream.
    # Keep them non-parallel and all-positive (a physically valid emission
    # profile) if you swap in calibrated values.
    COEFFS_A2E = np.array([1.00, 0.35, 0.05])         # A2E: dominant on pure-L1 pathway
    COEFFS_LIPOFUSCIN = np.array([0.05, 0.35, 1.00])  # lipofuscin: dominant on pure-L2

    def __init__(self, coeffs_A=None, coeffs_B=None):
        self.coeffs_A = self.COEFFS_A2E if coeffs_A is None else np.asarray(coeffs_A)
        self.coeffs_B = self.COEFFS_LIPOFUSCIN if coeffs_B is None else np.asarray(coeffs_B)

    def pathway_signals(self, m1, m2, conc_A, conc_B, atten=1.0):
        """
        Returns path1, path2, path3 time-domain signals (pre-summation) for a
        single pixel, given scalar fluorophore concentrations and a scalar
        excitation/emission attenuation factor for that pixel's depth.
        """
        p1_carrier = m1 ** 2
        p2_carrier = m1 * m2
        p3_carrier = m2 ** 2

        cA, cB = self.coeffs_A, self.coeffs_B
        path1 = atten * (conc_A * cA[0] + conc_B * cB[0]) * p1_carrier
        path2 = atten * (conc_A * cA[1] + conc_B * cB[1]) * p2_carrier
        path3 = atten * (conc_A * cA[2] + conc_B * cB[2]) * p3_carrier
        return path1, path2, path3


class DetectorModel:
    """Step 3: sums the three pathways and adds realistic detector noise."""

    def __init__(self, shot_noise_scale=0.02, gaussian_noise_std=0.01, rng=None):
        self.shot_noise_scale = shot_noise_scale
        self.gaussian_noise_std = gaussian_noise_std
        self.rng = rng or np.random.default_rng(0)

    def accumulate(self, path1, path2, path3):
        clean = path1 + path2 + path3
        # Poisson-like shot noise scaled to signal magnitude, plus Gaussian
        # read noise. We keep signal >= 0 as a real photon-counting detector
        # would produce.
        safe = np.clip(clean / max(clean.max(), 1e-12), 0, None)
        shot = self.rng.poisson(safe * 1000) / 1000.0 * self.shot_noise_scale * clean.max()
        gaussian = self.rng.normal(0, self.gaussian_noise_std * (clean.max() + 1e-9), size=clean.shape)
        noisy = clean + shot - shot.mean() + gaussian
        return np.clip(noisy, 0, None)


def simulate_pixel(source: ExcitationSource, mixer: FocalVolumeMixer,
                    detector: DetectorModel, conc_A: float, conc_B: float,
                    atten: float = 1.0):
    """Convenience wrapper: full Phase 1 pipeline for one pixel."""
    m1, m2 = source.modulated_carriers()
    p1, p2, p3 = mixer.pathway_signals(m1, m2, conc_A, conc_B, atten)
    raw = detector.accumulate(p1, p2, p3)
    return raw
