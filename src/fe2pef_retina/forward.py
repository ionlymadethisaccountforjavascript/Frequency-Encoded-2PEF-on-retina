"""Forward models for FE-2PEF, including fast and direct lock-in simulation."""

from __future__ import annotations

from dataclasses import asdict
from typing import Iterable

import numpy as np
from scipy.ndimage import gaussian_filter

from .config import DetectorConfig, LaserConfig, PSFConfig
from .lockin import build_reference_matrix
from .noise import first_order_detector_gain, lockin_noise_std, poisson_gaussian_samples
from .spectra import channel_frequency_hz, pathway_response_matrix


def convolve_concentration_maps(
    concentration_maps: np.ndarray,
    pixel_size_um: float,
    psf: PSFConfig,
) -> np.ndarray:
    """Convolve each species map with a Gaussian detection/excitation PSF.

    What happens in this function:
    1. Spatial PSF widths are converted from micrometres to array pixels.
    2. Every species is blurred independently to prevent artificial cross-species mixing.
    3. Two-dimensional and three-dimensional maps are both supported.
    4. The output retains the same shape and total-array convention as the input.
    """

    maps = np.asarray(concentration_maps, dtype=float)
    if maps.ndim not in (3, 4):
        raise ValueError("concentration_maps must be (species,y,x) or (species,z,y,x)")
    if pixel_size_um <= 0:
        raise ValueError("pixel_size_um must be positive")
    if maps.ndim == 3:
        sigma = (psf.sigma_xy_um / pixel_size_um,) * 2
    else:
        sigma = (
            psf.sigma_z_um / pixel_size_um,
            psf.sigma_xy_um / pixel_size_um,
            psf.sigma_xy_um / pixel_size_um,
        )
    return np.stack([gaussian_filter(species_map, sigma=sigma) for species_map in maps])


def effective_signature_with_detector(
    signature_matrix: np.ndarray,
    channels: Iterable[str],
    laser1: LaserConfig,
    laser2: LaserConfig,
    detector: DetectorConfig,
) -> np.ndarray:
    """Apply detector bandwidth attenuation to an FE signature matrix.

    What happens in this function:
    1. Each named channel is mapped to its physical frequency.
    2. A first-order detector gain is evaluated at that frequency.
    3. Each signature row is multiplied by its channel gain.
    4. The result is the matrix that should be used for calibrated inversion.
    """

    matrix = np.asarray(signature_matrix, dtype=float)
    gains = np.array(
        [
            first_order_detector_gain(
                channel_frequency_hz(channel, laser1, laser2),
                detector.detector_bandwidth_hz,
            )
            for channel in channels
        ],
        dtype=float,
    )
    if matrix.shape[0] != len(gains):
        raise ValueError("signature rows must match the number of channels")
    return gains[:, None] * matrix


def simulate_fe_channels_analytic(
    concentration_maps: np.ndarray,
    effective_signature: np.ndarray,
    channels: Iterable[str],
    detector: DetectorConfig,
    rng: np.random.Generator,
    subtract_known_background: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Simulate demodulated FE channels with a lock-in noise approximation.

    What happens in this function:
    1. The effective channel signature is multiplied by every spatial concentration vector.
    2. Relative fluorescence is converted to detected counts per temporal sample.
    3. DC signal sets the Poisson shot-noise level for every lock-in channel.
    4. Gaussian lock-in estimates are drawn using analytic estimator variance.
    5. Known mean background can be subtracted, while its shot noise remains.
    """

    maps = np.asarray(concentration_maps, dtype=float)
    signature = np.asarray(effective_signature, dtype=float)
    channels = tuple(channels)
    if signature.shape != (len(channels), maps.shape[0]):
        raise ValueError("effective_signature must be (channels, species)")

    spatial_shape = maps.shape[1:]
    flat_maps = maps.reshape(maps.shape[0], -1)
    relative_channels = signature @ flat_maps
    scale = detector.photons_per_relative_unit_per_sample
    expected = scale * relative_channels
    n_samples = max(1, int(round(detector.dwell_time_s * detector.sample_rate_hz)))
    background = detector.background_counts_per_sample + detector.dark_counts_per_sample

    try:
        dc_index = channels.index("dc")
        mean_total = np.clip(expected[dc_index] + background, 0.0, None)
    except ValueError:
        mean_total = np.clip(expected.sum(axis=0) + background, 0.0, None)

    measured = np.empty_like(expected)
    standard_deviation = np.empty_like(expected)
    for index, channel in enumerate(channels):
        is_dc = channel == "dc"
        std = lockin_noise_std(
            mean_total,
            n_samples,
            detector.read_noise_std_counts,
            is_dc=is_dc,
        )
        mean = expected[index].copy()
        if is_dc:
            mean += background
        measured[index] = rng.normal(mean, std)
        standard_deviation[index] = std
        if is_dc and subtract_known_background:
            measured[index] -= background

    measured = measured.reshape((len(channels),) + spatial_shape)
    standard_deviation = standard_deviation.reshape((len(channels),) + spatial_shape)
    diagnostics = {
        "samples_per_dwell": float(n_samples),
        "background_counts_per_sample": float(background),
        "mean_dc_counts_per_sample": float(mean_total.mean()),
        "maximum_dc_counts_per_sample": float(mean_total.max()),
    }
    return measured, standard_deviation, diagnostics


def simulate_fe_channels_time_domain(
    concentration_maps: np.ndarray,
    response_frame,
    channels: Iterable[str],
    laser1: LaserConfig,
    laser2: LaserConfig,
    temporal_overlap: float,
    detector: DetectorConfig,
    rng: np.random.Generator,
    chunk_pixels: int = 512,
    subtract_known_background: bool = True,
    add_noise: bool = True,
) -> tuple[np.ndarray, dict[str, float]]:
    """Simulate the detector waveform and digital lock-in projection directly.

    What happens in this function:
    1. Species maps are converted into three pathway-amplitude maps: 11, 12, and 22.
    2. Two intensity-modulated laser waveforms are evaluated across one pixel dwell.
    3. The quadratic fluorescence waveform is built for pixel chunks to limit memory.
    4. Optional Poisson shot noise and Gaussian read noise are added before demodulation.
    5. Digital lock-in references recover DC, fundamentals, and mixed-frequency channels.
    6. Detector bandwidth attenuation is applied to the demodulated amplitudes.
    """

    maps = np.asarray(concentration_maps, dtype=float)
    channels = tuple(channels)
    if maps.ndim not in (3, 4):
        raise ValueError("concentration_maps must be (species,y,x) or (species,z,y,x)")
    if chunk_pixels <= 0:
        raise ValueError("chunk_pixels must be positive")

    n_samples = max(2, int(round(detector.dwell_time_s * detector.sample_rate_hz)))
    time_s = np.arange(n_samples, dtype=float) / detector.sample_rate_hz
    phase1 = 2.0 * np.pi * laser1.modulation_frequency_hz * time_s + laser1.phase_rad
    phase2 = 2.0 * np.pi * laser2.modulation_frequency_hz * time_s + laser2.phase_rad
    intensity1 = laser1.average_power_mw * (
        1.0 + laser1.modulation_depth * np.sin(phase1)
    )
    intensity2 = laser2.average_power_mw * (
        1.0 + laser2.modulation_depth * np.sin(phase2)
    )

    if detector.relative_intensity_noise_std > 0:
        intensity1 *= np.exp(
            rng.normal(0.0, detector.relative_intensity_noise_std, size=n_samples)
            - 0.5 * detector.relative_intensity_noise_std**2
        )
        intensity2 *= np.exp(
            rng.normal(0.0, detector.relative_intensity_noise_std, size=n_samples)
            - 0.5 * detector.relative_intensity_noise_std**2
        )

    pathway = pathway_response_matrix(response_frame)
    flat_maps = maps.reshape(maps.shape[0], -1)
    pathway_maps = pathway @ flat_maps
    pathway11, pathway12, pathway22 = pathway_maps
    reference_weights = build_reference_matrix(channels, time_s, laser1, laser2)
    output = np.empty((len(channels), flat_maps.shape[1]), dtype=float)
    background = detector.background_counts_per_sample + detector.dark_counts_per_sample
    scale = detector.photons_per_relative_unit_per_sample

    for start in range(0, flat_maps.shape[1], chunk_pixels):
        stop = min(start + chunk_pixels, flat_maps.shape[1])
        a11 = pathway11[start:stop, None]
        a12 = pathway12[start:stop, None]
        a22 = pathway22[start:stop, None]
        relative_waveform = (
            a11 * intensity1[None, :] ** 2
            + a22 * intensity2[None, :] ** 2
            + 2.0
            * temporal_overlap
            * a12
            * intensity1[None, :]
            * intensity2[None, :]
        )
        expectation = scale * relative_waveform + background
        if add_noise:
            measured = poisson_gaussian_samples(
                expectation,
                rng,
                detector.read_noise_std_counts,
                detector.saturation_counts,
            )
        else:
            measured = expectation
        demodulated = reference_weights @ measured.T
        output[:, start:stop] = demodulated

    for index, channel in enumerate(channels):
        frequency = channel_frequency_hz(channel, laser1, laser2)
        output[index] *= first_order_detector_gain(
            frequency,
            detector.detector_bandwidth_hz,
        )
        if channel == "dc" and subtract_known_background:
            output[index] -= background

    diagnostics = {
        "samples_per_dwell": float(n_samples),
        "time_window_s": float(detector.dwell_time_s),
        "chunk_pixels": float(chunk_pixels),
        "detector": str(asdict(detector)),
    }
    return output.reshape((len(channels),) + maps.shape[1:]), diagnostics
