"""Small, transparent optical calculations used by the forward model."""

from __future__ import annotations

import math

import numpy as np

PLANCK_J_S = 6.62607015e-34
LIGHT_SPEED_M_S = 299792458.0


def photon_energy_joule(wavelength_nm: float) -> float:
    """Return the energy of one photon at the requested wavelength.

    What happens in this function:
    1. Nanometres are converted to metres.
    2. The relation E = h*c/lambda is evaluated.
    3. The result is returned in joules per photon.
    """

    if wavelength_nm <= 0:
        raise ValueError("wavelength_nm must be positive")
    wavelength_m = wavelength_nm * 1e-9
    return PLANCK_J_S * LIGHT_SPEED_M_S / wavelength_m


def effective_one_photon_wavelength_nm(
    wavelength1_nm: float,
    wavelength2_nm: float | None = None,
) -> float:
    """Convert a two-photon pathway into its one-photon-equivalent wavelength.

    What happens in this function:
    1. If one wavelength is supplied twice, a degenerate two-photon pathway is used.
    2. Photon energies are added through reciprocal wavelengths.
    3. The equivalent one-photon wavelength is returned for spectral interpretation.
    """

    if wavelength2_nm is None:
        wavelength2_nm = wavelength1_nm
    if wavelength1_nm <= 0 or wavelength2_nm <= 0:
        raise ValueError("wavelengths must be positive")
    return 1.0 / (1.0 / wavelength1_nm + 1.0 / wavelength2_nm)


def gaussian_pulse_peak_power_w(
    average_power_mw: float,
    repetition_rate_hz: float,
    pulse_width_fs: float,
) -> float:
    """Estimate peak power for a Gaussian pulse train.

    What happens in this function:
    1. Average power is divided by repetition rate to obtain pulse energy.
    2. Pulse FWHM is converted from femtoseconds to seconds.
    3. A Gaussian-area factor converts pulse energy to peak power.
    4. The result is an approximation and does not include optical losses.
    """

    if average_power_mw <= 0 or repetition_rate_hz <= 0 or pulse_width_fs <= 0:
        raise ValueError("power, repetition rate, and pulse width must be positive")
    pulse_energy_j = average_power_mw * 1e-3 / repetition_rate_hz
    pulse_width_s = pulse_width_fs * 1e-15
    gaussian_area_factor = math.sqrt(math.pi / (4.0 * math.log(2.0)))
    return pulse_energy_j / (pulse_width_s * gaussian_area_factor)


def relative_two_photon_dose(
    average_power_mw: float,
    repetition_rate_hz: float,
    pulse_width_fs: float,
    beam_sigma_um: float,
) -> float:
    """Return a relative two-photon excitation dose for scaling studies.

    What happens in this function:
    1. Peak power is estimated from the pulsed laser parameters.
    2. A Gaussian focal area is computed from the supplied spatial sigma.
    3. The square of peak intensity is multiplied by pulse duty time.
    4. The value is relative; absolute fluorescence needs calibrated cross-sections.
    """

    if beam_sigma_um <= 0:
        raise ValueError("beam_sigma_um must be positive")
    peak_power_w = gaussian_pulse_peak_power_w(
        average_power_mw,
        repetition_rate_hz,
        pulse_width_fs,
    )
    sigma_m = beam_sigma_um * 1e-6
    effective_area_m2 = 2.0 * math.pi * sigma_m**2
    peak_intensity = peak_power_w / effective_area_m2
    duty_time_s = pulse_width_fs * 1e-15 * repetition_rate_hz
    return float(peak_intensity**2 * duty_time_s)


def gaussian_sigma_from_fwhm(fwhm: float) -> float:
    """Convert a Gaussian full width at half maximum into standard deviation.

    What happens in this function:
    1. The analytic Gaussian conversion factor is evaluated.
    2. The result uses the same units as the supplied FWHM.
    """

    if fwhm <= 0:
        raise ValueError("fwhm must be positive")
    return float(fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0))))


def approximate_2pef_lateral_fwhm_um(wavelength_nm: float, numerical_aperture: float) -> float:
    """Estimate lateral 2PEF FWHM with a commonly used diffraction approximation.

    What happens in this function:
    1. The wavelength is converted from nanometres to micrometres.
    2. A 0.37*lambda/NA approximation is evaluated.
    3. The result is intended for sensitivity analysis, not safety certification.
    """

    if wavelength_nm <= 0 or numerical_aperture <= 0:
        raise ValueError("wavelength and numerical aperture must be positive")
    return float(0.37 * wavelength_nm * 1e-3 / numerical_aperture)


def approximate_2pef_axial_fwhm_um(
    wavelength_nm: float,
    numerical_aperture: float,
    refractive_index: float = 1.336,
) -> float:
    """Estimate axial 2PEF FWHM with a paraxial approximation.

    What happens in this function:
    1. Wavelength is converted into micrometres.
    2. Refractive index and numerical aperture set the axial scale.
    3. The approximation should be replaced by measured PSF data when available.
    """

    if wavelength_nm <= 0 or numerical_aperture <= 0 or refractive_index <= 0:
        raise ValueError("all optical inputs must be positive")
    return float(0.64 * wavelength_nm * 1e-3 * refractive_index / numerical_aperture**2)
