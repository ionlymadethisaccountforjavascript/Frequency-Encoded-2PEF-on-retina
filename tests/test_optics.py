"""Tests for transparent optical helper calculations."""

import numpy as np

from fe2pef_retina.optics import (
    effective_one_photon_wavelength_nm,
    gaussian_pulse_peak_power_w,
    photon_energy_joule,
)


def test_degenerate_effective_wavelength_is_half() -> None:
    """Check that two equal 900-nm photons equal one 450-nm photon in energy."""

    assert np.isclose(effective_one_photon_wavelength_nm(900.0), 450.0)


def test_mixed_effective_wavelength() -> None:
    """Check reciprocal-wavelength addition for a two-colour pathway."""

    expected = 1.0 / (1.0 / 740.0 + 1.0 / 910.0)
    assert np.isclose(effective_one_photon_wavelength_nm(740.0, 910.0), expected)


def test_photon_energy_and_peak_power_positive() -> None:
    """Check that physically valid inputs produce positive optical quantities."""

    assert photon_energy_joule(900.0) > 0
    assert gaussian_pulse_peak_power_w(1.0, 80e6, 100.0) > 0
