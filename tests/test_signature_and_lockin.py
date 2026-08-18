"""Tests for FE signature equations and direct lock-in demodulation."""

from dataclasses import replace
from pathlib import Path

import numpy as np

from fe2pef_retina.config import load_config
from fe2pef_retina.forward import (
    effective_signature_with_detector,
    simulate_fe_channels_time_domain,
)
from fe2pef_retina.spectra import build_frequency_signature_matrix, load_pathway_responses

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_direct_time_domain_matches_analytic_signature() -> None:
    """Check noiseless digital lock-in amplitudes against analytic channel coefficients."""

    config = load_config(PROJECT_ROOT / "configs" / "time_domain_validation.yaml")
    responses = load_pathway_responses(
        PROJECT_ROOT / config.pathway_responses_csv,
        config.species,
    )
    raw = build_frequency_signature_matrix(
        responses,
        config.laser1,
        config.laser2,
        config.pulse.temporal_overlap,
        config.simulation.channels,
    )
    effective = effective_signature_with_detector(
        raw,
        config.simulation.channels,
        config.laser1,
        config.laser2,
        config.detector,
    )
    maps = np.zeros((2, 2, 2), dtype=float)
    maps[0, 0, 0] = 1.0
    maps[1, 0, 1] = 1.0
    detector = replace(
        config.detector,
        background_counts_per_sample=0.0,
        dark_counts_per_sample=0.0,
        read_noise_std_counts=0.0,
        relative_intensity_noise_std=0.0,
    )
    direct, _ = simulate_fe_channels_time_domain(
        maps,
        responses,
        config.simulation.channels,
        config.laser1,
        config.laser2,
        config.pulse.temporal_overlap,
        detector,
        np.random.default_rng(0),
        chunk_pixels=4,
        add_noise=False,
    )
    expected = (
        detector.photons_per_relative_unit_per_sample
        * effective
        @ maps.reshape(2, -1)
    ).reshape(direct.shape)
    assert np.allclose(direct, expected, rtol=1e-10, atol=1e-10)
