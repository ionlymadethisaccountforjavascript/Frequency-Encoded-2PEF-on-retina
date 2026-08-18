"""End-to-end experiments that generate reproducible figures, arrays, and tables."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .baselines import (
    conventional_single_detector_image,
    load_emission_matrix,
    simulate_emission_filtered_channels,
)
from .config import ProjectConfig, load_config
from .forward import (
    convolve_concentration_maps,
    effective_signature_with_detector,
    simulate_fe_channels_analytic,
    simulate_fe_channels_time_domain,
)
from .io import ensure_directory, save_array_bundle, save_json, save_rows_csv
from .metrics import component_metrics, nrmse
from .phantoms import (
    overlap_phantom,
    rpe_mosaic_phantom,
    rpe_volume_phantom,
    two_spot_phantom,
)
from .plotting import (
    save_image_grid,
    save_metric_bar_chart,
    save_signature_heatmap,
    save_sweep_plot,
)
from .spectra import (
    build_frequency_signature_matrix,
    condition_number,
    load_pathway_responses,
)
from .unmixing import (
    align_components,
    calibrated_nnls_unmix,
    regularized_nmf,
    reshape_nmf_concentrations,
)


def _resolve_project_path(project_root: Path, configured_path: str) -> Path:
    """Resolve a configuration path relative to the project root.

    What happens in this function:
    1. Absolute paths are returned unchanged.
    2. Relative paths are joined to the supplied project root.
    3. The result is resolved for stable provenance output.
    """

    path = Path(configured_path)
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def _make_phantom(config: ProjectConfig, rng: np.random.Generator) -> tuple[np.ndarray, dict[str, Any]]:
    """Generate the concentration maps requested by an experiment configuration.

    What happens in this function:
    1. The configured phantom kind selects a dedicated generator.
    2. Morphology parameters are forwarded without hidden changes.
    3. A metadata dictionary records labels or generator assumptions.
    4. Current built-in generators produce two species; user maps can extend this later.
    """

    kind = config.phantom.kind
    metadata: dict[str, Any] = {"kind": kind}
    if kind == "two_spot":
        maps = two_spot_phantom(
            config.grid.shape,
            config.grid.pixel_size_um,
            config.phantom.separation_um,
        )
    elif kind == "overlap":
        maps = overlap_phantom(
            config.grid.shape,
            config.grid.pixel_size_um,
            config.phantom.overlap_fraction,
            rng,
        )
    elif kind == "rpe_mosaic":
        maps, labels = rpe_mosaic_phantom(
            config.grid.shape,
            config.grid.pixel_size_um,
            config.phantom.n_cells,
            config.phantom.granules_per_cell,
            config.phantom.overlap_fraction,
            rng,
        )
        metadata["cell_labels"] = labels
    elif kind == "rpe_volume":
        maps = rpe_volume_phantom(config.grid.shape, config.grid.pixel_size_um, rng)
    else:
        raise ValueError(f"unsupported phantom kind: {kind}")
    if maps.shape[0] != len(config.species):
        raise ValueError(
            "built-in phantom species count does not match configuration; "
            "use a two-species configuration or provide custom maps"
        )
    return maps, metadata


def _simulate_fe(
    config: ProjectConfig,
    maps: np.ndarray,
    response_frame: pd.DataFrame,
    effective_signature: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Dispatch FE simulation to analytic or direct time-domain mode.

    What happens in this function:
    1. The configuration's simulation mode is inspected.
    2. Analytic mode uses lock-in estimator variance for fast sweeps.
    3. Time-domain mode creates noisy detector waveforms in pixel chunks.
    4. Both modes return channel arrays with identical axis conventions.
    """

    if config.simulation.mode == "analytic":
        channels, channel_std, diagnostics = simulate_fe_channels_analytic(
            maps,
            effective_signature,
            config.simulation.channels,
            config.detector,
            rng,
            subtract_known_background=config.simulation.subtract_known_background,
        )
        diagnostics["channel_noise_std_mean"] = float(np.mean(channel_std))
        return channels, diagnostics
    channels, diagnostics = simulate_fe_channels_time_domain(
        maps,
        response_frame,
        config.simulation.channels,
        config.laser1,
        config.laser2,
        config.pulse.temporal_overlap,
        config.detector,
        rng,
        chunk_pixels=config.simulation.chunk_pixels,
        subtract_known_background=config.simulation.subtract_known_background,
        add_noise=True,
    )
    return channels, diagnostics


def run_demo(
    config_path: str | Path,
    output_dir: str | Path,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """Run the main retinal FE-2PEF reconstruction comparison.

    What happens in this function:
    1. Configuration, response tables, and a synthetic retinal phantom are loaded.
    2. A Gaussian PSF blurs each species concentration map.
    3. FE lock-in channels are simulated and unmixed by calibrated NNLS and blind NNMF.
    4. A conventional emission-filtered baseline is simulated at reduced throughput.
    5. Images, signatures, arrays, metrics, diagnostics, and provenance are saved.
    6. The returned dictionary summarizes key output paths and condition numbers.
    """

    config_path = Path(config_path).resolve()
    root = Path(project_root).resolve() if project_root else config_path.parent.parent
    output = ensure_directory(output_dir)
    config = load_config(config_path)
    rng = np.random.default_rng(config.seed)

    response_path = _resolve_project_path(root, config.pathway_responses_csv)
    response_frame = load_pathway_responses(response_path, config.species)
    concentration_maps, phantom_metadata = _make_phantom(config, rng)
    blurred_maps = convolve_concentration_maps(
        concentration_maps,
        config.grid.pixel_size_um,
        config.psf,
    )

    raw_signature = build_frequency_signature_matrix(
        response_frame,
        config.laser1,
        config.laser2,
        config.pulse.temporal_overlap,
        config.simulation.channels,
    )
    effective_signature = effective_signature_with_detector(
        raw_signature,
        config.simulation.channels,
        config.laser1,
        config.laser2,
        config.detector,
    )
    fe_channels, fe_diagnostics = _simulate_fe(
        config,
        blurred_maps,
        response_frame,
        effective_signature,
        rng,
    )
    calibrated_signature_counts = (
        config.detector.photons_per_relative_unit_per_sample * effective_signature
    )
    fe_nnls = calibrated_nnls_unmix(fe_channels, calibrated_signature_counts)

    flat_fe = np.clip(fe_channels.reshape(fe_channels.shape[0], -1), 0.0, None)
    nmf_result = regularized_nmf(
        flat_fe,
        config.unmixing.components,
        config.unmixing.alpha_l1,
        config.unmixing.alpha_l2,
        config.unmixing.iterations,
        config.unmixing.restarts,
        config.unmixing.tolerance,
        rng,
    )
    fe_nmf = reshape_nmf_concentrations(nmf_result, blurred_maps.shape[1:])
    fe_nmf_aligned, nmf_order = align_components(fe_nmf, blurred_maps)

    emission_path = _resolve_project_path(root, config.baseline.emission_matrix_csv)
    emission_matrix, detector_names, emission_frame = load_emission_matrix(
        emission_path,
        config.species,
    )
    emission_channels, emission_std, emission_diagnostics = simulate_emission_filtered_channels(
        blurred_maps,
        emission_matrix,
        config.baseline.optical_throughput,
        config.detector,
        rng,
        subtract_known_background=config.simulation.subtract_known_background,
    )
    emission_signature_counts = (
        config.detector.photons_per_relative_unit_per_sample
        * config.baseline.optical_throughput
        * emission_matrix
    )
    emission_nnls = calibrated_nnls_unmix(emission_channels, emission_signature_counts)
    conventional = conventional_single_detector_image(
        blurred_maps,
        response_frame["brightness"].to_numpy(float),
    )

    metric_rows: list[dict[str, Any]] = []
    for method, maps in (
        ("FE calibrated NNLS", fe_nnls),
        ("FE blind regularized NNMF", fe_nmf_aligned),
        ("Emission-filtered calibrated NNLS", emission_nnls),
    ):
        rows = component_metrics(
            blurred_maps,
            maps,
            list(config.species),
            config.grid.pixel_size_um,
        )
        for row in rows:
            row["method"] = method
            metric_rows.append(row)
    metrics_frame = pd.DataFrame(metric_rows)

    save_image_grid(
        list(blurred_maps),
        [f"Ground truth: {name}" for name in config.species],
        output / "ground_truth.png",
        columns=len(config.species),
    )
    save_image_grid(
        list(fe_channels),
        [f"FE raw: {name}" for name in config.simulation.channels],
        output / "fe_raw_channels.png",
        columns=2,
    )
    save_image_grid(
        list(fe_nnls),
        [f"FE NNLS: {name}" for name in config.species],
        output / "fe_nnls_reconstruction.png",
        columns=len(config.species),
    )
    save_image_grid(
        list(fe_nmf_aligned),
        [f"FE blind NNMF: {name}" for name in config.species],
        output / "fe_nmf_reconstruction.png",
        columns=len(config.species),
    )
    save_image_grid(
        list(emission_nnls),
        [f"Emission baseline: {name}" for name in config.species],
        output / "emission_reconstruction.png",
        columns=len(config.species),
    )
    save_image_grid(
        [conventional],
        ["Conventional single-detector summed TPEF"],
        output / "conventional_single_detector.png",
        columns=1,
    )
    save_signature_heatmap(
        effective_signature,
        config.simulation.channels,
        config.species,
        output / "fe_signature_matrix.png",
    )
    save_metric_bar_chart(metrics_frame, output / "method_nrmse_comparison.png")

    metrics_frame.to_csv(output / "metrics.csv", index=False)
    response_frame.to_csv(output / "pathway_responses_used.csv", index=False)
    emission_frame.to_csv(output / "emission_responses_used.csv", index=False)
    save_array_bundle(
        output / "arrays.npz",
        concentration_maps=concentration_maps,
        blurred_ground_truth=blurred_maps,
        fe_channels=fe_channels,
        fe_nnls=fe_nnls,
        fe_nmf_aligned=fe_nmf_aligned,
        emission_channels=emission_channels,
        emission_nnls=emission_nnls,
        conventional_single_detector=conventional,
        raw_signature=raw_signature,
        effective_signature=effective_signature,
        nmf_signature=nmf_result.signature,
        nmf_component_order=nmf_order,
        emission_noise_std=emission_std,
    )
    diagnostics = {
        "config": config,
        "config_path": str(config_path),
        "response_path": str(response_path),
        "emission_path": str(emission_path),
        "fe_signature_condition_number": condition_number(effective_signature),
        "emission_signature_condition_number": condition_number(emission_matrix),
        "fe": fe_diagnostics,
        "emission": emission_diagnostics,
        "blind_nmf_objective": nmf_result.objective,
        "blind_nmf_iterations": nmf_result.iterations,
        "phantom_metadata_keys": sorted(phantom_metadata.keys()),
        "detector_names": detector_names,
    }
    save_json(output / "diagnostics.json", diagnostics)
    return {
        "output_dir": str(output),
        "metrics_csv": str(output / "metrics.csv"),
        "arrays_npz": str(output / "arrays.npz"),
        "fe_condition_number": diagnostics["fe_signature_condition_number"],
    }


def run_channel_validation(
    config_path: str | Path,
    output_dir: str | Path,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """Validate the analytic FE signature against direct waveform demodulation.

    What happens in this function:
    1. Pure constant maps isolate one species at a time.
    2. Noiseless time-domain waveforms are generated and digitally demodulated.
    3. Analytic signature predictions are converted into the same count units.
    4. Relative channel errors and finite-window reference leakage are saved.
    5. This unit-style validation checks implementation, not biological accuracy.
    """

    config_path = Path(config_path).resolve()
    root = Path(project_root).resolve() if project_root else config_path.parent.parent
    output = ensure_directory(output_dir)
    config = load_config(config_path)
    response_path = _resolve_project_path(root, config.pathway_responses_csv)
    response_frame = load_pathway_responses(response_path, config.species)
    rng = np.random.default_rng(config.seed)

    maps = np.zeros((len(config.species), 2, 2), dtype=float)
    for index in range(len(config.species)):
        maps[index, index // 2, index % 2] = 1.0

    raw_signature = build_frequency_signature_matrix(
        response_frame,
        config.laser1,
        config.laser2,
        config.pulse.temporal_overlap,
        config.simulation.channels,
    )
    effective_signature = effective_signature_with_detector(
        raw_signature,
        config.simulation.channels,
        config.laser1,
        config.laser2,
        config.detector,
    )
    direct, diagnostics = simulate_fe_channels_time_domain(
        maps,
        response_frame,
        config.simulation.channels,
        config.laser1,
        config.laser2,
        config.pulse.temporal_overlap,
        replace(
            config.detector,
            background_counts_per_sample=0.0,
            dark_counts_per_sample=0.0,
            read_noise_std_counts=0.0,
            relative_intensity_noise_std=0.0,
            saturation_counts=None,
        ),
        rng,
        chunk_pixels=4,
        subtract_known_background=True,
        add_noise=False,
    )
    expected = (
        config.detector.photons_per_relative_unit_per_sample
        * effective_signature
        @ maps.reshape(len(config.species), -1)
    ).reshape(direct.shape)
    absolute_error = np.abs(direct - expected)
    relative_error = absolute_error / np.maximum(np.abs(expected), 1e-12)

    rows = []
    for channel_index, channel in enumerate(config.simulation.channels):
        for species_index, species in enumerate(config.species):
            row = species_index // 2
            column = species_index % 2
            rows.append(
                {
                    "channel": channel,
                    "species": species,
                    "direct": float(direct[channel_index, row, column]),
                    "analytic": float(expected[channel_index, row, column]),
                    "absolute_error": float(absolute_error[channel_index, row, column]),
                    "relative_error": float(relative_error[channel_index, row, column]),
                }
            )
    save_rows_csv(output / "channel_validation.csv", rows)
    save_array_bundle(
        output / "channel_validation_arrays.npz",
        direct=direct,
        expected=expected,
        relative_error=relative_error,
    )
    summary = {
        "maximum_relative_error": float(np.nanmax(relative_error)),
        "mean_relative_error": float(np.nanmean(relative_error)),
        "diagnostics": diagnostics,
    }
    save_json(output / "channel_validation_summary.json", summary)
    return summary


def _mean_component_nrmse(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Return mean scale-adjusted NRMSE across all species.

    What happens in this function:
    1. Each estimated species map receives its best nonnegative scalar fit.
    2. NRMSE is calculated against the corresponding reference map.
    3. Species errors are averaged into one sweep-friendly number.
    """

    errors = []
    for ref, est in zip(reference, estimate):
        scale = max(float(np.sum(ref * est) / max(np.sum(est**2), 1e-12)), 0.0)
        errors.append(nrmse(ref, est * scale))
    return float(np.mean(errors))


def run_robustness_sweeps(
    config_path: str | Path,
    output_dir: str | Path,
    project_root: str | Path | None = None,
    replicates: int = 6,
) -> dict[str, Any]:
    """Run the complete FE-versus-emission robustness study.

    What happens in this function:
    1. A small two-species phantom keeps the sweeps computationally affordable.
    2. Additive background is varied to compare FE and emission filtering.
    3. Spot separation tests spatial overlap without claiming super-resolution.
    4. Brightness imbalance tests the shared-detector dynamic-range limitation.
    5. FE-signature and emission-signature similarity reveal conditioning failures.
    6. Photon budget, detector bandwidth, pulse overlap, and filter throughput are swept.
    7. Every replicate is saved to CSV and summarized by uncertainty plots.
    """

    if replicates <= 0:
        raise ValueError("replicates must be positive")
    config_path = Path(config_path).resolve()
    root = Path(project_root).resolve() if project_root else config_path.parent.parent
    output = ensure_directory(output_dir)
    config = load_config(config_path)
    response_path = _resolve_project_path(root, config.pathway_responses_csv)
    base_responses = load_pathway_responses(response_path, config.species)
    emission_path = _resolve_project_path(root, config.baseline.emission_matrix_csv)
    emission_matrix, _, _ = load_emission_matrix(emission_path, config.species)

    sweep_shape = (48, 48)
    pixel_size = config.grid.pixel_size_um
    base_maps = overlap_phantom(
        sweep_shape,
        pixel_size,
        overlap_fraction=0.55,
        rng=np.random.default_rng(config.seed + 10),
    )
    base_maps = convolve_concentration_maps(base_maps, pixel_size, config.psf)

    def evaluate(
        maps: np.ndarray,
        responses: pd.DataFrame,
        detector_config,
        rng: np.random.Generator,
        temporal_overlap: float | None = None,
        emission_matrix_override: np.ndarray | None = None,
        optical_throughput: float | None = None,
    ) -> tuple[float, float, float]:
        """Simulate and score calibrated FE and emission reconstructions.

        What happens in this nested function:
        1. Optional stress-test values override pulse overlap, emission mixing, or throughput.
        2. FE signatures are rebuilt for the current response and detector settings.
        3. Fast analytic noise simulation generates FE and emission-filtered channels.
        4. Calibrated NNLS recovers both sets of species maps under matched detector noise.
        5. Mean NRMSE and the FE signature condition number are returned.
        """

        overlap = (
            config.pulse.temporal_overlap
            if temporal_overlap is None
            else float(temporal_overlap)
        )
        comparison_matrix = (
            emission_matrix
            if emission_matrix_override is None
            else np.asarray(emission_matrix_override, dtype=float)
        )
        throughput = (
            config.baseline.optical_throughput
            if optical_throughput is None
            else float(optical_throughput)
        )
        raw = build_frequency_signature_matrix(
            responses,
            config.laser1,
            config.laser2,
            overlap,
            config.simulation.channels,
        )
        effective = effective_signature_with_detector(
            raw,
            config.simulation.channels,
            config.laser1,
            config.laser2,
            detector_config,
        )
        fe_channels, _, _ = simulate_fe_channels_analytic(
            maps,
            effective,
            config.simulation.channels,
            detector_config,
            rng,
            subtract_known_background=True,
        )
        fe_estimate = calibrated_nnls_unmix(
            fe_channels,
            detector_config.photons_per_relative_unit_per_sample * effective,
        )
        emission_channels, _, _ = simulate_emission_filtered_channels(
            maps,
            comparison_matrix,
            throughput,
            detector_config,
            rng,
            subtract_known_background=True,
        )
        emission_estimate = calibrated_nnls_unmix(
            emission_channels,
            detector_config.photons_per_relative_unit_per_sample
            * throughput
            * comparison_matrix,
        )
        return (
            _mean_component_nrmse(maps, fe_estimate),
            _mean_component_nrmse(maps, emission_estimate),
            condition_number(effective),
        )

    noise_rows: list[dict[str, Any]] = []
    for background in (0.0, 0.02, 0.08, 0.25, 0.8, 2.0):
        detector = replace(config.detector, background_counts_per_sample=background)
        for replicate in range(replicates):
            rng = np.random.default_rng(config.seed + 1000 + replicate + int(background * 1000))
            fe_error, emission_error, cond = evaluate(base_maps, base_responses, detector, rng)
            noise_rows.extend(
                [
                    {
                        "background_counts_per_sample": background,
                        "replicate": replicate,
                        "method": "FE calibrated NNLS",
                        "mean_nrmse": fe_error,
                        "condition_number": cond,
                    },
                    {
                        "background_counts_per_sample": background,
                        "replicate": replicate,
                        "method": "Emission-filtered NNLS",
                        "mean_nrmse": emission_error,
                        "condition_number": condition_number(emission_matrix),
                    },
                ]
            )

    separation_rows: list[dict[str, Any]] = []
    for separation in (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 25.0):
        maps = two_spot_phantom(sweep_shape, pixel_size, separation, sigma_um=1.2)
        maps = convolve_concentration_maps(maps, pixel_size, config.psf)
        for replicate in range(replicates):
            rng = np.random.default_rng(config.seed + 2000 + replicate + int(separation * 10))
            fe_error, emission_error, cond = evaluate(maps, base_responses, config.detector, rng)
            separation_rows.extend(
                [
                    {
                        "separation_um": separation,
                        "replicate": replicate,
                        "method": "FE calibrated NNLS",
                        "mean_nrmse": fe_error,
                        "condition_number": cond,
                    },
                    {
                        "separation_um": separation,
                        "replicate": replicate,
                        "method": "Emission-filtered NNLS",
                        "mean_nrmse": emission_error,
                        "condition_number": condition_number(emission_matrix),
                    },
                ]
            )

    brightness_rows: list[dict[str, Any]] = []
    for ratio in (1.0, 2.0, 5.0, 10.0, 20.0, 50.0):
        maps = base_maps.copy()
        maps[1] *= ratio
        for replicate in range(replicates):
            rng = np.random.default_rng(config.seed + 3000 + replicate + int(ratio))
            fe_error, emission_error, cond = evaluate(maps, base_responses, config.detector, rng)
            brightness_rows.extend(
                [
                    {
                        "brightness_ratio_species2_to_species1": ratio,
                        "replicate": replicate,
                        "method": "FE calibrated NNLS",
                        "mean_nrmse": fe_error,
                        "condition_number": cond,
                    },
                    {
                        "brightness_ratio_species2_to_species1": ratio,
                        "replicate": replicate,
                        "method": "Emission-filtered NNLS",
                        "mean_nrmse": emission_error,
                        "condition_number": condition_number(emission_matrix),
                    },
                ]
            )

    similarity_rows: list[dict[str, Any]] = []
    numeric_columns = ["sigma_11", "sigma_12", "sigma_22"]
    first_signature = base_responses.loc[0, numeric_columns].to_numpy(float)
    original_second = base_responses.loc[1, numeric_columns].to_numpy(float)
    for similarity in (0.0, 0.25, 0.5, 0.75, 0.9, 0.97, 0.995):
        responses = base_responses.copy()
        responses.loc[1, numeric_columns] = (
            (1.0 - similarity) * original_second + similarity * first_signature
        )
        for replicate in range(replicates):
            rng = np.random.default_rng(config.seed + 4000 + replicate + int(similarity * 1000))
            fe_error, _, cond = evaluate(base_maps, responses, config.detector, rng)
            similarity_rows.append(
                {
                    "signature_similarity": similarity,
                    "replicate": replicate,
                    "method": "FE calibrated NNLS",
                    "mean_nrmse": fe_error,
                    "condition_number": cond,
                }
            )

    noise_frame = pd.DataFrame(noise_rows)
    separation_frame = pd.DataFrame(separation_rows)
    brightness_frame = pd.DataFrame(brightness_rows)
    photon_rows: list[dict[str, Any]] = []
    for photon_scale in (0.05, 0.1, 0.2, 0.35, 0.7, 1.4):
        detector = replace(
            config.detector,
            photons_per_relative_unit_per_sample=photon_scale,
        )
        for replicate in range(replicates):
            rng = np.random.default_rng(
                config.seed + 5000 + replicate + int(photon_scale * 1000)
            )
            fe_error, emission_error, cond = evaluate(
                base_maps, base_responses, detector, rng
            )
            photon_rows.extend(
                [
                    {
                        "photons_per_relative_unit_per_sample": photon_scale,
                        "replicate": replicate,
                        "method": "FE calibrated NNLS",
                        "mean_nrmse": fe_error,
                        "condition_number": cond,
                    },
                    {
                        "photons_per_relative_unit_per_sample": photon_scale,
                        "replicate": replicate,
                        "method": "Emission-filtered NNLS",
                        "mean_nrmse": emission_error,
                        "condition_number": condition_number(emission_matrix),
                    },
                ]
            )

    bandwidth_rows: list[dict[str, Any]] = []
    for bandwidth_hz in (1.8e6, 2.5e6, 4.0e6, 6.0e6, 12.0e6, 20.0e6):
        detector = replace(config.detector, detector_bandwidth_hz=bandwidth_hz)
        for replicate in range(replicates):
            rng = np.random.default_rng(
                config.seed + 6000 + replicate + int(bandwidth_hz / 1000)
            )
            fe_error, emission_error, cond = evaluate(
                base_maps, base_responses, detector, rng
            )
            bandwidth_rows.extend(
                [
                    {
                        "detector_bandwidth_hz": bandwidth_hz,
                        "replicate": replicate,
                        "method": "FE calibrated NNLS",
                        "mean_nrmse": fe_error,
                        "condition_number": cond,
                    },
                    {
                        "detector_bandwidth_hz": bandwidth_hz,
                        "replicate": replicate,
                        "method": "Emission-filtered NNLS",
                        "mean_nrmse": emission_error,
                        "condition_number": condition_number(emission_matrix),
                    },
                ]
            )

    overlap_rows: list[dict[str, Any]] = []
    for temporal_overlap in (0.0, 0.2, 0.5, 0.75, 0.9, 1.0):
        for replicate in range(replicates):
            rng = np.random.default_rng(
                config.seed + 7000 + replicate + int(temporal_overlap * 1000)
            )
            fe_error, emission_error, cond = evaluate(
                base_maps,
                base_responses,
                config.detector,
                rng,
                temporal_overlap=temporal_overlap,
            )
            overlap_rows.extend(
                [
                    {
                        "temporal_overlap": temporal_overlap,
                        "replicate": replicate,
                        "method": "FE calibrated NNLS",
                        "mean_nrmse": fe_error,
                        "condition_number": cond,
                    },
                    {
                        "temporal_overlap": temporal_overlap,
                        "replicate": replicate,
                        "method": "Emission-filtered NNLS",
                        "mean_nrmse": emission_error,
                        "condition_number": condition_number(emission_matrix),
                    },
                ]
            )

    emission_overlap_rows: list[dict[str, Any]] = []
    original_emission_second = emission_matrix[:, 1].copy()
    first_emission_signature = emission_matrix[:, 0].copy()
    for emission_similarity in (0.0, 0.25, 0.5, 0.75, 0.9, 0.97, 0.995):
        stressed_emission = emission_matrix.copy()
        stressed_emission[:, 1] = (
            (1.0 - emission_similarity) * original_emission_second
            + emission_similarity * first_emission_signature
        )
        for replicate in range(replicates):
            rng = np.random.default_rng(
                config.seed + 8000 + replicate + int(emission_similarity * 1000)
            )
            fe_error, emission_error, cond = evaluate(
                base_maps,
                base_responses,
                config.detector,
                rng,
                emission_matrix_override=stressed_emission,
            )
            emission_overlap_rows.extend(
                [
                    {
                        "emission_signature_similarity": emission_similarity,
                        "replicate": replicate,
                        "method": "FE calibrated NNLS",
                        "mean_nrmse": fe_error,
                        "condition_number": cond,
                    },
                    {
                        "emission_signature_similarity": emission_similarity,
                        "replicate": replicate,
                        "method": "Emission-filtered NNLS",
                        "mean_nrmse": emission_error,
                        "condition_number": condition_number(stressed_emission),
                    },
                ]
            )

    throughput_rows: list[dict[str, Any]] = []
    for throughput in (0.1, 0.2, 0.35, 0.5, 0.7, 0.9, 1.0):
        for replicate in range(replicates):
            rng = np.random.default_rng(
                config.seed + 9000 + replicate + int(throughput * 1000)
            )
            fe_error, emission_error, cond = evaluate(
                base_maps,
                base_responses,
                config.detector,
                rng,
                optical_throughput=throughput,
            )
            throughput_rows.extend(
                [
                    {
                        "emission_optical_throughput": throughput,
                        "replicate": replicate,
                        "method": "FE calibrated NNLS",
                        "mean_nrmse": fe_error,
                        "condition_number": cond,
                    },
                    {
                        "emission_optical_throughput": throughput,
                        "replicate": replicate,
                        "method": "Emission-filtered NNLS",
                        "mean_nrmse": emission_error,
                        "condition_number": condition_number(emission_matrix),
                    },
                ]
            )

    similarity_frame = pd.DataFrame(similarity_rows)
    photon_frame = pd.DataFrame(photon_rows)
    bandwidth_frame = pd.DataFrame(bandwidth_rows)
    overlap_frame = pd.DataFrame(overlap_rows)
    emission_overlap_frame = pd.DataFrame(emission_overlap_rows)
    throughput_frame = pd.DataFrame(throughput_rows)
    noise_frame.to_csv(output / "noise_sweep.csv", index=False)
    separation_frame.to_csv(output / "separation_sweep.csv", index=False)
    brightness_frame.to_csv(output / "brightness_sweep.csv", index=False)
    similarity_frame.to_csv(output / "signature_similarity_sweep.csv", index=False)
    photon_frame.to_csv(output / "photon_budget_sweep.csv", index=False)
    bandwidth_frame.to_csv(output / "detector_bandwidth_sweep.csv", index=False)
    overlap_frame.to_csv(output / "temporal_overlap_sweep.csv", index=False)
    emission_overlap_frame.to_csv(output / "emission_overlap_sweep.csv", index=False)
    throughput_frame.to_csv(output / "emission_throughput_sweep.csv", index=False)

    save_sweep_plot(
        noise_frame,
        "background_counts_per_sample",
        "mean_nrmse",
        "method",
        output / "noise_sweep.png",
        "Additive background counts per temporal sample",
        "Mean NRMSE",
    )
    save_sweep_plot(
        separation_frame,
        "separation_um",
        "mean_nrmse",
        "method",
        output / "separation_sweep.png",
        "Spot separation (micrometres)",
        "Mean NRMSE",
    )
    save_sweep_plot(
        brightness_frame,
        "brightness_ratio_species2_to_species1",
        "mean_nrmse",
        "method",
        output / "brightness_sweep.png",
        "Species-2 / species-1 brightness ratio",
        "Mean NRMSE",
    )
    save_sweep_plot(
        similarity_frame,
        "signature_similarity",
        "mean_nrmse",
        "method",
        output / "signature_similarity_sweep.png",
        "Interpolation of FE species 2 signature toward species 1",
        "Mean NRMSE",
    )
    save_sweep_plot(
        photon_frame,
        "photons_per_relative_unit_per_sample",
        "mean_nrmse",
        "method",
        output / "photon_budget_sweep.png",
        "Photon scale per relative unit and temporal sample",
        "Mean NRMSE",
    )
    save_sweep_plot(
        bandwidth_frame,
        "detector_bandwidth_hz",
        "mean_nrmse",
        "method",
        output / "detector_bandwidth_sweep.png",
        "Detector bandwidth (Hz)",
        "Mean NRMSE",
    )
    save_sweep_plot(
        overlap_frame,
        "temporal_overlap",
        "mean_nrmse",
        "method",
        output / "temporal_overlap_sweep.png",
        "Pulse temporal-overlap factor",
        "Mean NRMSE",
    )
    save_sweep_plot(
        emission_overlap_frame,
        "emission_signature_similarity",
        "mean_nrmse",
        "method",
        output / "emission_overlap_sweep.png",
        "Interpolation of emission species 2 signature toward species 1",
        "Mean NRMSE",
    )
    save_sweep_plot(
        throughput_frame,
        "emission_optical_throughput",
        "mean_nrmse",
        "method",
        output / "emission_throughput_sweep.png",
        "Conventional emission-channel optical throughput",
        "Mean NRMSE",
    )

    summary = {
        "output_dir": str(output),
        "replicates": replicates,
        "files": [
            "noise_sweep.csv",
            "separation_sweep.csv",
            "brightness_sweep.csv",
            "signature_similarity_sweep.csv",
            "photon_budget_sweep.csv",
            "detector_bandwidth_sweep.csv",
            "temporal_overlap_sweep.csv",
            "emission_overlap_sweep.csv",
            "emission_throughput_sweep.csv",
        ],
    }
    save_json(output / "sweep_summary.json", summary)
    return summary
