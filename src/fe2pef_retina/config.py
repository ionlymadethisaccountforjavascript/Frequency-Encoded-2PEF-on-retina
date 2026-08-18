"""Configuration models and validation for the RetinaFE simulation package."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import warnings

import yaml


@dataclass(frozen=True)
class GridConfig:
    """Describe the spatial grid used for a retinal phantom.

    What happens in this class:
    1. ``shape`` gives the number of samples along each spatial dimension.
    2. ``pixel_size_um`` converts array indices into micrometres.
    3. The same class supports 2-D images and small 3-D proof-of-concept volumes.
    """

    shape: tuple[int, ...] = (96, 96)
    pixel_size_um: float = 1.0


@dataclass(frozen=True)
class LaserConfig:
    """Store one excitation laser and its intensity-modulation parameters.

    What happens in this class:
    1. The optical wavelength and average power define the excitation pathway.
    2. The modulation frequency and depth define the frequency tag.
    3. The phase is used by the time-domain simulator and lock-in references.
    """

    wavelength_nm: float
    average_power_mw: float
    modulation_frequency_hz: float
    modulation_depth: float = 0.8
    phase_rad: float = 0.0


@dataclass(frozen=True)
class PulseConfig:
    """Store pulse-train parameters shared by the two excitation lasers.

    What happens in this class:
    1. Repetition rate converts average power into pulse energy.
    2. Pulse width converts pulse energy into an approximate peak power.
    3. Temporal overlap scales the mixed two-colour excitation pathway.
    """

    repetition_rate_hz: float = 80e6
    pulse_width_fs: float = 100.0
    temporal_overlap: float = 0.9


@dataclass(frozen=True)
class PSFConfig:
    """Describe the Gaussian point-spread function used by the forward model.

    What happens in this class:
    1. ``sigma_xy_um`` controls lateral blur in the retinal plane.
    2. ``sigma_z_um`` controls axial blur when a 3-D volume is supplied.
    3. Values should ideally come from a bead measurement or a validated model.
    """

    sigma_xy_um: float = 1.5
    sigma_z_um: float = 4.0


@dataclass(frozen=True)
class DetectorConfig:
    """Store detector, acquisition, and noise parameters.

    What happens in this class:
    1. Dwell time and sample rate determine the number of temporal samples.
    2. Photon scale maps relative fluorescence into detected counts per sample.
    3. Background, dark counts, and read noise create the main noise sources.
    4. Bandwidth attenuates high-frequency lock-in channels.
    5. Saturation optionally clips simulated detector samples.
    """

    dwell_time_s: float = 40e-6
    sample_rate_hz: float = 50e6
    photons_per_relative_unit_per_sample: float = 0.35
    background_counts_per_sample: float = 0.04
    dark_counts_per_sample: float = 0.002
    read_noise_std_counts: float = 0.35
    detector_bandwidth_hz: float = 12e6
    saturation_counts: float | None = None
    relative_intensity_noise_std: float = 0.0


@dataclass(frozen=True)
class PhantomConfig:
    """Control the synthetic retinal morphology generator.

    What happens in this class:
    1. ``kind`` selects a two-spot, overlap, RPE-mosaic, or 3-D phantom.
    2. Cell and granule counts control morphological complexity.
    3. ``overlap_fraction`` controls how much structure is shared by species.
    4. ``separation_um`` is used by two-spot resolution benchmarks.
    """

    kind: str = "rpe_mosaic"
    n_cells: int = 18
    granules_per_cell: int = 14
    overlap_fraction: float = 0.35
    separation_um: float = 8.0


@dataclass(frozen=True)
class UnmixingConfig:
    """Store inversion and blind-NMF settings.

    What happens in this class:
    1. ``components`` sets the number of fluorophore maps to recover.
    2. L1 and L2 penalties follow the regularized NNMF idea in Heuke et al.
    3. Multiple restarts reduce the chance of a poor local NMF solution.
    4. The tolerance and iteration count control convergence.
    """

    components: int = 2
    alpha_l1: float = 0.01
    alpha_l2: float = 0.01
    iterations: int = 1000
    restarts: int = 4
    tolerance: float = 1e-7


@dataclass(frozen=True)
class SimulationConfig:
    """Store numerical choices for forward simulation.

    What happens in this class:
    1. ``mode`` chooses a fast analytic lock-in approximation or direct time series.
    2. ``channels`` chooses which demodulation outputs are generated.
    3. ``chunk_pixels`` limits memory use during direct time-domain simulation.
    4. ``subtract_known_background`` models a calibrated dark/background frame.
    """

    mode: str = "analytic"
    channels: tuple[str, ...] = ("dc", "f1", "f2", "difference")
    chunk_pixels: int = 512
    subtract_known_background: bool = True


@dataclass(frozen=True)
class BaselineConfig:
    """Describe the conventional emission-filtered comparison.

    What happens in this class:
    1. ``emission_matrix_csv`` stores detector-channel mixing coefficients.
    2. ``optical_throughput`` models photons lost to filters and splitting.
    3. The same photon-scale and detector-noise framework is then reused.
    """

    emission_matrix_csv: str = "data/example_emission_mixing.csv"
    optical_throughput: float = 0.35


@dataclass(frozen=True)
class ProjectConfig:
    """Collect every configuration block required by one experiment.

    What happens in this class:
    1. Nested dataclasses keep physical, numerical, and morphology parameters separate.
    2. Relative data paths are resolved from the project root by experiment code.
    3. ``validate`` catches frequencies, shapes, and settings that cannot be interpreted.
    """

    name: str
    seed: int
    species: tuple[str, ...]
    pathway_responses_csv: str
    grid: GridConfig
    laser1: LaserConfig
    laser2: LaserConfig
    pulse: PulseConfig = field(default_factory=PulseConfig)
    psf: PSFConfig = field(default_factory=PSFConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    phantom: PhantomConfig = field(default_factory=PhantomConfig)
    unmixing: UnmixingConfig = field(default_factory=UnmixingConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    baseline: BaselineConfig = field(default_factory=BaselineConfig)

    def validate(self) -> None:
        """Validate physical and numerical settings before a simulation starts.

        What happens in this function:
        1. It checks positive grid, time, frequency, and power values.
        2. It checks that modulation depths and overlap lie in valid intervals.
        3. It checks that enough modulation cycles occur during one pixel dwell.
        4. It emits warnings for weak experimental designs instead of hiding them.
        """

        if len(self.grid.shape) not in (2, 3) or any(v <= 0 for v in self.grid.shape):
            raise ValueError("grid.shape must contain two or three positive integers")
        if self.grid.pixel_size_um <= 0:
            raise ValueError("grid.pixel_size_um must be positive")
        if self.detector.dwell_time_s <= 0 or self.detector.sample_rate_hz <= 0:
            raise ValueError("dwell time and sample rate must be positive")
        if self.detector.sample_rate_hz <= 2 * max(
            self.laser1.modulation_frequency_hz,
            self.laser2.modulation_frequency_hz,
        ):
            raise ValueError("sample rate must exceed the Nyquist rate of both tags")
        for label, laser in (("laser1", self.laser1), ("laser2", self.laser2)):
            if laser.wavelength_nm <= 0 or laser.average_power_mw <= 0:
                raise ValueError(f"{label} wavelength and average power must be positive")
            if laser.modulation_frequency_hz <= 0:
                raise ValueError(f"{label} modulation frequency must be positive")
            if not 0 <= laser.modulation_depth <= 1:
                raise ValueError(f"{label} modulation depth must be between 0 and 1")
            cycles = laser.modulation_frequency_hz * self.detector.dwell_time_s
            if cycles < 5:
                warnings.warn(
                    f"{label} completes only {cycles:.2f} cycles per pixel dwell; "
                    "lock-in separation will be unstable or acquisition will be very slow.",
                    RuntimeWarning,
                )
        if not 0 <= self.pulse.temporal_overlap <= 1:
            raise ValueError("pulse.temporal_overlap must be between 0 and 1")
        if len(self.species) != self.unmixing.components:
            raise ValueError("the number of species must equal unmixing.components")
        if self.simulation.mode not in {"analytic", "time_domain"}:
            raise ValueError("simulation.mode must be 'analytic' or 'time_domain'")
        allowed = {"dc", "f1", "f2", "difference", "sum", "2f1", "2f2"}
        unknown = set(self.simulation.channels) - allowed
        if unknown:
            raise ValueError(f"unknown lock-in channels: {sorted(unknown)}")


def _tupleify(value: Any) -> Any:
    """Convert YAML lists into tuples where immutable dataclasses expect tuples.

    What happens in this function:
    1. Lists are recursively converted into tuples.
    2. Dictionaries are recursively traversed without changing their keys.
    3. Scalar values pass through unchanged.
    """

    if isinstance(value, list):
        return tuple(_tupleify(item) for item in value)
    if isinstance(value, dict):
        return {key: _tupleify(item) for key, item in value.items()}
    return value


def load_config(path: str | Path) -> ProjectConfig:
    """Load a YAML experiment file into validated nested dataclasses.

    What happens in this function:
    1. YAML is parsed from disk and lists are converted to tuples.
    2. Each nested dictionary is passed to its matching dataclass constructor.
    3. The resulting project configuration is validated before it is returned.
    """

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = _tupleify(yaml.safe_load(handle))

    config = ProjectConfig(
        name=raw["name"],
        seed=int(raw.get("seed", 0)),
        species=tuple(raw["species"]),
        pathway_responses_csv=raw["pathway_responses_csv"],
        grid=GridConfig(**raw.get("grid", {})),
        laser1=LaserConfig(**raw["laser1"]),
        laser2=LaserConfig(**raw["laser2"]),
        pulse=PulseConfig(**raw.get("pulse", {})),
        psf=PSFConfig(**raw.get("psf", {})),
        detector=DetectorConfig(**raw.get("detector", {})),
        phantom=PhantomConfig(**raw.get("phantom", {})),
        unmixing=UnmixingConfig(**raw.get("unmixing", {})),
        simulation=SimulationConfig(**raw.get("simulation", {})),
        baseline=BaselineConfig(**raw.get("baseline", {})),
    )
    config.validate()
    return config
