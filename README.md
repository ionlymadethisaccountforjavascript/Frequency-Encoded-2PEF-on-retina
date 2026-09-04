**A simulation framework for frequency-encoded two-photon excited fluorescence (FE-2PEF) in retinal autofluorescence.**

This repository is a clean rebuild of the project concept. It implements the frequency-encoding mechanism described by Heuke *et al.*: two intensity-modulated excitation lasers, three two-photon pathways, a single detector, phase-aware lock-in channels, and nonnegative signal unmixing. It then applies that framework to transparent retinal morphology phantoms and compares it with conventional TPEF.

## Read this before using the results

The included A2E-like and FAD-like excitation responses are **illustrative normalized placeholders**, not measured two-photon cross-sections. They let the software, equations, noise analyses, and failure tests run end-to-end. They are deliberately labelled `_like` and carry provenance in the CSV files. A paper must replace them with measured or defensibly digitized pathway signatures at the selected wavelengths.

The simulation can support a paper; it does not guarantee publication. The strongest defensible contribution is a **computational evaluation of when FE-2PEF improves retinal fluorophore unmixing**, not a claim of super-resolution, clinical diagnosis, or proven ocular safety.

## What is implemented

- Degenerate pathways at `2ω1` and `2ω2`, plus the mixed pathway `ω1 + ω2`.
- Laser intensity tags at `f1` and `f2`.
- Lock-in channels at DC, `f1`, `f2`, `|f1-f2|`, and optional harmonics/sum.
- A fast analytic lock-in noise model for parameter sweeps.
- A direct, chunked time-domain detector simulation for implementation validation.
- Poisson shot noise, additive background, dark counts, read noise, detector bandwidth, and optional laser RIN.
- Gaussian 2-D and 3-D PSF convolution.
- Calibrated nonnegative least-squares unmixing.
- Blind regularized NNMF using L1 and L2 penalties based on the FE-2PEF paper.
- Conventional single-detector and emission-filtered multi-detector baselines.
- RPE-like mosaics, two-spot tests, overlap tests, and a small 3-D volume.
- Nine robustness studies covering background, photon budget, spatial separation, brightness imbalance, FE-signature conditioning, detector bandwidth, pulse overlap, emission-signature overlap, and conventional filter throughput.
- Automatic tests requiring a descriptive docstring for every function.

## Fast start

Windows:`run_windows.bat`; see `docs/QUICKSTART_WINDOWS.md`. macOS/Linux:`./run_example.sh`.

## Installation

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -e .
```

## Run the full example

```bash
retinafe all \
  --config configs/retina_a2e_fad.yaml \
  --output outputs/main_run \
  --project-root . \
  --replicates 6
```

Windows PowerShell uses backticks instead of backslashes, or place the command on one line.

Run each stage separately:

```bash
retinafe demo --config configs/retina_a2e_fad.yaml --output outputs/demo --project-root .
retinafe validate --config configs/time_domain_validation.yaml --output outputs/validation --project-root .
retinafe sweeps --config configs/retina_a2e_fad.yaml --output outputs/sweeps --project-root . --replicates 10
```

## Tests

```bash
pytest -q
```

The tests check channel equations, direct-versus-analytic demodulation, unmixing, reproducibility, and function documentation.

## Main configurations

- `retina_a2e_fad.yaml`: main two-species retinal study.
- `time_domain_validation.yaml`: direct waveform validation of the analytic channel model.
- `a2e_lipofuscin_stress_test.yaml`: deliberately difficult, ill-conditioned separation test.
- `rpe_volume_demo.yaml`: small 3-D software demonstration.
- `legacy_40hz_90hz_demonstration_only.yaml`: preserves the earlier low-frequency idea, but uses a long dwell because 40/90 Hz tags are not compatible with ordinary fast raster scanning.

## Output

The demo produces:

- exact numeric arrays in `arrays.npz`;
- `metrics.csv` and complete parameter provenance;
- raw FE channels;
- calibrated FE reconstruction;
- blind NNMF reconstruction;
- conventional emission-filtered reconstruction;
- condition numbers and noise diagnostics;
- figures suitable for initial internal review.

The sweeps save every replicate, not only mean curves. This is important for uncertainty estimates and statistical analysis.

[![DOI](https://zenodo.org)](https://doi.org/10.5281/zenodo.22305474)


## References

S. Heuke, C. Silva Martins, R. André, L. LeGoff, and H. Rigneault, “Frequency-encoded two-photon excited fluorescence microscopy,” *Optics Letters* **48**, 4113–4116 (2023). DOI: `10.1364/OL.496071`.

## Authors

Aarush Mandala and Youbin Duan.
