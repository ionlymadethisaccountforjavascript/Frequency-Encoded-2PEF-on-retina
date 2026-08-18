# Reproducibility protocol

## Fixed elements

- Every configuration contains an explicit random seed.
- Every run saves its configuration and response tables.
- Exact numeric arrays are stored in compressed NPZ files.
- Sweeps save every replicate to CSV.
- Figure generation is separated from numeric results.
- Unit tests compare direct waveform demodulation with analytic channel equations.

## Recommended paper workflow

1. Create a versioned release before final analysis.
2. Freeze a calibration CSV and record its acquisition date.
3. Run at least 20 independent noise replicates per condition.
4. Report means, standard deviations, confidence intervals, and all excluded runs.
5. Publish raw calibration frames or a justified derived signature matrix.
6. Include a hardware schematic and complete detector transfer function.
7. Deposit code, configs, and data with a DOI-bearing repository.

## Randomness

Seeds make the examples exactly repeatable. They do not replace uncertainty analysis. A publishable result should repeat over many seeds and report the distribution.
