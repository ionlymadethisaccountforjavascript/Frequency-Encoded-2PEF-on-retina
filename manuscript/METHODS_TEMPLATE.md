# Methods template

## Study design

State the primary comparison, preregistered outcomes, species pair, laser wavelengths, and calibration procedure.

## Retinal morphology

Describe whether the maps are synthetic, dataset-derived morphology priors, phantom measurements, or tissue images. Do not present morphology as molecular ground truth without evidence.

## Excitation model

Report `P1`, `P2`, wavelengths, pulse duration, repetition rate, modulation depths, frequencies, phases, and temporal overlap. Define all three pathway responses and their uncertainty.

## Detection model

Report PMT/detector model, collection path, optical throughput, sampling rate, dwell time, bandwidth, gain, dark counts, background, read noise, saturation, and digitization.

## Demodulation

List channels and reference phases. Report finite-window cycles and reference Gram matrix or leakage analysis.

## Unmixing

Describe calibrated NNLS and regularized NNMF, initialization, restarts, stopping criteria, L1/L2 penalties, and component alignment used only for simulated evaluation.

## Conventional comparator

Report filters, detector quantum efficiency, throughput, number of detectors, calibration matrix, and photon-budget assumptions.

## Metrics and statistics

Report NRMSE, correlation, SSIM, localization error, condition number, number of random replicates, confidence intervals, and multiple-comparison handling.

## Reproducibility

Provide software version, configuration files, random seeds, calibration files, data DOI, and exact commands.
