# Minimum experimental validation protocol

A simulation-only paper is possible with tightly limited claims, but even a modest optical validation would make the work substantially stronger.

## 1. Instrument characterization

Record the actual wavelengths, average powers at the sample, repetition rate, pulse width, modulation depths, tag frequencies, relative modulation phases, and temporal overlap. Measure the transfer function of the PMT, transimpedance amplifier, digitizer, and lock-in chain rather than assuming one nominal bandwidth.

## 2. Noise measurements

Acquire:

1. detector-dark recordings with lasers blocked;
2. background recordings with a nonfluorescent sample;
3. uniform fluorescent-sample recordings over several powers;
4. repeated recordings to separate Poisson-like variance, read noise, drift, and laser relative-intensity noise.

Fit the model parameters without looking at the final comparison results.

## 3. Pure-species calibration

For each fluorophore or retinal-autofluorescence fraction:

1. image a pure sample using the identical laser pair and dwell time;
2. demodulate DC, `f1`, `f2`, and `|f1-f2|`;
3. estimate the nonnegative channel-signature column and its uncertainty;
4. repeat across concentration and power to test linearity and saturation;
5. repeat after instrument realignment to quantify calibration drift.

This produces the matrix `Sigma` used by calibrated NNLS and provides an initialization/constraint for NNMF.

## 4. Known-mixture validation

Prepare mixtures with known concentration ratios, including strongly imbalanced cases. Recover the ratios without retuning parameters. Report bias, variance, limits of detection, crosstalk, and failure at saturation.

## 5. Spatial phantom

Use separated and overlapping fluorescent beads, patterned films, or a two-component tissue phantom. Keep the optical photon budget matched when comparing FE with emission-filtered TPEF. The spatial test evaluates molecular unmixing at overlapping locations; it is not a super-resolution experiment.

## 6. Retinal relevance

For an A2E study, define the comparator as a measured non-A2E residual lipofuscin fraction or another specific species. Whole lipofuscin should not be treated as a single pure dye. Cell or ex-vivo experiments require appropriate biological controls and independent expertise.

## 7. Safety

Do not infer ocular safety from average power alone. A real in-vivo protocol requires wavelength-, pulse-, pupil-, scan-, exposure-duration-, and retinal-spot-dependent hazard analysis by qualified collaborators and the applicable ethics and laser-safety review.

## 8. Preregistered comparison

Before the final run, freeze:

- primary metric;
- photon-budget matching rule;
- parameter ranges;
- excluded frames and saturation criteria;
- calibration procedure;
- statistical model and replicate count.

Report regimes in which conventional multi-detector TPEF outperforms FE-2PEF.
