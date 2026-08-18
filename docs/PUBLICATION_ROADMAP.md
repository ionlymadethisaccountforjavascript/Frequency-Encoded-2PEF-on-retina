# Eight-to-twelve-week publication roadmap

## Weeks 1-2: lock the scientific question

- Use fluorophore **unmixing**, not super-resolution, as the primary claim.
- Choose the species pair and laser wavelengths only after obtaining excitation data.
- Decide whether the paper is purely computational or includes a phantom/ex-vivo validation.
- Email the FE-2PEF authors and Jonathan Daniel with the forward model, one figure, and precise questions.

Deliverables: one-page study design, frozen outcomes, and a parameter-provenance table.

## Weeks 2-4: replace placeholders

- Measure or digitize pathway responses at `2ω1`, `ω1+ω2`, and `2ω2`.
- Characterize modulation depth, pulse overlap, detector bandwidth, dark signal, and read noise.
- Replace the example emission baseline with an actual filter/detector model.
- Validate PSF against a bead measurement or a documented optical model.

Deliverables: calibrated `pathway_responses.csv`, `emission_mixing.csv`, and uncertainty estimates.

## Weeks 4-6: complete validation

- Run direct-versus-analytic channel validation.
- Test pure-species samples and known mixtures.
- Verify concentration linearity and dynamic range.
- Add phase drift, power drift, and saturation stress tests if present in hardware.

Deliverables: calibration plots, residual diagnostics, and a locked analysis script.

## Weeks 6-8: main experiments

Run predeclared sweeps for:

- additive background;
- photon budget;
- spatial overlap;
- species brightness ratio;
- signature condition number;
- detector bandwidth;
- pulse-overlap mismatch;
- calibrated NNLS versus blind NNMF;
- FE versus conventional emission filtering.

Report failure regimes. A balanced result is more publishable than a simulation designed to always win.

## Weeks 8-10: manuscript and external review

- Draft Methods directly from `docs/MODEL_EQUATIONS.md`.
- Build the paper around one main comparison and two robustness figures.
- Send the draft to an optics researcher and a retinal-imaging specialist.
- Correct terminology, claims, and parameter provenance.

## Weeks 10-12: preprint and submission

- Freeze code and data release.
- Upload a preprint if coauthors and mentors approve.
- Submit to a realistic journal chosen after the final level of validation.
- For applications, accurately state “preprint” or “under review”; do not imply acceptance.

## Minimum evidence tiers

**Tier A - simulation only:** method framework, strong uncertainty analysis, carefully limited claims.

**Tier B - calibrated optical phantom:** substantially stronger; validates signal formation and unmixing.

**Tier C - ex-vivo/cell data:** best route for a biomedical-optics journal.

The code is designed to take the project from Tier A toward Tier B. It cannot create biological validation by itself.
