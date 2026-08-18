# Source notes

## Frequency-encoded 2PEF

Heuke *et al.* use two excitation wavelengths, distinct intensity-modulation frequencies, one detector, simultaneous demodulation, and nonnegative matrix factorization. Their practical four-channel analysis uses DC, `f1`, `f2`, and `f1-f2`. The paper also explicitly describes disadvantages: shared shot noise, shared detector dynamic range, and difficulty when fluorophore distributions overlap with large brightness differences.

This repository implements those advantages and disadvantages rather than assuming FE always wins.

## Retinal project presentation

The project presentation proposed a retinal patch, two modulated lasers, Poisson/background noise, lock-in demodulation, NNMF, RPE morphology, A2E/FAD or A2E/lipofuscin targets, and possible 3-D scaling. The rebuild preserves those goals but corrects the following:

- MHz tags are used for realistic fast dwell times; 40/90 Hz remains only as a legacy demonstration.
- A2E is not treated as independent from whole lipofuscin.
- KerNet is used only as a possible morphology prior.
- “clearer” is replaced by quantitative NRMSE, SSIM, correlation, and localization error.
- Safety and disease-detection claims are excluded from the computational conclusion.
