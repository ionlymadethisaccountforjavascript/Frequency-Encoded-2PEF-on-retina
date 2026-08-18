# Scientific scope and defensible claims

## The paper this code can support

A strong title would be:

> **Computational evaluation of frequency-encoded two-photon excited fluorescence for retinal autofluorescence unmixing**

The central question should be:

> Under which combinations of excitation-signature separation, additive noise, shot noise, spatial overlap, and brightness imbalance does FE-2PEF recover retinal fluorophore maps more accurately than conventional emission-filtered TPEF?

That is a testable and balanced question. The framework is designed to reveal both favourable and unfavourable regimes.

## Claims the current code can support after real calibration

- The FE-2PEF forward model can be implemented for retinal morphology phantoms.
- Lock-in channel equations agree with direct time-domain simulation.
- Calibrated and blind nonnegative unmixing can recover species maps under specified simulated conditions.
- Performance depends strongly on signature-matrix conditioning, background, brightness imbalance, and photon budget.
- A retinal application is computationally plausible and motivates experimental validation.

## Claims the current code cannot support

- FE-2PEF improves optical spatial resolution or beats the diffraction limit.
- A2E and whole lipofuscin are two cleanly independent fluorophores.
- A simulated map diagnoses AMD, glaucoma, or any other disease.
- One milliwatt is automatically safe for a human eye.
- The included placeholder pathway responses are real A2E/FAD two-photon cross-sections.
- Blind NNMF always produces zero crosstalk.
- FE-2PEF always outperforms a multi-detector system.

## A2E and lipofuscin

A2E is a bisretinoid component of RPE lipofuscin. Whole lipofuscin is a heterogeneous mixture whose excitation and emission vary with composition and irradiation. Treating “A2E” and “lipofuscin” as two simple pure dyes can make the inverse problem biologically misleading.

The repository therefore uses:

- `A2E_like` and `FAD_like` for the main software demonstration;
- `A2E_like` and `Lipofuscin_residual_like` as an intentionally ill-conditioned stress test.

A publishable A2E-versus-background study should define the second component as a measured **non-A2E residual lipofuscin signature**, not generic lipofuscin.

## Spatial localization versus fluorophore discrimination

Raster scanning already assigns a spatial coordinate to every pixel. Frequency encoding supplies additional molecular contrast by separating excitation signatures. It may improve the accuracy of a reconstructed species location when signals overlap, but that is not the same as increasing the optical resolution of the microscope.

The separation sweep therefore reports reconstruction error versus spot spacing. It should be described as an **unmixing/localization benchmark**, not a super-resolution result.

## Disease relevance

RPE autofluorescence and bisretinoid changes are relevant to retinal ageing and disease research. A disease claim, however, needs biological samples, appropriate controls, clinical labels, and independent validation. The simulation should be presented as an imaging-method study with a future biomedical application, not a diagnostic system.
