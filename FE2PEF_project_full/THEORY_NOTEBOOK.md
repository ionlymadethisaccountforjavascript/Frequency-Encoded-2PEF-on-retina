THEORY_NOTEBOOK - Frequency-Encoded Two-Photon Excitation Fluorescence (FE-2PEF)
==============================================================================

1) Basic two-photon excitation terms
-----------------------------------
For two lasers at optical angular frequencies ω1 and ω2 intensity-modulated at f1 and f2,
with electric fields E1(t) and E2(t), the instantaneous intensity relevant for two-photon
absorption (assuming classical intensity-dependent absorption) contains terms proportional to:
    - I1^2 ~ |E1|^4 -> produces modulation at DC and harmonics 2*f1
    - I2^2 ~ |E2|^4 -> produces modulation at DC and harmonics 2*f2
    - I1*I2 ~ |E1|^2|E2|^2 -> mixing term, contains components at frequencies f1-f2, f1+f2, and combinations
For intensity modulation with sinusoidal amplitude modulation A1 cos(2π f1 t), A2 cos(2π f2 t), the 2PEF
signal S(t) can be written (simplified):
    S(t) = α1 [1 + A1 cos(2π f1 t)]^2 + α2 [1 + A2 cos(2π f2 t)]^2 + α12 [1 + A1 cos(2π f1 t)] [1 + A2 cos(2π f2 t)]
Expand and keep only time-varying terms; cross-terms give rise to components at f1, f2, 2f1, 2f2, f1±f2.

2) Lock-in demodulation mapping
------------------------------
- Demod at 2f1 -> isolates |E1|^4 (pure two-photon from laser1)
- Demod at 2f2 -> isolates |E2|^4 (pure two-photon from laser2)
- Demod at f1+f2 or |f1-f2| -> isolates cross-mixing term proportional to E1^2*E2^2 (two-color excitation)
- Demod at f1 or f2 contains contributions from mixing with DC but is less isolated.

3) NNMF demixing
-----------------
After demodulating at the chosen frequencies for each pixel/timepoint, we obtain a small spectrum of amplitudes.
Stack these demod amplitudes into matrix M (pixels x channels). Use nonnegative matrix factorization (NNMF)
to factor M ≈ W H where columns of W are spatial maps (fluorophore abundances) and H are per-channel responses.
With careful calibration of reference spectra (from single-fluorophore samples) we can convert H to concentrations.

4) PSF, NA and collection model (practical)
-------------------------------------------
- Use a 2D Gaussian PSF (or 3D if you embed retinasim volumes) with width w determined by NA and wavelength:
    w ~ 0.61 * λ / NA  (Airy disc approximate)
- Collection efficiency & scattering: apply an exponential attenuation with depth and a scattering kernel convolved to the emitted photons.
- Emission discrimination simulation: simulate spectral filters by multiplying emission spectra with filter transmission curves (0..1). Photons outside passband are lost.

5) SNR & throughput comparison
------------------------------
- Simulate photon counts per pixel for each fluorophore, apply Poisson shot noise, add detector dark noise (Gaussian),
  and compute SNR = mean_signal / std_noise.
- Compare FE-2PEF (all photons collected, demodulate and unmix) vs filter-based emission discrimination (photons lost by filters).
- Report improvement in SNR and throughput as % gain under varying conditions (photon budget, cross-talk).

6) Implementation notes
-----------------------
- run_demo.py provides a minimal end-to-end simulation using a 2D map of fluorophores and a small PSF model.
- Replace stimulation.py with full retinasim stimulation.py for realistic vascular/retina geometry and 3D embedding.
- The demod math above is kept in the notebook and derivations; for publication-ready derivations, expand symbolic expansion using sympy.
