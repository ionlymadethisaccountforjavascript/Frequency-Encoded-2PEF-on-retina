# Model equations

## 1. Fluorophore maps

For species `s`, the unknown nonnegative concentration map is

\[
C_s(\mathbf r) \ge 0.
\]

The maps are inputs to the optical model. Illumination creates fluorescence; it does **not** create the underlying fluorophore concentration.

## 2. Spatial imaging model

A Gaussian PSF approximates combined excitation and collection blur:

\[
\widetilde C_s(\mathbf r)=h(\mathbf r)*C_s(\mathbf r).
\]

The code supports 2-D and small 3-D arrays. For a paper, replace the Gaussian with a measured PSF or a validated aberrated-eye model.

## 3. Intensity modulation

The two average laser intensities are represented as

\[
I_1(t)=P_1[1+m_1\sin(2\pi f_1t+\phi_1)],
\]

\[
I_2(t)=P_2[1+m_2\sin(2\pi f_2t+\phi_2)].
\]

`m1` and `m2` lie between zero and one. The pulse train is represented through cycle-averaged pathway coefficients; the mixed pathway is multiplied by temporal-overlap factor `rho`.

## 4. Three excitation pathways

Each species has nonnegative responses to

- `sigma_11`: two photons from laser 1;
- `sigma_12`: one photon from each laser;
- `sigma_22`: two photons from laser 2.

The ideal relative fluorescence at one location is

\[
F(t)=\sum_s \widetilde C_s
\left[
\sigma_{s,11}I_1^2(t)
+2\rho\sigma_{s,12}I_1(t)I_2(t)
+\sigma_{s,22}I_2^2(t)
\right].
\]

This implements the physical structure of the 2023 FE-2PEF paper. The numerical pathway responses supplied with the repository are placeholders.

## 5. Analytic lock-in channels

Expanding the modulation gives the following positive phase-aware amplitudes for each species:

\[
S_{DC}=\sigma_{11}P_1^2(1+m_1^2/2)
+\sigma_{22}P_2^2(1+m_2^2/2)
+2\rho\sigma_{12}P_1P_2,
\]

\[
S_{f_1}=2m_1(\sigma_{11}P_1^2+\rho\sigma_{12}P_1P_2),
\]

\[
S_{f_2}=2m_2(\sigma_{22}P_2^2+\rho\sigma_{12}P_1P_2),
\]

\[
S_{|f_1-f_2|}=\rho\sigma_{12}P_1P_2m_1m_2.
\]

Optional `2f1`, `2f2`, and `f1+f2` channels are implemented. The default follows the practical four-channel selection in Heuke *et al.*, because their second-harmonic channels were experimentally noisy.

## 6. Linear mixing model

After demodulation, every pixel follows

\[
\mathbf S = \mathbf\Sigma\mathbf C + \boldsymbol\epsilon,
\]

where `S` is the channel vector, `Sigma` is the channel-by-species signature matrix, `C` is the nonnegative species concentration vector, and `epsilon` is noise.

A known calibration matrix is inverted using nonnegative least squares. Blind recovery uses regularized NNMF.

## 7. Regularized NNMF

The implemented objective is

\[
\min_{\Sigma,C\ge0}
\|S-\Sigma C\|_F^2
+\alpha_1\|C\|_1
+\alpha_2\|C\|_F^2.
\]

The multiplicative updates follow the structure reported in the FE-2PEF article. Columns of `Sigma` are normalized at each iteration to control scale ambiguity.

## 8. Noise

The direct simulator samples Poisson photon counts and adds Gaussian read noise before lock-in projection.

The fast analytic simulator uses the approximate lock-in variance

\[
\operatorname{Var}(\widehat A_{AC})\approx
\frac{2(\mu_{DC}+\sigma_r^2)}{N},
\]

and

\[
\operatorname{Var}(\widehat A_{DC})\approx
\frac{\mu_{DC}+\sigma_r^2}{N},
\]

where `N` is temporal samples per pixel, `mu_DC` is mean detected count per sample, and `sigma_r` is read noise.

## 9. Detector bandwidth

Each channel amplitude is multiplied by a first-order response

\[
|H(f)|=[1+(f/f_c)^2]^{-1/2}.
\]

The actual PMT, amplifier, digitizer, and lock-in transfer function should replace this approximation.

## 10. Conventional comparator

The emission-filtered baseline uses

\[
\mathbf D=\eta\mathbf M\mathbf C+\boldsymbol\epsilon,
\]

where `M` is a detector-by-species emission matrix and `eta` is optical throughput. This makes it possible to test the published claim that FE can benefit additive-noise-limited conditions while acknowledging shot-noise and brightness-imbalance disadvantages.
