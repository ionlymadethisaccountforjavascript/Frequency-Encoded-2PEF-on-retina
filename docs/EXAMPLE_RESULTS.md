# Example run: interpretation, not biological evidence

The repository includes a completed software run in `outputs/example/`. Its purpose is to verify the full workflow and expose failure modes before real calibration data are substituted.

## Important limitation

The example uses explicitly labelled, normalized **A2E-like** and **FAD-like** pathway responses and an illustrative overlapping emission matrix. These are not measured retinal two-photon cross-sections. Therefore, the values below are software-example outcomes, not publishable biological results.

## Implementation validation

The direct time-domain detector simulation and the analytic lock-in signature agree to a maximum relative error of approximately `1.02e-14` in the noiseless channel-validation test. This confirms that the two implementations use the same modulation algebra; it does not validate the assumed spectra or retinal biology.

## Main reconstruction example

For the supplied noisy RPE-like phantom:

- FE calibrated NNLS mean NRMSE: approximately `0.163`.
- FE blind regularized NNMF mean raw-scale NRMSE: approximately `0.982`.
- Emission-filtered calibrated NNLS mean NRMSE: approximately `0.988`.

The blind NNMF maps retain substantial morphology but show background and quantitative-scale ambiguity. This is intentionally reported rather than hidden. The calibrated method is the main quantitative result until pure-species calibration and stronger blind-identifiability conditions are available.

## Robustness examples

With four fixed-seed replicates per point:

- Increasing additive background from `0` to `2` counts per temporal sample increased FE mean NRMSE from about `0.099` to `0.140`.
- Interpolating the second FE signature toward the first increased FE mean NRMSE from about `0.098` to `0.703`, showing the expected conditioning failure.
- Increasing detector bandwidth from `1.8 MHz` to `20 MHz` reduced FE mean NRMSE from about `0.208` to `0.096` in this setup.
- Increasing the photon scale from `0.05` to `1.4` reduced FE mean NRMSE from about `0.305` to `0.048`.
- Making the two emission signatures increasingly similar degraded the emission-filtered comparator while leaving the FE result largely unchanged, because the FE excitation signatures were held fixed.
- Increasing conventional filter throughput improved the emission-filtered comparator. This explicitly prevents the study from assuming that FE always wins.

## How to use these outputs

Use the figures to review software logic and choose experiments. Do not copy the numerical values into a manuscript abstract. Re-run all analyses after replacing both example response tables and validating the detector model.
