# Run verification

Verification date: 2026-08-06

## Automated tests

`pytest -q -p no:cacheprovider`: **8 passed**.

The test suite checks optical calculations, analytic-versus-direct lock-in signatures, nonnegative unmixing, deterministic seeded output, and the requirement that every source function contains a detailed “What happens” docstring.

## Channel-equation validation

- Maximum relative analytic/direct error: `1.0210278541131697e-14`.
- Mean relative analytic/direct error: `1.3553548006995797e-15`.
- Temporal samples per dwell: `2000`.

## Smoke-tested configurations

- `retina_a2e_fad.yaml`
- `time_domain_validation.yaml`
- `a2e_lipofuscin_stress_test.yaml`
- `rpe_volume_demo.yaml`
- `legacy_40hz_90hz_demonstration_only.yaml`

The CLI `all` workflow, nine robustness sweeps, the small 3-D demo, and both stress-test demos completed successfully.

## Packaging

- Wheel: `dist/retinafe-0.1.0-py3-none-any.whl`
- Wheel SHA-256: `48fa59668e469c3adc1b8a237aae33b01c1ddf0198ce4c4b8a8dd855972bcf2d`
- Documented source/script functions: `74`

## Scientific status

Software verification confirms internal consistency, not retinal biological validity. The included spectral tables remain explicit placeholders and must be replaced before manuscript claims are finalized.
