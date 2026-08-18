"""Run validation, main demo, and robustness sweeps from a source checkout."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fe2pef_retina.experiments import (  # noqa: E402
    run_channel_validation,
    run_demo,
    run_robustness_sweeps,
)


def main() -> int:
    """Execute every included analysis using the default configurations.

    What happens in this function:
    1. Channel-equation validation is run first.
    2. The main retinal reconstruction comparison is run second.
    3. Robustness sweeps are run with six random replicates per condition.
    4. Output subdirectories are printed for inspection.
    """

    output_root = PROJECT_ROOT / "outputs" / "full_run"
    validation = run_channel_validation(
        PROJECT_ROOT / "configs" / "time_domain_validation.yaml",
        output_root / "validation",
        PROJECT_ROOT,
    )
    demo = run_demo(
        PROJECT_ROOT / "configs" / "retina_a2e_fad.yaml",
        output_root / "demo",
        PROJECT_ROOT,
    )
    sweeps = run_robustness_sweeps(
        PROJECT_ROOT / "configs" / "retina_a2e_fad.yaml",
        output_root / "sweeps",
        PROJECT_ROOT,
        replicates=6,
    )
    print("Validation:", validation)
    print("Demo:", demo)
    print("Sweeps:", sweeps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
