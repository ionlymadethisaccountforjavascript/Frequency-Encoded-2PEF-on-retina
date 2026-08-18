"""Run the default retinal FE-2PEF demonstration from a source checkout."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fe2pef_retina.experiments import run_demo  # noqa: E402


def main() -> int:
    """Run the default demo and print the output directory.

    What happens in this function:
    1. Project-relative config and output paths are constructed.
    2. The end-to-end demo experiment is executed.
    3. A compact completion message is printed.
    4. Zero is returned for shell compatibility.
    """

    result = run_demo(
        PROJECT_ROOT / "configs" / "retina_a2e_fad.yaml",
        PROJECT_ROOT / "outputs" / "demo",
        PROJECT_ROOT,
    )
    print(f"Demo complete: {result['output_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
