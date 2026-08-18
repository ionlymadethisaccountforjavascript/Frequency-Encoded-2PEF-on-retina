$ErrorActionPreference = "Stop"
# Create a local virtual environment, install the package, run tests, and execute the example.
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    py -3 -m venv .venv
}
& .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pytest -q
retinafe all --config configs/retina_a2e_fad.yaml --output outputs/main_run --project-root . --replicates 6
