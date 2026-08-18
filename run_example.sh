#!/usr/bin/env bash
set -euo pipefail
# Create a local virtual environment, install the package, run tests, and execute the example.
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pytest -q
retinafe all --config configs/retina_a2e_fad.yaml --output outputs/main_run --project-root . --replicates 6
