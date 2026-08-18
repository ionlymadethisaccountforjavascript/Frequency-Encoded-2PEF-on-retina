@echo off
setlocal
REM Create a local virtual environment, install the package, run tests, and execute the example.
if not exist .venv\Scripts\python.exe py -3 -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pytest -q
retinafe all --config configs\retina_a2e_fad.yaml --output outputs\main_run --project-root . --replicates 6
endlocal
