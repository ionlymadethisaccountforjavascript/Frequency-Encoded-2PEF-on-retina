# Windows quick start

## One-click command prompt route

1. Install Python 3.10 or newer and select **Add Python to PATH** during installation.
2. Extract the ZIP.
3. Double-click `run_windows.bat`.
4. Generated figures and CSV files will appear in `outputs/main_run/`.

## PowerShell route

From the extracted project folder:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\run_example.ps1
```

## Manual route

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pytest -q
retinafe all --config configs/retina_a2e_fad.yaml --output outputs/main_run --project-root . --replicates 6
```

The first installation needs internet access to download Python dependencies. Later runs can use the existing `.venv`.
