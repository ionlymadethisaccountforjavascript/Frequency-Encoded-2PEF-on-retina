THIRD_PARTY_README - Integrating retinasim
------------------------------------------
You told me you have a 'retinasim' folder (ls -R output posted in chat).
To integrate it into this demo:
  1) Place your retinasim folder at:
        <project_root>/third_party/retinasim
  2) Ensure Python path includes that location, or install retinasim into your venv:
        pip install -e third_party/retinasim
  3) Replace the partial stimulation.py included here with the full file from retinasim (paste the rest of the file into stimulation.py).
  4) run_demo.py will detect the retinasim package and use it to sample realistic fluorophore maps.
