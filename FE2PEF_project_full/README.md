
FE2PEF_project_full
===================

This archive is a project scaffold for a Frequency-Encoded Two-Photon Excited Fluorescence (FE-2PEF) simulation and demo
that you asked me to assemble from the code fragments and repositories you provided in the chat.

What is included (scaffold & runnable demo)
-------------------------------------------
- src/fe2pef/run_demo.py        : A runnable demonstration script that simulates FE-2PEF and emission-discrimination imaging on a 2D retina plane.
- src/fe2pef/optics.py         : Simple optical model (PSF, NA, scattering approximation).
- src/fe2pef/aotf_sim.py       : AOTF / laser-channel simulation with per-channel amplitude and digital modulation control.
- src/fe2pef/demod.py          : Demodulation (lock-in style) utilities using FFT-based extraction of modulated components.
- src/fe2pef/nnmf_demo.py      : Simple NNMF wrapper to unmix channels (uses sklearn if available, fallback simple NMF implementation included).
- src/fe2pef/stimulation_wrapper.py : A wrapper that tries to import & use third_party.retinasim Simulation (if present) or falls back to a simple 2D retina generator.
- third_party/retinasim/       : PLACEHOLDER directory. **Drop the real retinasim repository here** (the ls -R you provided suggests many files). The demo will detect it and use it.
- NOTE_FILES/TPA_station_snippets.txt : A file containing the TPA GUI code snippets you pasted in the chat (for your convenience).

Why there are placeholders
--------------------------
Your retinasim tree is large and you pasted many files; to avoid missing any file or introducing transcription errors I included a placeholder for the third_party/retinasim directory. Please copy your full retinasim tree into third_party/retinasim/ inside this archive (or extract your local copy into that path). The included stimulation_wrapper will try to import retinasim once it's placed there.

How to run the demo
-------------------
1. Extract or place the retinasim repository into third_party/retinasim/ inside the zip (so path third_party/retinasim/ contains the Python package).
2. From the project root run:
   python3 -m src.fe2pef.run_demo

This will run a short simulation that:
- builds a retina plane (or uses retinasim if available)
- populates several fluorophores in different retinal regions
- simulates two lasers with different wavelengths & modulation frequencies (FE approach)
- simulates naive emission-discrimination (spectral filters) for comparison
- demodulates FE signals and runs an NNMF unmix to reconstruct separated images
- reports SNR / throughput metrics and writes figures to `output/`

Notes / limitations
-------------------
- This scaffold is intentionally conservative. The full retinasim package is NOT included here to keep the archive small. Drop your retinasim folder into third_party/retinasim and the demo will use it.
- The physics models (PSF, scattering) are simplified but parameterized; you can improve realism by editing src/fe2pef/optics.py.
- The AOTF/laser specs used in the demo are realistic examples (wavelengths, pulse durations) but please change them to your true hardware specs as needed.
- The demo and code include references to the TPA GUI snippets you pasted in the chat; the full GUI sources are included as a text file to paste into your working copy.

If you'd like, I can now:
- include the full retinasim tree if you upload it here (I will merge it under third_party/retinasim/),
- or attempt a more complete integration using the stimulation.py you pasted (paste more files or confirm you already gave the entire file).
