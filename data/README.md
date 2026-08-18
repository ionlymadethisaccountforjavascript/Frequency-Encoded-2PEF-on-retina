# Data and calibration policy

The two CSV files in this directory are **software test inputs**, not publishable biological measurements.

## `example_pathway_responses.csv`

Each species has normalized responses for:

- `sigma_11`: degenerate two-photon pathway driven by laser 1;
- `sigma_12`: mixed pathway using one photon from each laser;
- `sigma_22`: degenerate pathway driven by laser 2;
- `brightness`: relative concentration/quantum-yield scaling.

The included rows are labelled `illustrative_normalized_placeholder_not_measured`. Do not remove that label unless the numbers are replaced by measured or defensibly digitized values with uncertainty.

For a real experiment, prepare pure A2E, FAD, or other reference samples; record FE channels under the same laser powers, alignment, pulse overlap, detector gain, and dwell time as the tissue scan; then estimate the channel signature matrix from those calibration images.

## `example_emission_mixing.csv`

This file defines a conventional two-detector baseline with deliberately overlapping emission channels. Replace it with filter transmission, detector quantum efficiency, fluorophore emission spectra, and optical throughput from the actual comparator system.

## KerNet and retinal images

KerNet can supply cell morphology or segmentation priors, but keratin fluorescence is not an A2E/FAD concentration map. Do not relabel a morphology dataset as molecular ground truth. See `docs/KERNET_INTEGRATION.md`.
